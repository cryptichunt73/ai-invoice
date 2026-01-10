import os
import re
import json
import time
import hashlib
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from flask import Flask, render_template, request, jsonify, send_file, abort
from jinja2 import Environment, StrictUndefined, select_autoescape
from google import genai
from google.genai import types

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
# Optional OpenAI (only required if you want AI-generated templates)
USE_OPENAI = True
try:
    from openai import OpenAI
    from pydantic import BaseModel, Field
except Exception:
    USE_OPENAI = False

# ----------------------------
# App + Storage
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage" / "companies"
EXPORT_DIR = BASE_DIR / "storage" / "exports"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-2024-08-06")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

def _extract_json(text: str):
    """
    Robustly extracts the FIRST JSON value from a string, even if extra text follows.
    Handles: trailing citations, tool output, multiple JSON blocks, etc.
    """
    if text is None:
        raise ValueError("Empty model output.")

    s = text.strip()

    # Fast path
    try:
        return json.loads(s)
    except Exception:
        pass

    # Robust path: find first JSON object/array and decode only that portion
    decoder = json.JSONDecoder()

    # Try from first '{'
    i = s.find("{")
    if i != -1:
        try:
            obj, end = decoder.raw_decode(s[i:])
            return obj
        except Exception:
            pass

    # Try from first '[' (in case model returns a list)
    j = s.find("[")
    if j != -1:
        try:
            obj, end = decoder.raw_decode(s[j:])
            return obj
        except Exception:
            pass

    # If we get here, we couldn't decode any JSON
    raise ValueError("No valid JSON found in model output.")


def reconcile_to_target(items, target, tol=0.01):
    """
    Make subtotal match target by adjusting the last item's rate slightly.
    If rounding still leaves a tiny remainder, add an Adjustment line.
    """
    items = items or []
    if not items:
        return [], 0.0, round(target, 2), "No items returned by AI."

    # Round rates to 2 decimals for invoice realism
    for it in items:
        it["qty"] = float(it.get("qty") or 0)
        it["rate"] = round(float(it.get("rate") or 0), 2)

    def subtotal():
        return round(sum(float(it["qty"]) * float(it["rate"]) for it in items), 2)

    sub = subtotal()
    delta = round(target - sub, 2)

    if abs(delta) <= tol:
        return items, sub, delta, "No reconciliation needed."

    # Try adjusting last line's rate
    last = items[-1]
    q = float(last["qty"]) if float(last["qty"]) > 0 else 1.0
    new_rate = round(float(last["rate"]) + (delta / q), 2)

    # Ensure rate doesn't go <= 0 (fallback to adjustment line)
    if new_rate > 0:
        last["rate"] = new_rate
        sub2 = subtotal()
        delta2 = round(target - sub2, 2)

        if abs(delta2) <= tol:
            return items, sub2, delta2, f"Adjusted last item rate to {new_rate} to hit target."

        # If rounding still leaves a few cents, add adjustment line
        if abs(delta2) > tol:
            items.append({
                "item_no": "ADJ-0001",
                "description": "Price Adjustment (rounding)",
                "qty": 1,
                "unit": "EA",
                "rate": float(delta2)  # can be + or -
            })
            sub3 = subtotal()
            delta3 = round(target - sub3, 2)
            return items, sub3, delta3, "Adjusted last item + added rounding adjustment line."

    # Final fallback: add adjustment item
    items.append({
        "item_no": "ADJ-0001",
        "description": "Price Adjustment",
        "qty": 1,
        "unit": "EA",
        "rate": float(delta)
    })
    sub4 = subtotal()
    delta4 = round(target - sub4, 2)
    return items, sub4, delta4, "Added adjustment line to match target."

@app.post("/api/ai/suggest_items")
def ai_suggest_items():
    try:
        if not client:
            return jsonify(ok=False, error="Missing GEMINI_API_KEY in environment."), 400

        payload = request.get_json(force=True) or {}
        company = (payload.get("company") or "").strip()
        origin  = (payload.get("origin") or "").strip()
        amount  = float(payload.get("amount") or 0)
        currency = (payload.get("currency") or "USD").strip().upper()
        notes   = (payload.get("notes") or "").strip()

        if not company or not origin or amount <= 0:
            return jsonify(ok=False, error="company, origin, amount are required and amount must be > 0"), 400

        prompt = f"""
You are helping build an export invoice item list.

Goal:
- Company: {company}
- Origin/Region: {origin}
- Target total amount: {amount} {currency}

Task:
Suggest REALISTIC product items that this company sells (or is widely known for),
with quantities and market unit prices so that the total is close to the target amount.

Output STRICT JSON (no markdown):
{{
  "items": [
    {{
      "item_no": "SKU-0001",
      "description": "Product name + short spec",
      "qty": 10,
      "unit": "PCS",
      "rate": 123.45
    }}
  ],
  "notes": "Assumptions about pricing sources and why these products fit"
}}

Rules:
- 3 to 8 items
- qty and rate must be positive numbers
- Keep units simple (PCS, BOX, SET)
- Rates should be plausible market prices (estimate if needed, but say so in notes)
- Make the total close to target amount
Extra notes from user: {notes if notes else "(none)"}
""".strip()

        grounding_tool = types.Tool(google_search=types.GoogleSearch())

        config = types.GenerateContentConfig(
            tools=[grounding_tool],
        )

        # ✅ IMPORTANT: no trailing comma here
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=config,
        )

        # Gemini should return JSON text, but still parse robustly
        data = _extract_json(resp.text or "")
        items = data.get("items") or []

        out_items = []
        for it in items:
            item_no = str(it.get("item_no") or it.get("sku") or "").strip()
            desc    = str(it.get("description") or it.get("name") or "").strip()
            unit    = str(it.get("unit") or "PCS").strip()
            qty     = float(it.get("qty") or 0)
            rate    = float(it.get("rate") or 0)

            if not desc or qty <= 0 or rate <= 0:
                continue

            out_items.append({
                "item_no": item_no or "SKU-XXXX",
                "description": desc,
                "qty": qty,
                "unit": unit or "PCS",
                "rate": rate,
            })

        if not out_items:
            return jsonify(ok=False, error="AI returned no valid items. Try different company/origin or add notes."), 400

        out_items, final_sub, final_delta, recon_note = reconcile_to_target(out_items, amount, tol=0.01)

        return jsonify(
            ok=True,
            items=out_items,
            notes=(str(data.get("notes") or "") +
                   f"\n\nReconcile:\n{recon_note}\nFinal subtotal: {final_sub}\nDelta: {final_delta}").strip()
        )

    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500

import os
import requests
from flask import jsonify, abort

NOTION_TOKEN = os.getenv("NOTION_TOKEN", "").strip()
NOTION_API = "https://api.notion.com/v1"
NOTION_VERSION = os.getenv("NOTION_VERSION", "2022-06-28")

def _notion_headers():
    if not NOTION_TOKEN:
        return None
    return {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }

def _prop_to_text(prop: dict) -> str:
    """Convert a Notion property object to plain text (best effort)."""
    if not prop or "type" not in prop:
        return ""

    t = prop["type"]
    v = prop.get(t)

    if t == "title":
        return "".join(x.get("plain_text", "") for x in (v or []))
    if t == "rich_text":
        return "".join(x.get("plain_text", "") for x in (v or []))
    if t == "select":
        return (v or {}).get("name", "") if isinstance(v, dict) else ""
    if t == "multi_select":
        return ", ".join(x.get("name", "") for x in (v or []))
    if t == "number":
        return "" if v is None else str(v)
    if t == "email":
        return v or ""
    if t == "phone_number":
        return v or ""
    if t == "url":
        return v or ""

    # fallback (relation/people/files/formula etc.)
    return ""

def _guess_title_prop(properties: dict) -> str:
    for k, p in (properties or {}).items():
        if p.get("type") == "title":
            return k
    return ""

def _guess_address_prop(properties: dict) -> str:
    # prefer any prop whose name contains "address"
    for k in (properties or {}).keys():
        if "address" in k.lower():
            return k
    return ""

def _fetch_database_rows(database_id: str, name_prop: str = "", address_prop: str = ""):
    headers = _notion_headers()
    if not headers:
        raise RuntimeError("NOTION_TOKEN is missing in .env")

    url = f"{NOTION_API}/databases/{database_id}/query"
    payload = {"page_size": 100}

    items = []
    has_more = True
    start_cursor = None

    while has_more:
        if start_cursor:
            payload["start_cursor"] = start_cursor

        r = requests.post(url, headers=headers, json=payload, timeout=30)
        if r.status_code >= 400:
            # bubble error as text for debugging
            raise RuntimeError(f"Notion error {r.status_code}: {r.text}")

        data = r.json()
        results = data.get("results", [])

        for page in results:
            props = page.get("properties", {}) or {}

            # auto-detect if not provided
            title_key = name_prop or _guess_title_prop(props)
            addr_key = address_prop or _guess_address_prop(props)

            name = _prop_to_text(props.get(title_key, {})) if title_key else ""
            address = _prop_to_text(props.get(addr_key, {})) if addr_key else ""

            name = (name or "").strip()
            address = (address or "").strip()

            # If title is empty, skip
            if not name:
                continue

            items.append({"name": name, "address": address})

        has_more = bool(data.get("has_more"))
        start_cursor = data.get("next_cursor")

        # safety stop
        if len(items) > 2000:
            break

    # sort A-Z
    items.sort(key=lambda x: x["name"].lower())
    return items

@app.get("/api/parties/<kind>")
def api_parties(kind: str):
    try:
        kind = (kind or "").lower().strip()

        if kind == "buyer":
            db_id = os.getenv("BUYER_PARTY_DATABASE_ID", "").strip()
            if not db_id:
                return jsonify(ok=False, error="Missing BUYER_PARTY_DATABASE_ID in .env"), 400

            name_prop = os.getenv("BUYER_NAME_PROP", "").strip()
            addr_prop = os.getenv("BUYER_ADDRESS_PROP", "").strip()
            items = _fetch_database_rows(db_id, name_prop=name_prop, address_prop=addr_prop)
            return jsonify(ok=True, items=items)

        if kind == "notify":
            db_id = os.getenv("NOTIFY_PARTY_DATABASE_ID", "").strip()
            if not db_id:
                return jsonify(ok=False, error="Missing NOTIFY_PARTY_DATABASE_ID in .env"), 400

            name_prop = os.getenv("NOTIFY_NAME_PROP", "").strip()
            addr_prop = os.getenv("NOTIFY_ADDRESS_PROP", "").strip()
            items = _fetch_database_rows(db_id, name_prop=name_prop, address_prop=addr_prop)
            return jsonify(ok=True, items=items)

        return jsonify(ok=False, error="kind must be buyer or notify"), 404

    except Exception as e:
        # send back debug info (so you can see Notion errors in browser console too)
        return jsonify(ok=False, error=str(e)), 500

# ----------------------------
# Jinja environment for rendering templates from strings
# ----------------------------
jinja_env = Environment(
    undefined=StrictUndefined,
    autoescape=select_autoescape(["html", "xml"]),
)

# ----------------------------
# Design Archetypes (18 distinct)  (KEEPING AS-IS)

ARCHETYPES_DIR = Path(os.getenv("ARCHETYPES_DIR", BASE_DIR / "archetypes")).resolve()

# ----------------------------
def load_archetype_index() -> List[dict]:
    """
    Auto-load archetype templates from ARCHETYPES_DIR/*.html
    File name (without .html) becomes archetype_id.
    """
    if not ARCHETYPES_DIR.exists():
        raise RuntimeError(
            f"Archetypes folder not found: {ARCHETYPES_DIR}\n"
            f"Create it next to app.py, or set ARCHETYPES_DIR env var."
        )

    files = sorted(ARCHETYPES_DIR.glob("*.html"))
    if not files:
        raise RuntimeError(
            f"No archetype templates found in: {ARCHETYPES_DIR}\n"
            f"Expected: {ARCHETYPES_DIR}/A01_something.html etc."
        )

    out = []
    for f in files:
        archetype_id = f.stem  # filename without .html
        out.append({
            "id": archetype_id,
            "name": archetype_id.replace("_", " "),
            "rules": f"Loaded from {f.name}",
            "path": str(f),
        })
    return out


ARCHETYPES = load_archetype_index()
ARCHETYPE_BY_ID = {a["id"]: a for a in ARCHETYPES}


# ----------------------------
# Data schema expected from UI
# ----------------------------
REQUIRED_TOP_KEYS = ["company", "invoice", "buyer", "notify", "category", "items", "totals", "bank"]

def slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "company"

def safe_filename(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "invoice.html"

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def company_dir(slug: str) -> Path:
    d = STORAGE_DIR / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / "invoices").mkdir(parents=True, exist_ok=True)
    return d

def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def list_existing_templates() -> List[Tuple[str, str]]:
    out = []
    if not STORAGE_DIR.exists():
        return out
    for d in STORAGE_DIR.iterdir():
        if d.is_dir():
            tpl = d / "template.html"
            if tpl.exists():
                out.append((d.name, tpl.read_text(encoding="utf-8")))
    return out

# ----------------------------
# Similarity (Jaccard over character shingles)
# ----------------------------
def normalize_for_similarity(html: str) -> str:
    """
    Structural fingerprint only (ignore CSS + inline styles + most attributes).
    This prevents COMMON_STYLE from making everything look similar.
    """
    # remove style + script blocks entirely
    html = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)

    # remove jinja blocks/vars (already similar across all templates)
    html = re.sub(r"{%.*?%}", " ", html, flags=re.DOTALL)
    html = re.sub(r"{{.*?}}", " ", html, flags=re.DOTALL)

    # remove inline style attributes and common noisy attrs
    html = re.sub(r'\sstyle="[^"]*"', " ", html, flags=re.IGNORECASE)
    html = re.sub(r"\sclass='[^']*'|\sclass=\"[^\"]*\"", " ", html, flags=re.IGNORECASE)
    html = re.sub(r"\sid='[^']*'|\sid=\"[^\"]*\"", " ", html, flags=re.IGNORECASE)

    # keep only tag names sequence (structure)
    tags = re.findall(r"</?\s*([a-z0-9]+)", html.lower())
    # drop super-common tags to reduce noise
    drop = {"html","head","meta","title","body","div","span","b","br"}
    tags = [t for t in tags if t not in drop]

    return " ".join(tags)


def shingles(s: str, k: int = 7) -> set:
    # token shingles (not char shingles)
    toks = s.split()
    if len(toks) <= k:
        return {s}
    return {" ".join(toks[i:i+k]) for i in range(len(toks)-k+1)}


def jaccard_similarity(a: str, b: str) -> float:
    sa = shingles(normalize_for_similarity(a))
    sb = shingles(normalize_for_similarity(b))
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union


def max_similarity(candidate: str, existing: List[Tuple[str, str]]) -> float:
    if not existing:
        return 0.0
    return max(jaccard_similarity(candidate, tpl) for _, tpl in existing)

# ----------------------------
# Monochrome validation
# ----------------------------
ALLOWED_COLOR_KEYWORDS = {"black", "white", "gray", "grey", "transparent", "currentcolor", "inherit"}

def is_grayscale_rgb(value: str) -> bool:
    m = re.match(r"rgba?\(\s*([0-9]{1,3})\s*,\s*([0-9]{1,3})\s*,\s*([0-9]{1,3})(?:\s*,\s*([0-9.]+))?\s*\)", value)
    if not m:
        return False
    r, g, b = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if r > 255 or g > 255 or b > 255:
        return False
    return r == g == b

def is_allowed_hex(value: str) -> bool:
    m = re.match(r"#([0-9a-f]{3}|[0-9a-f]{6})$", value.lower())
    if not m:
        return False
    hx = m.group(1)
    if len(hx) == 3:
        return hx[0] == hx[1] == hx[2]
    return hx[0:2] == hx[2:4] == hx[4:6]

def extract_css_blocks(html: str) -> str:
    styles = []
    for m in re.finditer(r"<style[^>]*>(.*?)</style>", html, flags=re.DOTALL | re.IGNORECASE):
        styles.append(m.group(1))
    for m in re.finditer(r'style="(.*?)"', html, flags=re.DOTALL | re.IGNORECASE):
        styles.append(m.group(1))
    return "\n".join(styles)

def validate_monochrome_and_print(html: str) -> List[str]:
    errors = []
    if not re.search(r"<style[^>]*>.*?</style>", html, flags=re.DOTALL | re.IGNORECASE):
        errors.append("Template must include a <style>...</style> block (inline CSS only).")
    if "@media print" not in html:
        errors.append("Template must include @media print rules.")
    if "@page" not in html:
        errors.append("Template must include @page margin rules.")

    if re.search(r"<link[^>]+rel=['\"]stylesheet['\"]", html, flags=re.IGNORECASE):
        errors.append("External stylesheets (<link rel='stylesheet'>) are not allowed (inline CSS only).")
    css = extract_css_blocks(html)
    if re.search(r"@import\s+", css, flags=re.IGNORECASE):
        errors.append("CSS @import is not allowed (inline only).")
    if re.search(r"url\(\s*['\"]?\s*https?://", css, flags=re.IGNORECASE):
        errors.append("External URLs in CSS are not allowed (must be self-contained).")

    # Disallow gradients explicitly (you asked monochrome + your validator)
    if re.search(r"gradient\(", css, flags=re.IGNORECASE):
        errors.append("CSS gradients are not allowed (use solid grayscale only).")

    color_tokens = re.findall(r"(?i)\b(color|background|background-color|border-color|outline-color)\s*:\s*([^;]+);", css)
    for prop, raw in color_tokens:
        val = raw.strip().lower()
        parts = re.split(r"\s+", val)
        for part in parts:
            part = part.strip().strip(",")
            if not part:
                continue
            if part in ALLOWED_COLOR_KEYWORDS:
                continue
            if is_allowed_hex(part):
                continue
            if is_grayscale_rgb(part):
                continue
            if part in {"none", "solid", "dashed", "dotted", "double"}:
                continue
            if re.match(r"^[0-9.]+(px|pt|em|rem|%)$", part):
                continue
            if "#" in part or "rgb" in part:
                errors.append(f"Non-monochrome color detected in {prop}: {raw.strip()}")
                break
    return errors

# ----------------------------
# Jinja validation
# ----------------------------
REQUIRED_JINJA_SNIPPETS = [
    "{{ company.name",
    "{{ company.address",
    "{{ buyer.name",
    "{{ buyer.address",
    "{{ invoice.title",
    "{{ invoice.number",
    "{{ invoice.date",
    "{% for it in items %}",
    "{% endfor %}",
    "{{ totals.grand_total",
    "{{ bank.bank_name",
]

def validate_jinja(html: str) -> List[str]:
    errors = []
    for snip in REQUIRED_JINJA_SNIPPETS:
        if snip not in html:
            errors.append(f"Missing required placeholder/structure: {snip}")
    try:
        tpl = jinja_env.from_string(html)
        dummy = dummy_invoice_data()
        tpl.render(**dummy)
    except Exception as e:
        errors.append(f"Jinja compile/render error: {e}")
    return errors

def dummy_invoice_data() -> dict:
    return {
        "company": {"name": "Demo Co", "address": "123 Demo Street\nCity"},
        "invoice": {
            "title": "PROFORMA INVOICE",
            "number": "PI-0001",
            "date": "2025-01-01",
            "shipment_method": "VIA SEA",
            "port_loading": "CHINA",
            "port_discharge": "CHENNAI, INDIA",
        },
        "buyer": {"name": "Buyer Co", "address": "Buyer Address\nLine 2"},
        "notify": {"name": "Notify Co", "address": "Notify Address\nLine 2"},
        "category": "LED LIGHTING FIXTURES AND SPARES",
        "items": [
            {"sl_no": 1, "item_no": "SKU-0001", "description": "Product A", "qty": 10, "unit": "NOS", "rate": 12.5, "amount": 125.0},
            {"sl_no": 2, "item_no": "SKU-0002", "description": "Product B", "qty": 5, "unit": "NOS", "rate": 20.0, "amount": 100.0},
        ],
        "totals": {
            "subtotal": 225.0,
            "freight": 0.0,
            "insurance": 0.0,
            "other": 0.0,
            "grand_total": 225.0,
            "amount_in_words": "Two hundred twenty five only",
        },
        "bank": {
            "bank_name": "Demo Bank",
            "account_name": "Demo Co",
            "account_no": "1234567890",
            "ifsc": "DEMO0001234",
            "swift": "",
            "iban": "",
            "routing": "",
            "branch": "Main Branch",
        },
    }

# ----------------------------
# OpenAI Structured Output schema (recipe + template)
# ----------------------------
if USE_OPENAI:
    class TemplateRecipe(BaseModel):
        archetype_id: str = Field(..., description="One of the provided archetype IDs.")
        seed: str = Field(..., description="Stable seed to keep layout stable for this company.")
        layout_notes: str = Field(..., description="Short explanation of layout decisions to keep stable.")
        monochrome_rules: str = Field(..., description="Explicit note that only grayscale is used.")
        print_rules: str = Field(..., description="Explicit note about @page margins and @media print.")

    class TemplateBundle(BaseModel):
        recipe: TemplateRecipe
        template_html: str = Field(..., description="A complete HTML document with inline CSS and Jinja2 placeholders.")

# ----------------------------
# Prompting strategy
# ----------------------------
def build_generation_prompt(company_name: str, archetype: dict, seed: str, existing_fingerprints: List[str]) -> str:
    negative = "\n".join("- " + fp for fp in existing_fingerprints[:10]) or "- (none yet)"
    return (
        "You are an expert print designer. Create a professional invoice template.\n"
        "HARD CONSTRAINTS (non-negotiable):\n"
        "1) Strictly monochrome only: black/white/gray. No colors, no gradients.\n"
        "2) Print-friendly: inline CSS only inside <style>, include @page and @media print.\n"
        "3) Self-contained: no external fonts, no external CSS, no images from URLs.\n"
        "4) Output MUST be valid Jinja2 and include the items loop exactly:\n"
        "   {% for it in items %} ... {% endfor %}\n\n"
        "DATA CONTRACT (must use these exact objects/keys):\n"
        "- company.name, company.address\n"
        "- invoice.title, invoice.number, invoice.date, invoice.shipment_method, invoice.port_loading, invoice.port_discharge\n"
        "- buyer.name, buyer.address\n"
        "- notify.name, notify.address\n"
        "- category\n"
        "- items: list of {sl_no, item_no, description, qty, unit, rate, amount}\n"
        "- totals.subtotal, totals.freight, totals.insurance, totals.other, totals.grand_total, totals.amount_in_words\n"
        "- bank.bank_name, bank.account_name, bank.account_no, bank.ifsc, bank.swift, bank.iban, bank.routing, bank.branch\n\n"
        "DIVERSITY REQUIREMENT:\n"
        "This company MUST look genuinely different from existing templates. Avoid these fingerprints:\n"
        f"{negative}\n\n"
        "ARCHETYPE TO FOLLOW:\n"
        f"- archetype_id: {archetype['id']}\n"
        f"- archetype_name: {archetype['name']}\n"
        f"- archetype_rules: {archetype['rules']}\n"
        f"- seed: {seed}\n\n"
        "STYLE RULES:\n"
        "- Use system fonts only (e.g., Arial/Helvetica/Times/Courier).\n"
        "- Use hairline rules, boxes, and spacing creatively.\n"
        "- Use clear typographic hierarchy.\n"
        "- Use |e filter for user-provided strings where appropriate.\n\n"
        "Return a JSON object matching the schema: { recipe: {...}, template_html: \"...\" }.\n"
        "The HTML must be a FULL document: <!doctype html><html>...<body>...</body></html>.\n"
    )

# ----------------------------
# FALLBACK: 18 archetypes supported offline (distinct structures)
# ----------------------------
COMMON_STYLE = (
    "<style>"
    "@page{margin:14mm;}"
    "@media print{.no-print{display:none!important;}}"
    "html,body{margin:0;padding:0;}"
    "body{color:#111;}"
    ".muted{color:#444;}"
    ".caps{letter-spacing:.08em;text-transform:uppercase;}"
    ".pre{white-space:pre-line;}"
    ".hr{border-top:2px solid #111;margin:12px 0;}"
    ".h1{margin:0;font-weight:800;}"
    ".small{font-size:12px;}"
    ".tiny{font-size:11px;}"
    ".right{text-align:right;}"
    ".center{text-align:center;}"
    "table{width:100%;border-collapse:collapse;}"
    "th,td{padding:8px 6px;vertical-align:top;}"
    "th{font-size:11px;letter-spacing:.08em;text-transform:uppercase;}"
    "</style>"
)

# Items variants (to reduce similarity across archetypes)
ITEMS_TABLE_STRICT = (
    "<table>"
    "<thead><tr>"
    "<th style='width:44px'>Sl</th><th style='width:120px'>Item No</th><th>Description</th>"
    "<th class='right' style='width:64px'>Qty</th><th style='width:70px'>Unit</th>"
    "<th class='right' style='width:90px'>Rate</th><th class='right' style='width:110px'>Amount</th>"
    "</tr></thead>"
    "<tbody>"
    "{% for it in items %}"
    "<tr style='border-bottom:1px solid #111'>"
    "<td>{{ it.sl_no }}</td><td>{{ it.item_no|e }}</td><td class='pre'>{{ it.description|e }}</td>"
    "<td class='right'>{{ it.qty }}</td><td>{{ it.unit|e }}</td>"
    "<td class='right'>{{ it.rate }}</td><td class='right'>{{ it.amount }}</td>"
    "</tr>"
    "{% endfor %}"
    "</tbody></table>"
)

ITEMS_LEDGER = (
    "<table style='font-family:Courier New,Courier,monospace;font-size:12px;'>"
    "<thead><tr>"
    "<th style='border-bottom:2px solid #111'>#</th>"
    "<th style='border-bottom:2px solid #111'>SKU</th>"
    "<th style='border-bottom:2px solid #111'>DESC</th>"
    "<th class='right' style='border-bottom:2px solid #111'>QTY</th>"
    "<th style='border-bottom:2px solid #111'>U</th>"
    "<th class='right' style='border-bottom:2px solid #111'>RATE</th>"
    "<th class='right' style='border-bottom:2px solid #111'>AMT</th>"
    "</tr></thead><tbody>"
    "{% for it in items %}"
    "<tr>"
    "<td>{{ it.sl_no }}</td><td>{{ it.item_no|e }}</td><td class='pre'>{{ it.description|e }}</td>"
    "<td class='right'>{{ it.qty }}</td><td>{{ it.unit|e }}</td>"
    "<td class='right'>{{ it.rate }}</td><td class='right'>{{ it.amount }}</td>"
    "</tr>"
    "{% endfor %}"
    "</tbody></table>"
)

ITEMS_GRID = (
    "<div style='border-top:2px solid #111;margin-top:10px;'>"
    "<div style='display:grid;grid-template-columns:44px 120px 1fr 64px 70px 90px 110px;"
    "gap:0;border-bottom:2px solid #111;font-size:11px;letter-spacing:.08em;text-transform:uppercase;padding:8px 0;'>"
    "<div>Sl</div><div>Item</div><div>Description</div><div class='right'>Qty</div><div>Unit</div>"
    "<div class='right'>Rate</div><div class='right'>Amount</div></div>"
    "{% for it in items %}"
    "<div style='display:grid;grid-template-columns:44px 120px 1fr 64px 70px 90px 110px;gap:0;"
    "border-bottom:1px solid #bbb;padding:8px 0;'>"
    "<div>{{ it.sl_no }}</div><div>{{ it.item_no|e }}</div><div class='pre'>{{ it.description|e }}</div>"
    "<div class='right'>{{ it.qty }}</div><div>{{ it.unit|e }}</div>"
    "<div class='right'>{{ it.rate }}</div><div class='right'>{{ it.amount }}</div>"
    "</div>"
    "{% endfor %}"
    "</div>"
)

ITEMS_TICKET_LINES = (
    "<div style='margin-top:10px;border-top:2px dashed #111;padding-top:10px;'>"
    "{% for it in items %}"
    "<div style='display:flex;gap:10px;align-items:flex-start;border-bottom:1px dashed #999;padding:8px 0;'>"
    "<div style='width:36px;font-weight:800;'>{{ it.sl_no }}</div>"
    "<div style='flex:1;'>"
    "<div style='font-weight:700;'>{{ it.item_no|e }}</div>"
    "<div class='pre tiny muted'>{{ it.description|e }}</div>"
    "</div>"
    "<div class='right' style='width:70px;'>{{ it.qty }}</div>"
    "<div style='width:60px;'>{{ it.unit|e }}</div>"
    "<div class='right' style='width:90px;'>{{ it.rate }}</div>"
    "<div class='right' style='width:110px;font-weight:800;'>{{ it.amount }}</div>"
    "</div>"
    "{% endfor %}"
    "</div>"
)

ITEMS_FORM_BOXES = (
    "<table style='table-layout:fixed;'>"
    "<thead><tr>"
    "<th style='border:1px solid #111;width:44px'>Sl</th>"
    "<th style='border:1px solid #111;width:120px'>Item No</th>"
    "<th style='border:1px solid #111'>Description</th>"
    "<th style='border:1px solid #111;width:64px'>Qty</th>"
    "<th style='border:1px solid #111;width:70px'>Unit</th>"
    "<th style='border:1px solid #111;width:90px'>Rate</th>"
    "<th style='border:1px solid #111;width:110px'>Amount</th>"
    "</tr></thead><tbody>"
    "{% for it in items %}"
    "<tr>"
    "<td style='border:1px solid #111'>{{ it.sl_no }}</td>"
    "<td style='border:1px solid #111'>{{ it.item_no|e }}</td>"
    "<td style='border:1px solid #111' class='pre'>{{ it.description|e }}</td>"
    "<td style='border:1px solid #111' class='right'>{{ it.qty }}</td>"
    "<td style='border:1px solid #111'>{{ it.unit|e }}</td>"
    "<td style='border:1px solid #111' class='right'>{{ it.rate }}</td>"
    "<td style='border:1px solid #111' class='right'>{{ it.amount }}</td>"
    "</tr>"
    "{% endfor %}"
    "</tbody></table>"
)

ITEMS_VARIANTS = {
    "strict": ITEMS_TABLE_STRICT,
    "ledger": ITEMS_LEDGER,
    "grid": ITEMS_GRID,
    "ticket": ITEMS_TICKET_LINES,
    "form": ITEMS_FORM_BOXES,
}

def _bank_block(style: str = "boxed") -> str:
    if style == "inline":
        return (
            "<div class='tiny' style='border-top:2px solid #111;padding-top:10px;margin-top:12px;'>"
            "<span class='caps tiny'>Bank</span>: {{ bank.bank_name|e }} | "
            "<span class='caps tiny'>A/C</span>: {{ bank.account_name|e }} ({{ bank.account_no|e }}) | "
            "<span class='caps tiny'>IFSC</span>: {{ bank.ifsc|e }} "
            "{% if bank.swift %}| <span class='caps tiny'>SWIFT</span>: {{ bank.swift|e }}{% endif %}"
            "{% if bank.iban %}| <span class='caps tiny'>IBAN</span>: {{ bank.iban|e }}{% endif %}"
            "{% if bank.routing %}| <span class='caps tiny'>Routing</span>: {{ bank.routing|e }}{% endif %}"
            "{% if bank.branch %}| <span class='caps tiny'>Branch</span>: {{ bank.branch|e }}{% endif %}"
            "</div>"
        )
    return (
        "<div style='border:1px solid #111;padding:10px;border-radius:10px;'>"
        "<div class='caps tiny' style='font-weight:800;margin-bottom:6px;'>Bank Details</div>"
        "<div class='tiny'><b>Bank</b>: {{ bank.bank_name|e }}</div>"
        "<div class='tiny'><b>Account Name</b>: {{ bank.account_name|e }}</div>"
        "<div class='tiny'><b>Account No</b>: {{ bank.account_no|e }}</div>"
        "<div class='tiny'><b>IFSC</b>: {{ bank.ifsc|e }}</div>"
        "{% if bank.swift %}<div class='tiny'><b>SWIFT</b>: {{ bank.swift|e }}</div>{% endif %}"
        "{% if bank.iban %}<div class='tiny'><b>IBAN</b>: {{ bank.iban|e }}</div>{% endif %}"
        "{% if bank.routing %}<div class='tiny'><b>Routing</b>: {{ bank.routing|e }}</div>{% endif %}"
        "{% if bank.branch %}<div class='tiny'><b>Branch</b>: {{ bank.branch|e }}</div>{% endif %}"
        "</div>"
    )

def _totals_block(style: str = "boxed") -> str:
    if style == "rail":
        return (
            "<div style='border:2px solid #111;padding:10px;border-radius:12px;'>"
            "<div class='caps tiny' style='font-weight:800;margin-bottom:8px;'>Totals</div>"
            "<div class='tiny' style='display:flex;justify-content:space-between;'><span>Subtotal</span><span>{{ totals.subtotal }}</span></div>"
            "<div class='tiny' style='display:flex;justify-content:space-between;'><span>Freight</span><span>{{ totals.freight }}</span></div>"
            "<div class='tiny' style='display:flex;justify-content:space-between;'><span>Insurance</span><span>{{ totals.insurance }}</span></div>"
            "<div class='tiny' style='display:flex;justify-content:space-between;'><span>Other</span><span>{{ totals.other }}</span></div>"
            "<div style='border-top:2px solid #111;margin:10px 0;'></div>"
            "<div style='display:flex;justify-content:space-between;font-weight:800;'><span>Grand Total</span><span>{{ totals.grand_total }}</span></div>"
            "<div class='tiny muted pre' style='margin-top:8px;'>{{ totals.amount_in_words|e }}</div>"
            "</div>"
        )
    return (
        "<div style='border:1px solid #111;padding:10px;border-radius:10px;'>"
        "<div class='caps tiny' style='font-weight:800;margin-bottom:6px;'>Summary</div>"
        "<div class='tiny' style='display:flex;justify-content:space-between;'><span>Subtotal</span><span>{{ totals.subtotal }}</span></div>"
        "<div class='tiny' style='display:flex;justify-content:space-between;'><span>Freight</span><span>{{ totals.freight }}</span></div>"
        "<div class='tiny' style='display:flex;justify-content:space-between;'><span>Insurance</span><span>{{ totals.insurance }}</span></div>"
        "<div class='tiny' style='display:flex;justify-content:space-between;'><span>Other</span><span>{{ totals.other }}</span></div>"
        "<div style='border-top:2px solid #111;margin:10px 0;'></div>"
        "<div style='display:flex;justify-content:space-between;font-weight:800;'><span>Grand Total</span><span>{{ totals.grand_total }}</span></div>"
        "<div class='tiny muted pre' style='margin-top:8px;'>{{ totals.amount_in_words|e }}</div>"
        "</div>"
    )

def fallback_template(archetype_id: str) -> str:
    a = ARCHETYPE_BY_ID.get(archetype_id)
    if not a:
        raise RuntimeError(f"Unknown archetype_id: {archetype_id}")

    p = Path(a["path"])
    if not p.exists():
        raise RuntimeError(f"Archetype file missing on disk: {p}")

    return p.read_text(encoding="utf-8")


# ----------------------------
# Template generation pipeline
# ----------------------------
def choose_candidate_archetypes(company_slug: str, k: int = 12) -> List[dict]:
    if not ARCHETYPES:
        raise RuntimeError("No archetype templates found in ./archetypes folder.")
    counts = {a["id"]: 0 for a in ARCHETYPES}
    if STORAGE_DIR.exists():
        for d in STORAGE_DIR.iterdir():
            if d.is_dir():
                r = load_json(d / "recipe.json")
                if r and r.get("archetype_id") in counts:
                    counts[r["archetype_id"]] += 1

    def score(a: dict) -> Tuple[int, str]:
        h = sha256_text(company_slug + "::" + a["id"])
        return (counts[a["id"]], h)

    return sorted(ARCHETYPES, key=score)[:k]

def generate_template_candidates(company_name: str, company_slug: str, k: int = 12) -> List[Tuple[dict, str]]:
    existing = list_existing_templates()
    fingerprints = []
    for slug, tpl in existing[:12]:
        fp = normalize_for_similarity(tpl)[:220]
        fingerprints.append(slug + ": " + fp)

    candidates = []
    seed = sha256_text(company_slug)[:12]
    archetypes = choose_candidate_archetypes(company_slug, k=k)

    for arch in archetypes:
        # Try OpenAI first (optional)
        if USE_OPENAI and OPENAI_API_KEY:
            client = OpenAI(api_key=OPENAI_API_KEY)
            prompt = build_generation_prompt(company_name, arch, seed + "_" + arch["id"][-3:], fingerprints)
            try:
                response = client.responses.parse(
                    model=OPENAI_MODEL,
                    input=[
                        {"role": "system", "content": "Return only structured JSON matching the schema."},
                        {"role": "user", "content": prompt},
                    ],
                    text_format=TemplateBundle,  # type: ignore
                )
                bundle = response.output_parsed  # type: ignore
                recipe = bundle.recipe.dict()
                html = bundle.template_html
                recipe["archetype_id"] = arch["id"]
                candidates.append((recipe, html))
                continue
            except Exception:
                pass

        # Offline fallback (NOW unique per archetype)
        recipe = {
            "archetype_id": arch["id"],
            "seed": seed + "_" + arch["id"][-3:],
            "layout_notes": "Offline fallback template for " + arch["name"],
            "monochrome_rules": "grayscale only (black/white/gray); no gradients",
            "print_rules": "Includes @page margins and @media print",
        }
        html = fallback_template(arch["id"])
        candidates.append((recipe, html))

    return candidates

def pick_best_candidate(company_slug: str, candidates: List[Tuple[dict, str]]) -> Tuple[dict, str, dict]:
    """
    Validation-lite:
    - Only checks that the template can compile + render with dummy data.
    - Still picks the least similar template if multiple pass.
    """
    existing = list_existing_templates()

    best = None
    debug = {"candidates": []}

    for recipe, html in candidates:
        errs = []

        # Only ensure Jinja can render (prevents broken templates crashing later)
        try:
            tpl = jinja_env.from_string(html)
            tpl.render(**dummy_invoice_data())
        except Exception as e:
            errs.append(f"Jinja compile/render error: {e}")

        sim = max_similarity(html, existing)
        debug["candidates"].append({
            "archetype_id": recipe.get("archetype_id"),
            "similarity_max": round(sim, 4),
            "errors": errs[:8],
            "ok": len(errs) == 0,
        })

        if errs:
            continue

        if best is None or sim < best["sim"]:
            best = {"recipe": recipe, "html": html, "sim": sim}

    if best is None:
        raise ValueError("All generated candidates failed Jinja render. See debug info.")

    return best["recipe"], best["html"], debug

def save_company_template(company_slug: str, company_name: str, recipe: dict, template_html: str) -> None:
    d = company_dir(company_slug)
    (d / "template.html").write_text(template_html, encoding="utf-8")
    save_json(d / "recipe.json", recipe)
    save_json(d / "meta.json", {
        "company_name": company_name,
        "saved_at": int(time.time()),
        "template_sha256": sha256_text(template_html),
        "archetype_id": recipe.get("archetype_id"),
        "seed": recipe.get("seed"),
    })

def load_company_template(company_slug: str) -> Optional[str]:
    d = company_dir(company_slug)
    path = d / "template.html"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")

def load_company_recipe(company_slug: str) -> Optional[dict]:
    d = company_dir(company_slug)
    return load_json(d / "recipe.json")

def save_rendered_invoice(company_slug: str, invoice_number: str, rendered_html: str) -> Tuple[str, str]:
    """
    Saves final invoice HTML:
      - storage/companies/<slug>/invoices/<filename>
      - storage/exports/<filename>  (easy)
    Returns (filename, export_filename)
    """
    d = company_dir(company_slug)
    ts = int(time.time())
    inv = invoice_number.strip() if invoice_number else "INV"
    filename = safe_filename(company_slug + "_" + inv + "_" + str(ts) + ".html")

    company_out = d / "invoices" / filename
    company_out.write_text(rendered_html, encoding="utf-8")

    export_out = EXPORT_DIR / filename
    export_out.write_text(rendered_html, encoding="utf-8")

    return filename, filename

# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def index():
    return render_template("index.html")

@app.get("/health")
def health():
    return jsonify({"ok": True})

@app.get("/download/<company_slug>/<filename>")
def download_company_invoice(company_slug: str, filename: str):
    p = STORAGE_DIR / company_slug / "invoices" / filename
    if not p.exists():
        abort(404)
    return send_file(p, mimetype="text/html", as_attachment=False, download_name=filename)

@app.get("/exports/<filename>")
def download_export(filename: str):
    p = EXPORT_DIR / filename
    if not p.exists():
        abort(404)
    return send_file(p, mimetype="text/html", as_attachment=False, download_name=filename)

@app.post("/api/generate")
def api_generate():
    try:
        payload = request.get_json(force=True, silent=False)
        if not isinstance(payload, dict):
            return jsonify({"ok": False, "error": "Invalid JSON body"}), 400

        for k in REQUIRED_TOP_KEYS:
            if k not in payload:
                return jsonify({"ok": False, "error": "Missing key: " + k}), 400

        company_name = str(payload["company"].get("name", "")).strip()
        if not company_name:
            return jsonify({"ok": False, "error": "company.name is required"}), 400

        slug = slugify(company_name)
        invoice_number = str(payload["invoice"].get("number", "")).strip()

        existing_tpl = load_company_template(slug)
        template_saved = False
        debug_info = None

        if not existing_tpl:
            candidates = generate_template_candidates(company_name, slug, k=12)
            recipe, tpl_html, debug_info = pick_best_candidate(slug, candidates)
            save_company_template(slug, company_name, recipe, tpl_html)
            existing_tpl = tpl_html
            template_saved = True

        # Render using saved template
        tpl = jinja_env.from_string(existing_tpl)
        rendered = tpl.render(
            company=payload["company"],
            invoice=payload["invoice"],
            buyer=payload["buyer"],
            notify=payload["notify"],
            category=payload["category"],
            items=payload["items"],
            totals=payload["totals"],
            bank=payload["bank"],
        )

        # SAVE FINAL RENDERED INVOICE HTML
        company_filename, export_filename = save_rendered_invoice(slug, invoice_number, rendered)

        recipe = load_company_recipe(slug) or {}
        return jsonify({
            "ok": True,
            "company_slug": slug,
            "template_saved": template_saved,
            "archetype_id": recipe.get("archetype_id"),
            "rendered_html": rendered,  # for preview in UI
            "filename": company_filename,
            "download_url": "/download/" + slug + "/" + company_filename,
            "export_url": "/exports/" + export_filename,
            "debug": debug_info,  # only meaningful on first-time generation
        })

    except Exception as e:
        trace = traceback.format_exc()
        return jsonify({"ok": False, "error": str(e), "trace": trace}), 500

from flask import render_template_string

@app.get("/preview")
def preview_all_archetypes():
    """
    Renders all 18 archetypes using dummy_invoice_data() so you can visually compare designs.
    Open: http://127.0.0.1:5055/preview
    """
    dummy = dummy_invoice_data()

    cards = []
    for a in ARCHETYPES:
        arch_id = a["id"]
        # Render the archetype HTML with dummy data
        tpl = jinja_env.from_string(fallback_template(arch_id))
        rendered = tpl.render(**dummy)

        cards.append({
            "id": arch_id,
            "name": a["name"],
            "rules": a["rules"],
            "html": rendered
        })

    # Single page that iframes each design so CSS doesn't conflict between templates
    page = """
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <title>Invoice Archetype Preview</title>
      <style>
        body{font-family:Arial,Helvetica,sans-serif;margin:16px;background:#fafafa;color:#111;}
        h1{margin:0 0 12px 0;}
        .grid{display:grid;grid-template-columns:1fr;gap:14px;}
        .card{border:1px solid #111;border-radius:14px;background:#fff;padding:12px;}
        .top{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;}
        .meta{font-size:12px;color:#444;margin-top:4px;white-space:pre-line;}
        .tag{border:1px solid #111;border-radius:999px;padding:6px 10px;font-size:12px;font-weight:700;}
        iframe{width:100%;height:980px;border:1px solid #bbb;border-radius:12px;background:#fff;}
        .btns{margin-top:8px;display:flex;gap:8px;flex-wrap:wrap;}
        a{color:#111;text-decoration:none;border:1px solid #111;border-radius:999px;padding:6px 10px;font-size:12px;}
        a:hover{background:#f1f1f1;}
      </style>
    </head>
    <body>
      <h1>Invoice Archetype Preview ({{ cards|length }})</h1>
      <div class="grid">
        {% for c in cards %}
          <div class="card" id="{{ c.id }}">
            <div class="top">
              <div>
                <div style="font-weight:800;">{{ c.id }} — {{ c.name }}</div>
                <div class="meta">{{ c.rules }}</div>
              </div>
              <div class="tag">/preview/{{ c.id }}</div>
            </div>

            <div class="btns">
              <a href="/preview/{{ c.id }}" target="_blank">Open full page</a>
              <a href="#{{ c.id }}">Permalink</a>
            </div>

            <iframe srcdoc="{{ c.html|e }}"></iframe>
          </div>
        {% endfor %}
      </div>
    </body>
    </html>
    """
    return render_template_string(page, cards=cards)


@app.get("/preview/<archetype_id>")
def preview_one_archetype(archetype_id: str):
    """
    Renders a single archetype full page.
    Open: http://127.0.0.1:5055/preview/A01_sidebar_letterhead
    """
    a = ARCHETYPE_BY_ID.get(archetype_id)
    if not a:
        abort(404)

    dummy = dummy_invoice_data()
    tpl = jinja_env.from_string(fallback_template(archetype_id))
    rendered = tpl.render(**dummy)
    return rendered

if __name__ == "__main__":
    # Run:  OPENAI_API_KEY=... python app.py
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5055")), debug=True)
