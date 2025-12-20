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
# ----------------------------
ARCHETYPES = [
    {"id": "A01_sidebar_letterhead", "name": "Left Sidebar Letterhead",
     "rules": "Strong left vertical sidebar; right content; sidebar contains company + invoice meta; sections as stacked cards."},
    {"id": "A02_top_banner_blocks", "name": "Top Banner + Block Grid",
     "rules": "Large top banner; below it a 2x2 block grid for buyer/notify/shipping/meta; totals in a separate right-aligned block."},
    {"id": "A03_ledger_minimal", "name": "Ledger Minimal",
     "rules": "Looks like an accounting ledger; thin rules; monospaced feel; table dominates; header is compact."},
    {"id": "A04_ticket_stub", "name": "Ticket / Stub",
     "rules": "Perforation-like dashed divider; left 'stub' with meta; main body with items; feels like a ticket."},
    {"id": "A05_magazine_editorial", "name": "Magazine Editorial",
     "rules": "Editorial typography scale; big invoice title; multi-column summary; items in a clean grid; lots of whitespace."},
    {"id": "A06_blueprint_technical", "name": "Technical Blueprint",
     "rules": "Engineering drawing vibe; thin borders; corner marks; small caps labels; structured sections; watermark-like grid (CSS)."},
    {"id": "A07_receipt_tall", "name": "Tall Receipt",
     "rules": "Narrow centered receipt layout; dense; totals emphasized; sections separated by double rules."},
    {"id": "A08_two_column_classic", "name": "Two-Column Classic",
     "rules": "Classic letter format; left column addresses; right column invoice meta; items below; totals in a boxed area."},
    {"id": "A09_cards_stack", "name": "Stacked Cards",
     "rules": "Everything is separate bordered cards; items table in its own card; bank details card; modern blocks."},
    {"id": "A10_vertical_title_strip", "name": "Vertical Title Strip",
     "rules": "Invoice title rotated or vertical on edge; very distinct; rest is clean grid."},
    {"id": "A11_stamp_seal", "name": "Stamp / Seal",
     "rules": "Big circular 'seal' style element (CSS-only); bold grand total; sections arranged around seal."},
    {"id": "A12_corner_frame", "name": "Corner Frame",
     "rules": "Decorative corner frames (CSS borders); content inside; looks like certificate-ish invoice."},
    {"id": "A13_split_items_totals", "name": "Split Items + Totals Rail",
     "rules": "Items take left 70%; right rail always shows running totals + bank block."},
    {"id": "A14_microgrid_dense", "name": "Microgrid Dense",
     "rules": "Dense microgrid; many thin separators; compact; looks like a shipping document."},
    {"id": "A15_outline_only_borderless", "name": "Borderless Outline Only",
     "rules": "Almost borderless; uses only spacing + typography + hairline rules; very airy."},
    {"id": "A16_typographic_hierarchy", "name": "Typographic Hierarchy",
     "rules": "Huge numerals for invoice number + total; minimal lines; typography does the work."},
    {"id": "A17_diagonal_header", "name": "Diagonal Header",
     "rules": "A diagonal corner header block (CSS); rest aligned; very visually distinct."},
    {"id": "A18_form_like_boxes", "name": "Form-like Boxes",
     "rules": "Looks like a government form; many labeled boxes; items in strict grid; very structured."},
]

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
    """
    Offline generator for ALL 18 archetypes.
    Each archetype returns a full HTML doc with different structure + CSS.
    """
    # pick item variant per archetype to reduce similarity
    item_variant_map = {
        "A01_sidebar_letterhead": "strict",
        "A02_top_banner_blocks": "grid",
        "A03_ledger_minimal": "ledger",
        "A04_ticket_stub": "ticket",
        "A05_magazine_editorial": "grid",
        "A06_blueprint_technical": "form",
        "A07_receipt_tall": "ticket",
        "A08_two_column_classic": "strict",
        "A09_cards_stack": "grid",
        "A10_vertical_title_strip": "strict",
        "A11_stamp_seal": "grid",
        "A12_corner_frame": "strict",
        "A13_split_items_totals": "strict",
        "A14_microgrid_dense": "form",
        "A15_outline_only_borderless": "grid",
        "A16_typographic_hierarchy": "ticket",
        "A17_diagonal_header": "strict",
        "A18_form_like_boxes": "form",
    }
    items_html = ITEMS_VARIANTS[item_variant_map.get(archetype_id, "strict")]

    # base fonts per archetype
    font_map = {
        "A03_ledger_minimal": "font-family:Courier New,Courier,monospace;",
        "A05_magazine_editorial": "font-family:Georgia,'Times New Roman',serif;",
        "A15_outline_only_borderless": "font-family:Arial,Helvetica,sans-serif;",
        "A16_typographic_hierarchy": "font-family:Arial,Helvetica,sans-serif;",
    }
    base_font = font_map.get(archetype_id, "font-family:Arial,Helvetica,sans-serif;")

    # helpers
    totals_box = _totals_block("boxed")
    totals_rail = _totals_block("rail")
    bank_box = _bank_block("boxed")
    bank_inline = _bank_block("inline")

    # --- 18 distinct documents ---
    # Keep them compact but structurally different
    if archetype_id == "A01_sidebar_letterhead":
        return (
            "<!doctype html><html><head><meta charset='utf-8'/>"
            "<title>{{ invoice.title|e }} - {{ company.name|e }}</title>"
            + COMMON_STYLE +
            "<style>"
            "body{" + base_font + "}"
            ".page{display:flex;min-height:100vh;}"
            ".side{width:220px;border-right:2px solid #111;padding:14px;background:#f5f5f5;}"
            ".main{flex:1;padding:14px;}"
            ".chip{border:2px solid #111;padding:8px 10px;border-radius:14px;font-weight:800;display:inline-block;}"
            ".card{border:1px solid #111;border-radius:12px;padding:10px;margin-top:10px;}"
            "</style></head><body>"
            "<div class='page'>"
            "<div class='side'>"
            "<div class='caps tiny muted'>Company</div>"
            "<div style='font-weight:800;margin-top:6px;'>{{ company.name|e }}</div>"
            "<div class='tiny muted pre' style='margin-top:6px;'>{{ company.address|e }}</div>"
            "<div class='hr'></div>"
            "<div class='caps tiny muted'>Invoice</div>"
            "<div class='tiny'><b>No</b>: {{ invoice.number|e }}</div>"
            "<div class='tiny'><b>Date</b>: {{ invoice.date|e }}</div>"
            "<div class='tiny'><b>Ship</b>: {{ invoice.shipment_method|e }}</div>"
            "<div class='tiny'><b>POL</b>: {{ invoice.port_loading|e }}</div>"
            "<div class='tiny'><b>POD</b>: {{ invoice.port_discharge|e }}</div>"
            "</div>"
            "<div class='main'>"
            "<div class='chip'>{{ invoice.title|e }}</div>"
            "<div class='card'><div class='caps tiny muted'>Buyer</div>"
            "<div style='font-weight:800;'>{{ buyer.name|e }}</div><div class='tiny pre muted'>{{ buyer.address|e }}</div></div>"
            "<div class='card'><div class='caps tiny muted'>Notify</div>"
            "<div style='font-weight:800;'>{{ notify.name|e }}</div><div class='tiny pre muted'>{{ notify.address|e }}</div></div>"
            "<div class='card'><span class='caps tiny muted'>Category</span>: {{ category|e }}</div>"
            "<div class='card'>" + items_html + "</div>"
            "<div style='display:grid;grid-template-columns:1fr 360px;gap:12px;margin-top:12px;'>"
            "<div>" + bank_box + "</div><div>" + totals_rail + "</div>"
            "</div>"
            "</div></div></body></html>"
        )

    if archetype_id == "A02_top_banner_blocks":
        return (
            "<!doctype html><html><head><meta charset='utf-8'/>"
            "<title>{{ invoice.title|e }} - {{ company.name|e }}</title>"
            + COMMON_STYLE +
            "<style>"
            "body{" + base_font + "}"
            ".wrap{max-width:900px;margin:0 auto;padding:14px;border:2px solid #111;}"
            ".banner{border:2px solid #111;background:#f2f2f2;padding:12px;display:flex;justify-content:space-between;gap:12px;}"
            ".badge{border:2px solid #111;background:#fff;padding:8px 10px;border-radius:999px;font-weight:800;}"
            ".grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:12px;}"
            ".block{border:1px solid #111;border-radius:12px;padding:10px;}"
            "</style></head><body>"
            "<div class='wrap'>"
            "<div class='banner'>"
            "<div><div class='caps tiny muted'>{{ company.name|e }}</div>"
            "<div class='tiny pre muted' style='margin-top:6px;'>{{ company.address|e }}</div></div>"
            "<div class='right'>"
            "<div class='badge'>{{ invoice.title|e }}</div>"
            "<div class='tiny' style='margin-top:8px;'><b>No</b>: {{ invoice.number|e }}</div>"
            "<div class='tiny'><b>Date</b>: {{ invoice.date|e }}</div>"
            "</div></div>"
            "<div class='grid'>"
            "<div class='block'><div class='caps tiny muted'>Buyer</div><div style='font-weight:800'>{{ buyer.name|e }}</div><div class='tiny pre muted'>{{ buyer.address|e }}</div></div>"
            "<div class='block'><div class='caps tiny muted'>Notify</div><div style='font-weight:800'>{{ notify.name|e }}</div><div class='tiny pre muted'>{{ notify.address|e }}</div></div>"
            "<div class='block'><div class='caps tiny muted'>Shipping</div>"
            "<div class='tiny'><b>Method</b>: {{ invoice.shipment_method|e }}</div>"
            "<div class='tiny'><b>POL</b>: {{ invoice.port_loading|e }}</div>"
            "<div class='tiny'><b>POD</b>: {{ invoice.port_discharge|e }}</div></div>"
            "<div class='block'><div class='caps tiny muted'>Category</div><div class='tiny'>{{ category|e }}</div></div>"
            "</div>"
            "<div style='margin-top:12px;'>" + items_html + "</div>"
            "<div style='display:flex;justify-content:flex-end;margin-top:12px;'>" + totals_box + "</div>"
            "<div style='margin-top:12px;'>" + bank_box + "</div>"
            "</div></body></html>"
        )

    if archetype_id == "A03_ledger_minimal":
        return (
            "<!doctype html><html><head><meta charset='utf-8'/>"
            "<title>{{ invoice.title|e }} - {{ company.name|e }}</title>"
            + COMMON_STYLE +
            "<style>"
            "body{" + base_font + "}"
            ".wrap{max-width:900px;margin:0 auto;padding:14px;}"
            ".top{display:flex;justify-content:space-between;gap:12px;border-bottom:2px solid #111;padding-bottom:10px;}"
            ".tag{border:1px solid #111;padding:6px 8px;background:#f5f5f5;font-weight:800;}"
            ".mini td{padding:4px 0;}"
            "</style></head><body>"
            "<div class='wrap'>"
            "<div class='top'>"
            "<div><div style='font-weight:800;'>{{ company.name|e }}</div><div class='tiny pre muted'>{{ company.address|e }}</div></div>"
            "<div class='right'>"
            "<div class='tag'>{{ invoice.title|e }}</div>"
            "<table class='mini' style='margin-left:auto;font-size:12px;'><tr><td><b>No</b></td><td style='padding-left:10px;'>{{ invoice.number|e }}</td></tr>"
            "<tr><td><b>Date</b></td><td style='padding-left:10px;'>{{ invoice.date|e }}</td></tr>"
            "<tr><td><b>Ship</b></td><td style='padding-left:10px;'>{{ invoice.shipment_method|e }}</td></tr>"
            "<tr><td><b>POL</b></td><td style='padding-left:10px;'>{{ invoice.port_loading|e }}</td></tr>"
            "<tr><td><b>POD</b></td><td style='padding-left:10px;'>{{ invoice.port_discharge|e }}</td></tr></table>"
            "</div></div>"
            "<div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:10px;'>"
            "<div><div class='caps tiny muted'>Buyer</div><div style='font-weight:800;'>{{ buyer.name|e }}</div><div class='tiny pre muted'>{{ buyer.address|e }}</div></div>"
            "<div><div class='caps tiny muted'>Notify</div><div style='font-weight:800;'>{{ notify.name|e }}</div><div class='tiny pre muted'>{{ notify.address|e }}</div></div>"
            "</div>"
            "<div class='hr'></div>"
            "<div class='tiny'><b>Category</b>: {{ category|e }}</div>"
            "<div style='margin-top:10px;'>" + items_html + "</div>"
            "<div style='display:grid;grid-template-columns:1fr 340px;gap:12px;margin-top:12px;'>"
            "<div>" + bank_inline + "</div><div>" + totals_box + "</div>"
            "</div>"
            "</div></body></html>"
        )

    if archetype_id == "A04_ticket_stub":
        return (
            "<!doctype html><html><head><meta charset='utf-8'/>"
            "<title>{{ invoice.title|e }} - {{ company.name|e }}</title>"
            + COMMON_STYLE +
            "<style>"
            "body{" + base_font + "}"
            ".wrap{max-width:940px;margin:0 auto;padding:14px;}"
            ".ticket{border:2px solid #111;display:grid;grid-template-columns:260px 1fr;}"
            ".stub{border-right:2px dashed #111;background:#f5f5f5;padding:12px;}"
            ".main{padding:12px;}"
            ".seal{border:2px solid #111;border-radius:999px;padding:8px 10px;display:inline-block;font-weight:800;}"
            "</style></head><body>"
            "<div class='wrap'>"
            "<div class='ticket'>"
            "<div class='stub'>"
            "<div class='seal'>{{ invoice.title|e }}</div>"
            "<div style='margin-top:10px;font-weight:800;'>{{ company.name|e }}</div>"
            "<div class='tiny pre muted'>{{ company.address|e }}</div>"
            "<div class='hr'></div>"
            "<div class='tiny'><b>No</b>: {{ invoice.number|e }}</div>"
            "<div class='tiny'><b>Date</b>: {{ invoice.date|e }}</div>"
            "<div class='tiny'><b>Ship</b>: {{ invoice.shipment_method|e }}</div>"
            "<div class='tiny'><b>POL</b>: {{ invoice.port_loading|e }}</div>"
            "<div class='tiny'><b>POD</b>: {{ invoice.port_discharge|e }}</div>"
            "</div>"
            "<div class='main'>"
            "<div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;'>"
            "<div style='border:1px dashed #111;padding:10px;border-radius:12px;'><div class='caps tiny muted'>Buyer</div>"
            "<div style='font-weight:800'>{{ buyer.name|e }}</div><div class='tiny pre muted'>{{ buyer.address|e }}</div></div>"
            "<div style='border:1px dashed #111;padding:10px;border-radius:12px;'><div class='caps tiny muted'>Notify</div>"
            "<div style='font-weight:800'>{{ notify.name|e }}</div><div class='tiny pre muted'>{{ notify.address|e }}</div></div>"
            "</div>"
            "<div style='margin-top:10px;'><span class='caps tiny muted'>Category</span>: {{ category|e }}</div>"
            + items_html +
            "<div style='display:grid;grid-template-columns:1fr 360px;gap:12px;margin-top:12px;'>"
            "<div>" + bank_box + "</div><div>" + totals_rail + "</div>"
            "</div>"
            "</div></div></div></body></html>"
        )

    if archetype_id == "A05_magazine_editorial":
        return (
            "<!doctype html><html><head><meta charset='utf-8'/>"
            "<title>{{ invoice.title|e }} - {{ company.name|e }}</title>"
            + COMMON_STYLE +
            "<style>"
            "body{" + base_font + "}"
            ".wrap{max-width:980px;margin:0 auto;padding:16px;}"
            ".mast{border-bottom:2px solid #111;padding-bottom:12px;display:flex;justify-content:space-between;gap:12px;}"
            ".big{font-size:28px;letter-spacing:.12em;text-transform:uppercase;font-weight:800;}"
            ".cols{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-top:12px;}"
            ".box{border:1px solid #111;border-radius:14px;padding:10px;}"
            "</style></head><body>"
            "<div class='wrap'>"
            "<div class='mast'>"
            "<div><div class='caps tiny muted'>{{ company.name|e }}</div><div class='tiny pre muted' style='margin-top:6px;'>{{ company.address|e }}</div></div>"
            "<div class='right'>"
            "<div class='big'>{{ invoice.title|e }}</div>"
            "<div class='tiny muted'>Invoice No. <b>{{ invoice.number|e }}</b></div>"
            "<div class='tiny muted'>Date <b>{{ invoice.date|e }}</b></div>"
            "</div></div>"
            "<div class='cols'>"
            "<div class='box'><div class='caps tiny muted'>Buyer</div><div style='font-weight:800;'>{{ buyer.name|e }}</div><div class='tiny pre muted'>{{ buyer.address|e }}</div></div>"
            "<div class='box'><div class='caps tiny muted'>Notify</div><div style='font-weight:800;'>{{ notify.name|e }}</div><div class='tiny pre muted'>{{ notify.address|e }}</div></div>"
            "<div class='box'><div class='caps tiny muted'>Shipping</div>"
            "<div class='tiny'><b>{{ invoice.shipment_method|e }}</b></div>"
            "<div class='tiny muted'>POL {{ invoice.port_loading|e }}</div>"
            "<div class='tiny muted'>POD {{ invoice.port_discharge|e }}</div></div>"
            "</div>"
            "<div style='margin-top:10px;'><span class='caps tiny muted'>Category</span>: {{ category|e }}</div>"
            + items_html +
            "<div style='display:grid;grid-template-columns:1fr 360px;gap:12px;margin-top:14px;'>"
            "<div>" + bank_box + "</div><div>" + totals_rail + "</div>"
            "</div>"
            "</div></body></html>"
        )

    if archetype_id == "A06_blueprint_technical":
        # NOTE: no gradients (your validator disallows). Use corner marks + dotted lines only.
        return (
            "<!doctype html><html><head><meta charset='utf-8'/>"
            "<title>{{ invoice.title|e }} - {{ company.name|e }}</title>"
            + COMMON_STYLE +
            "<style>"
            "body{" + base_font + "}"
            ".sheet{max-width:980px;margin:0 auto;padding:14px;border:2px solid #111;position:relative;}"
            ".corner{position:absolute;width:18px;height:18px;border:2px solid #111;}"
            ".c1{top:8px;left:8px;border-right:none;border-bottom:none;}"
            ".c2{top:8px;right:8px;border-left:none;border-bottom:none;}"
            ".c3{bottom:8px;left:8px;border-right:none;border-top:none;}"
            ".c4{bottom:8px;right:8px;border-left:none;border-top:none;}"
            ".grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px;}"
            ".cell{border:1px dotted #111;padding:10px;border-radius:10px;}"
            ".label{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:#444;}"
            "</style></head><body>"
            "<div class='sheet'>"
            "<div class='corner c1'></div><div class='corner c2'></div><div class='corner c3'></div><div class='corner c4'></div>"
            "<div style='display:flex;justify-content:space-between;gap:12px;'>"
            "<div><div class='label'>Company</div><div style='font-weight:800;'>{{ company.name|e }}</div>"
            "<div class='tiny pre muted'>{{ company.address|e }}</div></div>"
            "<div class='right'><div style='font-weight:800;letter-spacing:.12em;text-transform:uppercase;'>{{ invoice.title|e }}</div>"
            "<div class='tiny'><b>No</b> {{ invoice.number|e }}</div><div class='tiny'><b>Date</b> {{ invoice.date|e }}</div></div>"
            "</div>"
            "<div class='grid'>"
            "<div class='cell'><div class='label'>Buyer</div><div style='font-weight:800;'>{{ buyer.name|e }}</div><div class='tiny pre muted'>{{ buyer.address|e }}</div></div>"
            "<div class='cell'><div class='label'>Notify</div><div style='font-weight:800;'>{{ notify.name|e }}</div><div class='tiny pre muted'>{{ notify.address|e }}</div></div>"
            "<div class='cell'><div class='label'>Shipping</div><div class='tiny'><b>Method</b> {{ invoice.shipment_method|e }}</div>"
            "<div class='tiny'><b>POL</b> {{ invoice.port_loading|e }}</div><div class='tiny'><b>POD</b> {{ invoice.port_discharge|e }}</div></div>"
            "<div class='cell'><div class='label'>Category</div><div class='tiny'>{{ category|e }}</div></div>"
            "</div>"
            "<div style='margin-top:12px;'>" + ITEMS_FORM_BOXES + "</div>"
            "<div style='display:grid;grid-template-columns:1fr 360px;gap:12px;margin-top:12px;'>"
            "<div>" + bank_box + "</div><div>" + totals_rail + "</div>"
            "</div>"
            "</div></body></html>"
        )

    if archetype_id == "A07_receipt_tall":
        return (
            "<!doctype html><html><head><meta charset='utf-8'/>"
            "<title>{{ invoice.title|e }} - {{ company.name|e }}</title>"
            + COMMON_STYLE +
            "<style>"
            "body{" + base_font + "}"
            ".receipt{max-width:520px;margin:0 auto;border:2px solid #111;padding:14px;}"
            ".double{border-top:3px double #111;margin:12px 0;}"
            ".big{font-size:22px;font-weight:900;letter-spacing:.1em;text-transform:uppercase;}"
            "</style></head><body>"
            "<div class='receipt'>"
            "<div class='center big'>{{ invoice.title|e }}</div>"
            "<div class='center tiny muted' style='margin-top:6px;'>{{ company.name|e }}</div>"
            "<div class='center tiny pre muted'>{{ company.address|e }}</div>"
            "<div class='double'></div>"
            "<div class='tiny'><b>No</b>: {{ invoice.number|e }} &nbsp; <b>Date</b>: {{ invoice.date|e }}</div>"
            "<div class='tiny'><b>Ship</b>: {{ invoice.shipment_method|e }}</div>"
            "<div class='tiny'><b>POL</b>: {{ invoice.port_loading|e }} &nbsp; <b>POD</b>: {{ invoice.port_discharge|e }}</div>"
            "<div class='double'></div>"
            "<div class='tiny'><b>Buyer</b>: {{ buyer.name|e }}</div><div class='tiny pre muted'>{{ buyer.address|e }}</div>"
            "<div class='tiny' style='margin-top:8px;'><b>Notify</b>: {{ notify.name|e }}</div><div class='tiny pre muted'>{{ notify.address|e }}</div>"
            "<div class='double'></div>"
            "<div class='tiny'><b>Category</b>: {{ category|e }}</div>"
            + ITEMS_TICKET_LINES +
            "<div class='double'></div>"
            + totals_rail +
            "<div class='double'></div>"
            + bank_inline +
            "</div></body></html>"
        )

    if archetype_id == "A08_two_column_classic":
        return (
            "<!doctype html><html><head><meta charset='utf-8'/>"
            "<title>{{ invoice.title|e }} - {{ company.name|e }}</title>"
            + COMMON_STYLE +
            "<style>"
            "body{font-family:Times New Roman,Times,serif;}"
            ".wrap{max-width:980px;margin:0 auto;padding:14px;}"
            ".head{display:grid;grid-template-columns:1.2fr .8fr;gap:14px;border-bottom:2px solid #111;padding-bottom:10px;}"
            ".title{font-family:Arial,Helvetica,sans-serif;font-weight:900;letter-spacing:.12em;text-transform:uppercase;}"
            "</style></head><body>"
            "<div class='wrap'>"
            "<div class='head'>"
            "<div><div class='title'>{{ company.name|e }}</div><div class='tiny pre muted'>{{ company.address|e }}</div></div>"
            "<div class='right'><div class='title'>{{ invoice.title|e }}</div>"
            "<div class='tiny'><b>No</b>: {{ invoice.number|e }}</div>"
            "<div class='tiny'><b>Date</b>: {{ invoice.date|e }}</div>"
            "<div class='tiny'><b>Ship</b>: {{ invoice.shipment_method|e }}</div>"
            "<div class='tiny'><b>POL</b>: {{ invoice.port_loading|e }}</div>"
            "<div class='tiny'><b>POD</b>: {{ invoice.port_discharge|e }}</div>"
            "</div></div>"
            "<div style='display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:12px;'>"
            "<div><div class='caps tiny muted'>Buyer</div><div style='font-weight:800;'>{{ buyer.name|e }}</div><div class='tiny pre muted'>{{ buyer.address|e }}</div></div>"
            "<div><div class='caps tiny muted'>Notify</div><div style='font-weight:800;'>{{ notify.name|e }}</div><div class='tiny pre muted'>{{ notify.address|e }}</div></div>"
            "</div>"
            "<div class='hr'></div>"
            "<div class='tiny'><b>Category</b>: {{ category|e }}</div>"
            "<div style='margin-top:10px;'>" + items_html + "</div>"
            "<div style='display:flex;justify-content:flex-end;margin-top:12px;'>" + totals_box + "</div>"
            "<div style='margin-top:12px;'>" + bank_box + "</div>"
            "</div></body></html>"
        )

    if archetype_id == "A09_cards_stack":
        return (
            "<!doctype html><html><head><meta charset='utf-8'/>"
            "<title>{{ invoice.title|e }} - {{ company.name|e }}</title>"
            + COMMON_STYLE +
            "<style>"
            "body{" + base_font + "}"
            ".wrap{max-width:980px;margin:0 auto;padding:14px;background:#fff;}"
            ".card{border:1px solid #111;border-radius:16px;padding:12px;margin-top:12px;}"
            ".row{display:flex;justify-content:space-between;gap:12px;}"
            ".pill{border:2px solid #111;border-radius:999px;padding:8px 10px;font-weight:900;}"
            "</style></head><body>"
            "<div class='wrap'>"
            "<div class='card'>"
            "<div class='row'>"
            "<div><div class='caps tiny muted'>{{ company.name|e }}</div><div class='tiny pre muted'>{{ company.address|e }}</div></div>"
            "<div class='right'><div class='pill'>{{ invoice.title|e }}</div>"
            "<div class='tiny' style='margin-top:8px;'><b>No</b>: {{ invoice.number|e }}</div>"
            "<div class='tiny'><b>Date</b>: {{ invoice.date|e }}</div></div>"
            "</div></div>"
            "<div class='card'><div class='row'>"
            "<div style='flex:1;'><div class='caps tiny muted'>Buyer</div><div style='font-weight:800;'>{{ buyer.name|e }}</div><div class='tiny pre muted'>{{ buyer.address|e }}</div></div>"
            "<div style='flex:1;'><div class='caps tiny muted'>Notify</div><div style='font-weight:800;'>{{ notify.name|e }}</div><div class='tiny pre muted'>{{ notify.address|e }}</div></div>"
            "</div></div>"
            "<div class='card'><div class='row'>"
            "<div><span class='caps tiny muted'>Category</span>: {{ category|e }}</div>"
            "<div class='right'><span class='caps tiny muted'>Ship</span>: {{ invoice.shipment_method|e }} | "
            "<span class='caps tiny muted'>POL</span>: {{ invoice.port_loading|e }} | "
            "<span class='caps tiny muted'>POD</span>: {{ invoice.port_discharge|e }}</div>"
            "</div></div>"
            "<div class='card'>" + items_html + "</div>"
            "<div class='card'><div style='display:grid;grid-template-columns:1fr 360px;gap:12px;'>"
            "<div>" + bank_box + "</div><div>" + totals_rail + "</div></div></div>"
            "</div></body></html>"
        )

    if archetype_id == "A10_vertical_title_strip":
        return (
            "<!doctype html><html><head><meta charset='utf-8'/>"
            "<title>{{ invoice.title|e }} - {{ company.name|e }}</title>"
            + COMMON_STYLE +
            "<style>"
            "body{" + base_font + "}"
            ".wrap{max-width:980px;margin:0 auto;border:2px solid #111;display:grid;grid-template-columns:54px 1fr;}"
            ".strip{writing-mode:vertical-rl;transform:rotate(180deg);background:#111;color:#fff;"
            "display:flex;align-items:center;justify-content:center;font-weight:900;letter-spacing:.18em;text-transform:uppercase;}"
            ".main{padding:14px;}"
            "</style></head><body>"
            "<div class='wrap'>"
            "<div class='strip'>{{ invoice.title|e }}</div>"
            "<div class='main'>"
            "<div style='display:flex;justify-content:space-between;gap:12px;border-bottom:2px solid #111;padding-bottom:10px;'>"
            "<div><div style='font-weight:900;font-size:18px;'>{{ company.name|e }}</div><div class='tiny pre muted'>{{ company.address|e }}</div></div>"
            "<div class='right tiny'>"
            "<div><b>No</b>: {{ invoice.number|e }}</div><div><b>Date</b>: {{ invoice.date|e }}</div>"
            "<div><b>Ship</b>: {{ invoice.shipment_method|e }}</div>"
            "<div><b>POL</b>: {{ invoice.port_loading|e }}</div><div><b>POD</b>: {{ invoice.port_discharge|e }}</div>"
            "</div></div>"
            "<div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px;'>"
            "<div style='border:1px solid #111;padding:10px;border-radius:12px;'><div class='caps tiny muted'>Buyer</div><div style='font-weight:800'>{{ buyer.name|e }}</div><div class='tiny pre muted'>{{ buyer.address|e }}</div></div>"
            "<div style='border:1px solid #111;padding:10px;border-radius:12px;'><div class='caps tiny muted'>Notify</div><div style='font-weight:800'>{{ notify.name|e }}</div><div class='tiny pre muted'>{{ notify.address|e }}</div></div>"
            "</div>"
            "<div style='margin-top:10px;'><span class='caps tiny muted'>Category</span>: {{ category|e }}</div>"
            "<div style='margin-top:10px;'>" + items_html + "</div>"
            "<div style='display:grid;grid-template-columns:1fr 360px;gap:12px;margin-top:12px;'>"
            "<div>" + bank_inline + "</div><div>" + totals_rail + "</div>"
            "</div>"
            "</div></div></body></html>"
        )

    if archetype_id == "A11_stamp_seal":
        return (
            "<!doctype html><html><head><meta charset='utf-8'/>"
            "<title>{{ invoice.title|e }} - {{ company.name|e }}</title>"
            + COMMON_STYLE +
            "<style>"
            "body{" + base_font + "}"
            ".wrap{max-width:980px;margin:0 auto;padding:14px;}"
            ".seal{width:110px;height:110px;border:3px solid #111;border-radius:999px;"
            "display:flex;align-items:center;justify-content:center;text-align:center;font-weight:900;"
            "letter-spacing:.12em;text-transform:uppercase;font-size:11px;background:#f5f5f5;}"
            ".head{display:flex;justify-content:space-between;gap:12px;align-items:center;border-bottom:2px solid #111;padding-bottom:10px;}"
            ".grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px;}"
            ".box{border:1px solid #111;border-radius:14px;padding:10px;}"
            "</style></head><body>"
            "<div class='wrap'>"
            "<div class='head'>"
            "<div><div style='font-weight:900;font-size:18px;'>{{ company.name|e }}</div><div class='tiny pre muted'>{{ company.address|e }}</div></div>"
            "<div class='seal'>{{ invoice.title|e }}<br><span style='letter-spacing:0;'>{{ invoice.number|e }}</span></div>"
            "<div class='right tiny'>"
            "<div><b>Date</b>: {{ invoice.date|e }}</div>"
            "<div><b>Ship</b>: {{ invoice.shipment_method|e }}</div>"
            "<div><b>POL</b>: {{ invoice.port_loading|e }}</div>"
            "<div><b>POD</b>: {{ invoice.port_discharge|e }}</div>"
            "</div></div>"
            "<div class='grid'>"
            "<div class='box'><div class='caps tiny muted'>Buyer</div><div style='font-weight:800'>{{ buyer.name|e }}</div><div class='tiny pre muted'>{{ buyer.address|e }}</div></div>"
            "<div class='box'><div class='caps tiny muted'>Notify</div><div style='font-weight:800'>{{ notify.name|e }}</div><div class='tiny pre muted'>{{ notify.address|e }}</div></div>"
            "</div>"
            "<div style='margin-top:10px;'><span class='caps tiny muted'>Category</span>: {{ category|e }}</div>"
            + items_html +
            "<div style='display:grid;grid-template-columns:1fr 360px;gap:12px;margin-top:12px;'>"
            "<div>" + bank_box + "</div><div>" + totals_rail + "</div>"
            "</div>"
            "</div></body></html>"
        )

    if archetype_id == "A12_corner_frame":
        return (
            "<!doctype html><html><head><meta charset='utf-8'/>"
            "<title>{{ invoice.title|e }} - {{ company.name|e }}</title>"
            + COMMON_STYLE +
            "<style>"
            "body{" + base_font + "}"
            ".frame{max-width:980px;margin:0 auto;padding:18px;border:2px solid #111;position:relative;}"
            ".f{position:absolute;width:26px;height:26px;border:2px solid #111;}"
            ".tl{top:10px;left:10px;border-right:none;border-bottom:none;}"
            ".tr{top:10px;right:10px;border-left:none;border-bottom:none;}"
            ".bl{bottom:10px;left:10px;border-right:none;border-top:none;}"
            ".br{bottom:10px;right:10px;border-left:none;border-top:none;}"
            "</style></head><body>"
            "<div class='frame'>"
            "<div class='f tl'></div><div class='f tr'></div><div class='f bl'></div><div class='f br'></div>"
            "<div style='text-align:center;'>"
            "<div class='caps tiny muted'>{{ company.name|e }}</div>"
            "<div class='tiny pre muted' style='margin-top:6px;'>{{ company.address|e }}</div>"
            "<div style='margin-top:10px;font-weight:900;letter-spacing:.14em;text-transform:uppercase;'>{{ invoice.title|e }}</div>"
            "<div class='tiny muted'>No {{ invoice.number|e }} · {{ invoice.date|e }}</div>"
            "</div>"
            "<div class='hr'></div>"
            "<div style='display:grid;grid-template-columns:1fr 1fr;gap:14px;'>"
            "<div><div class='caps tiny muted'>Buyer</div><div style='font-weight:800;'>{{ buyer.name|e }}</div><div class='tiny pre muted'>{{ buyer.address|e }}</div></div>"
            "<div><div class='caps tiny muted'>Notify</div><div style='font-weight:800;'>{{ notify.name|e }}</div><div class='tiny pre muted'>{{ notify.address|e }}</div></div>"
            "</div>"
            "<div style='margin-top:10px;'><span class='caps tiny muted'>Shipping</span>: {{ invoice.shipment_method|e }} | "
            "<span class='caps tiny muted'>POL</span>: {{ invoice.port_loading|e }} | "
            "<span class='caps tiny muted'>POD</span>: {{ invoice.port_discharge|e }}</div>"
            "<div style='margin-top:6px;'><span class='caps tiny muted'>Category</span>: {{ category|e }}</div>"
            "<div style='margin-top:12px;'>" + items_html + "</div>"
            "<div style='display:flex;justify-content:flex-end;margin-top:12px;'>" + totals_box + "</div>"
            "<div style='margin-top:12px;'>" + bank_box + "</div>"
            "</div></body></html>"
        )

    if archetype_id == "A13_split_items_totals":
        return (
            "<!doctype html><html><head><meta charset='utf-8'/>"
            "<title>{{ invoice.title|e }} - {{ company.name|e }}</title>"
            + COMMON_STYLE +
            "<style>"
            "body{" + base_font + "}"
            ".wrap{max-width:1100px;margin:0 auto;padding:14px;}"
            ".top{border-bottom:2px solid #111;padding-bottom:10px;display:flex;justify-content:space-between;gap:12px;}"
            ".layout{display:grid;grid-template-columns:1.4fr .6fr;gap:12px;margin-top:12px;}"
            ".rail{border-left:2px solid #111;padding-left:12px;}"
            "</style></head><body>"
            "<div class='wrap'>"
            "<div class='top'>"
            "<div><div style='font-weight:900;font-size:18px;'>{{ company.name|e }}</div><div class='tiny pre muted'>{{ company.address|e }}</div></div>"
            "<div class='right'><div style='font-weight:900;letter-spacing:.14em;text-transform:uppercase;'>{{ invoice.title|e }}</div>"
            "<div class='tiny'><b>No</b>: {{ invoice.number|e }}</div><div class='tiny'><b>Date</b>: {{ invoice.date|e }}</div></div>"
            "</div>"
            "<div class='layout'>"
            "<div>"
            "<div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;'>"
            "<div style='border:1px solid #111;border-radius:12px;padding:10px;'><div class='caps tiny muted'>Buyer</div><div style='font-weight:800;'>{{ buyer.name|e }}</div><div class='tiny pre muted'>{{ buyer.address|e }}</div></div>"
            "<div style='border:1px solid #111;border-radius:12px;padding:10px;'><div class='caps tiny muted'>Notify</div><div style='font-weight:800;'>{{ notify.name|e }}</div><div class='tiny pre muted'>{{ notify.address|e }}</div></div>"
            "</div>"
            "<div style='margin-top:10px;'><span class='caps tiny muted'>Category</span>: {{ category|e }}</div>"
            "<div style='margin-top:10px;'>" + items_html + "</div>"
            "</div>"
            "<div class='rail'>"
            + totals_rail +
            "<div style='margin-top:12px;'>" + bank_box + "</div>"
            "</div>"
            "</div>"
            "</div></body></html>"
        )

    if archetype_id == "A14_microgrid_dense":
        return (
            "<!doctype html><html><head><meta charset='utf-8'/>"
            "<title>{{ invoice.title|e }} - {{ company.name|e }}</title>"
            + COMMON_STYLE +
            "<style>"
            "body{font-family:Courier New,Courier,monospace;}"
            ".wrap{max-width:980px;margin:0 auto;padding:10px;border:2px solid #111;}"
            ".grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-top:8px;}"
            ".cell{border:1px solid #111;padding:8px;}"
            ".k{font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:#444;}"
            ".v{font-size:12px;}"
            "</style></head><body>"
            "<div class='wrap'>"
            "<div style='display:flex;justify-content:space-between;gap:10px;'>"
            "<div><div class='k'>Company</div><div class='v' style='font-weight:900;'>{{ company.name|e }}</div>"
            "<div class='tiny pre muted'>{{ company.address|e }}</div></div>"
            "<div class='right'><div class='k'>Doc</div><div class='v' style='font-weight:900;'>{{ invoice.title|e }}</div>"
            "<div class='tiny'><b>No</b> {{ invoice.number|e }}</div><div class='tiny'><b>Date</b> {{ invoice.date|e }}</div></div>"
            "</div>"
            "<div class='grid'>"
            "<div class='cell'><div class='k'>Buyer</div><div class='v' style='font-weight:900;'>{{ buyer.name|e }}</div><div class='tiny pre muted'>{{ buyer.address|e }}</div></div>"
            "<div class='cell'><div class='k'>Notify</div><div class='v' style='font-weight:900;'>{{ notify.name|e }}</div><div class='tiny pre muted'>{{ notify.address|e }}</div></div>"
            "<div class='cell'><div class='k'>Shipping</div><div class='tiny'><b>M</b> {{ invoice.shipment_method|e }}</div><div class='tiny'><b>POL</b> {{ invoice.port_loading|e }}</div><div class='tiny'><b>POD</b> {{ invoice.port_discharge|e }}</div></div>"
            "</div>"
            "<div style='margin-top:8px;'><span class='k'>Category</span> <span class='v'>{{ category|e }}</span></div>"
            "<div style='margin-top:8px;'>" + ITEMS_FORM_BOXES + "</div>"
            "<div style='display:grid;grid-template-columns:1fr 340px;gap:10px;margin-top:10px;'>"
            "<div>" + bank_inline + "</div><div>" + totals_box + "</div>"
            "</div>"
            "</div></body></html>"
        )

    if archetype_id == "A15_outline_only_borderless":
        return (
            "<!doctype html><html><head><meta charset='utf-8'/>"
            "<title>{{ invoice.title|e }} - {{ company.name|e }}</title>"
            + COMMON_STYLE +
            "<style>"
            "body{" + base_font + "}"
            ".wrap{max-width:980px;margin:0 auto;padding:18px;}"
            ".hair{border-top:1px solid #111;margin:14px 0;}"
            ".title{font-weight:900;font-size:22px;letter-spacing:.16em;text-transform:uppercase;}"
            "</style></head><body>"
            "<div class='wrap'>"
            "<div style='display:flex;justify-content:space-between;gap:12px;'>"
            "<div><div style='font-weight:900;font-size:18px;'>{{ company.name|e }}</div><div class='tiny pre muted'>{{ company.address|e }}</div></div>"
            "<div class='right'><div class='title'>{{ invoice.title|e }}</div>"
            "<div class='tiny muted'>No {{ invoice.number|e }}</div><div class='tiny muted'>{{ invoice.date|e }}</div></div>"
            "</div>"
            "<div class='hair'></div>"
            "<div style='display:grid;grid-template-columns:1fr 1fr;gap:24px;'>"
            "<div><div class='caps tiny muted'>Buyer</div><div style='font-weight:800;'>{{ buyer.name|e }}</div><div class='tiny pre muted'>{{ buyer.address|e }}</div></div>"
            "<div><div class='caps tiny muted'>Notify</div><div style='font-weight:800;'>{{ notify.name|e }}</div><div class='tiny pre muted'>{{ notify.address|e }}</div></div>"
            "</div>"
            "<div class='hair'></div>"
            "<div class='tiny muted'>Shipment: <b>{{ invoice.shipment_method|e }}</b> · POL <b>{{ invoice.port_loading|e }}</b> · POD <b>{{ invoice.port_discharge|e }}</b></div>"
            "<div class='tiny muted' style='margin-top:6px;'>Category: <b>{{ category|e }}</b></div>"
            + items_html +
            "<div class='hair'></div>"
            "<div style='display:grid;grid-template-columns:1fr 360px;gap:24px;'>"
            "<div>" + bank_inline + "</div><div>" + totals_rail + "</div>"
            "</div>"
            "</div></body></html>"
        )

    if archetype_id == "A16_typographic_hierarchy":
        return (
            "<!doctype html><html><head><meta charset='utf-8'/>"
            "<title>{{ invoice.title|e }} - {{ company.name|e }}</title>"
            + COMMON_STYLE +
            "<style>"
            "body{" + base_font + "}"
            ".wrap{max-width:980px;margin:0 auto;padding:16px;}"
            ".hero{display:grid;grid-template-columns:1fr 1fr;gap:12px;align-items:end;border-bottom:2px solid #111;padding-bottom:12px;}"
            ".mega{font-size:48px;line-height:1;font-weight:900;letter-spacing:.04em;}"
            ".sub{font-size:12px;color:#444;letter-spacing:.12em;text-transform:uppercase;}"
            "</style></head><body>"
            "<div class='wrap'>"
            "<div class='hero'>"
            "<div><div class='sub'>Invoice No</div><div class='mega'>{{ invoice.number|e }}</div></div>"
            "<div class='right'><div class='sub'>Grand Total</div><div class='mega'>{{ totals.grand_total }}</div></div>"
            "</div>"
            "<div style='display:flex;justify-content:space-between;gap:12px;margin-top:10px;'>"
            "<div><div style='font-weight:900;font-size:18px;'>{{ company.name|e }}</div><div class='tiny pre muted'>{{ company.address|e }}</div></div>"
            "<div class='right tiny'>"
            "<div style='font-weight:900;letter-spacing:.16em;text-transform:uppercase;'>{{ invoice.title|e }}</div>"
            "<div><b>Date</b>: {{ invoice.date|e }}</div>"
            "<div><b>Ship</b>: {{ invoice.shipment_method|e }}</div>"
            "<div><b>POL</b>: {{ invoice.port_loading|e }}</div>"
            "<div><b>POD</b>: {{ invoice.port_discharge|e }}</div>"
            "</div></div>"
            "<div style='display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:12px;'>"
            "<div style='border:1px solid #111;border-radius:12px;padding:10px;'><div class='caps tiny muted'>Buyer</div><div style='font-weight:800;'>{{ buyer.name|e }}</div><div class='tiny pre muted'>{{ buyer.address|e }}</div></div>"
            "<div style='border:1px solid #111;border-radius:12px;padding:10px;'><div class='caps tiny muted'>Notify</div><div style='font-weight:800;'>{{ notify.name|e }}</div><div class='tiny pre muted'>{{ notify.address|e }}</div></div>"
            "</div>"
            "<div style='margin-top:10px;'><span class='caps tiny muted'>Category</span>: {{ category|e }}</div>"
            + ITEMS_TICKET_LINES +
            "<div style='display:grid;grid-template-columns:1fr 360px;gap:12px;margin-top:12px;'>"
            "<div>" + bank_box + "</div><div>" + totals_box + "</div>"
            "</div>"
            "</div></body></html>"
        )

    if archetype_id == "A17_diagonal_header":
        return (
            "<!doctype html><html><head><meta charset='utf-8'/>"
            "<title>{{ invoice.title|e }} - {{ company.name|e }}</title>"
            + COMMON_STYLE +
            "<style>"
            "body{" + base_font + "}"
            ".wrap{max-width:980px;margin:0 auto;border:2px solid #111;overflow:hidden;}"
            ".diag{background:#111;color:#fff;padding:14px;position:relative;}"
            ".diag:after{content:'';position:absolute;right:-60px;top:0;width:120px;height:120px;background:#111;transform:rotate(45deg);}"
            ".main{padding:14px;}"
            ".white{color:#fff;}"
            "</style></head><body>"
            "<div class='wrap'>"
            "<div class='diag'>"
            "<div style='position:relative;z-index:2;display:flex;justify-content:space-between;gap:12px;'>"
            "<div><div class='white' style='font-weight:900;font-size:18px;'>{{ company.name|e }}</div><div class='tiny pre' style='color:#e9e9e9;'>{{ company.address|e }}</div></div>"
            "<div class='right'><div class='white' style='font-weight:900;letter-spacing:.14em;text-transform:uppercase;'>{{ invoice.title|e }}</div>"
            "<div class='tiny' style='color:#e9e9e9;'><b>No</b>: {{ invoice.number|e }}</div>"
            "<div class='tiny' style='color:#e9e9e9;'><b>Date</b>: {{ invoice.date|e }}</div></div>"
            "</div></div>"
            "<div class='main'>"
            "<div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;'>"
            "<div style='border:1px solid #111;border-radius:12px;padding:10px;'><div class='caps tiny muted'>Buyer</div><div style='font-weight:800'>{{ buyer.name|e }}</div><div class='tiny pre muted'>{{ buyer.address|e }}</div></div>"
            "<div style='border:1px solid #111;border-radius:12px;padding:10px;'><div class='caps tiny muted'>Notify</div><div style='font-weight:800'>{{ notify.name|e }}</div><div class='tiny pre muted'>{{ notify.address|e }}</div></div>"
            "</div>"
            "<div style='margin-top:10px;' class='tiny muted'>Ship <b>{{ invoice.shipment_method|e }}</b> · POL <b>{{ invoice.port_loading|e }}</b> · POD <b>{{ invoice.port_discharge|e }}</b></div>"
            "<div style='margin-top:6px;'><span class='caps tiny muted'>Category</span>: {{ category|e }}</div>"
            "<div style='margin-top:12px;'>" + items_html + "</div>"
            "<div style='display:grid;grid-template-columns:1fr 360px;gap:12px;margin-top:12px;'>"
            "<div>" + bank_box + "</div><div>" + totals_rail + "</div>"
            "</div>"
            "</div></div></body></html>"
        )

    # A18_form_like_boxes
    return (
        "<!doctype html><html><head><meta charset='utf-8'/>"
        "<title>{{ invoice.title|e }} - {{ company.name|e }}</title>"
        + COMMON_STYLE +
        "<style>"
        "body{" + base_font + "}"
        ".wrap{max-width:980px;margin:0 auto;padding:14px;border:2px solid #111;}"
        ".box{border:1px solid #111;padding:10px;}"
        ".grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px;}"
        ".label{font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:#444;}"
        ".val{font-size:12px;}"
        "</style></head><body>"
        "<div class='wrap'>"
        "<div class='box'>"
        "<div class='grid'>"
        "<div><div class='label'>Company Name</div><div class='val' style='font-weight:900;'>{{ company.name|e }}</div></div>"
        "<div class='right'><div class='label'>Document</div><div class='val' style='font-weight:900;'>{{ invoice.title|e }}</div></div>"
        "<div style='grid-column:1/3;'><div class='label'>Company Address</div><div class='val pre muted'>{{ company.address|e }}</div></div>"
        "</div></div>"
        "<div class='grid'>"
        "<div class='box'><div class='label'>Invoice No</div><div class='val'>{{ invoice.number|e }}</div></div>"
        "<div class='box'><div class='label'>Invoice Date</div><div class='val'>{{ invoice.date|e }}</div></div>"
        "<div class='box'><div class='label'>Shipment Method</div><div class='val'>{{ invoice.shipment_method|e }}</div></div>"
        "<div class='box'><div class='label'>Ports</div><div class='val'>POL {{ invoice.port_loading|e }} · POD {{ invoice.port_discharge|e }}</div></div>"
        "</div>"
        "<div class='grid'>"
        "<div class='box'><div class='label'>Buyer</div><div class='val' style='font-weight:900;'>{{ buyer.name|e }}</div><div class='val pre muted'>{{ buyer.address|e }}</div></div>"
        "<div class='box'><div class='label'>Notify</div><div class='val' style='font-weight:900;'>{{ notify.name|e }}</div><div class='val pre muted'>{{ notify.address|e }}</div></div>"
        "</div>"
        "<div class='box' style='margin-top:10px;'><div class='label'>Category</div><div class='val'>{{ category|e }}</div></div>"
        "<div style='margin-top:10px;'>" + ITEMS_FORM_BOXES + "</div>"
        "<div class='grid' style='margin-top:10px;'>"
        "<div class='box'>" + bank_box + "</div>"
        "<div class='box'>" + totals_rail + "</div>"
        "</div>"
        "</div></body></html>"
    )

# ----------------------------
# Template generation pipeline
# ----------------------------
def choose_candidate_archetypes(company_slug: str, k: int = 12) -> List[dict]:
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
    existing = list_existing_templates()

    best = None
    debug = {"candidates": []}

    for recipe, html in candidates:
        errs = []
        errs += validate_monochrome_and_print(html)
        errs += validate_jinja(html)

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
        raise ValueError("All generated candidates failed validation. See debug info.")

    # Strict novelty gate (tune threshold if needed)
    if best["sim"] > 0.42:
        debug["note"] = "Best candidate still too similar; increase K or strengthen prompt."
        raise ValueError("Templates are still too similar to existing ones. Increase candidate count K or strengthen prompt.")

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

if __name__ == "__main__":
    # Run:  OPENAI_API_KEY=... python app.py
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5055")), debug=True)
