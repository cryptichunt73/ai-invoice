# app.py — Invoice Generator Pro (Monochrome, Locked Templates, UI from templates/index.html)

import os, json, pathlib, re, random, hashlib, textwrap, html as _html
from datetime import datetime
from typing import Any, Dict, List
from decimal import Decimal, InvalidOperation
from flask import Flask, render_template, request, jsonify, url_for, send_from_directory, abort
from jinja2 import Template
from openai import OpenAI
from google import genai
from google.genai import types
from concurrent.futures import ThreadPoolExecutor, as_completed
import re  # for wattage parsing (guardrails)
from decimal import Decimal, ROUND_FLOOR, ROUND_HALF_UP


app = Flask(__name__)


# Route for the root URL (so opening the site works)


# ------------------- Config -------------------
APP_PORT = int(os.getenv("PORT", "5055"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# OpenAI client (reads OPENAI_API_KEY from env)

client = OpenAI(api_key=OPENAI_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
DATA_DIR = pathlib.Path("data/templates")
TPL_HTML_DIR = pathlib.Path("data/templates_html")
INVOICE_DIR = pathlib.Path("html_invoices")
for d in (DATA_DIR, TPL_HTML_DIR, INVOICE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ... all your other imports stay the same ...


# --------------------------------------------------
# 1. Gemini invoice item generation
# --------------------------------------------------
def _d2(x: Any) -> Decimal:
    try:
        return Decimal(str(x).replace(",", "")).quantize(Decimal("0.01"))
    except Exception:
        return Decimal("0.00")

def _scale_to_total(items: List[Dict[str, Any]], target_total: Decimal) -> List[Dict[str, Any]]:
    """Scale 'amount' so sum == target_total."""
    if not items:
        return items
    current = sum(_d2(i.get("amount", "0")) for i in items)
    if current <= 0:
        per = (target_total / Decimal(len(items))).quantize(Decimal("0.01"))
        for it in items:
            it["amount"] = f"{per:.2f}"
    else:
        ratio = target_total / current
        for it in items:
            amt = (_d2(it.get("amount", "0")) * ratio).quantize(Decimal("0.01"))
            it["amount"] = f"{amt:.2f}"

    # recompute rate
    for idx, it in enumerate(items, 1):
        q = _d2(it.get("quantity", "1")) or Decimal("1")
        amt = _d2(it.get("amount", "0"))
        it["rate"] = f"{(amt/q).quantize(Decimal('0.01')):.2f}" if q > 0 else "0.00"
        it["quantity"] = str(int(q)) if q % 1 == 0 else str(q)
        it["unit"] = it.get("unit") or "NOS"
        it["item_no"] = it.get("item_no") or f"SKU{idx:03d}"
        it["description"] = it.get("description") or "Product"
    return items

def _safe_decimal(x, default="0.00"):
    try:
        return Decimal(str(x).replace(",", "")).quantize(Decimal("0.01"))
    except Exception:
        return Decimal(default)

def _median(nums: List[Decimal]) -> Decimal:
    arr = sorted(nums)
    n = len(arr)
    if n == 0: return Decimal("0.00")
    mid = n // 2
    return (arr[mid] if n % 2 else (arr[mid-1] + arr[mid]) / 2).quantize(Decimal("0.01"))





def generate_items_with_grounding(
    company: str,
    total: str,
    currency: str = "USD",
    n_items: int = 6,
) -> Dict[str, Any]:
    """
    Use Gemini + Google Search grounding to propose items.
    **Use web_price as the unit rate exactly** (no discounts/other modifications),
    then compute integer quantities to approximate the requested total.
    Returns:
        {
          "items": [{ "slno","item_no","description","quantity","unit","rate","amount","source_url","source_note" }, ...],
          "total": "######.##"
        }
    """
    if not gemini_client:
      raise RuntimeError("GEMINI_API_KEY missing")

    target_total = _d2(total)

    # --- Prompt: ask Gemini for items with web_price only (no amount/rate) ---
    prompt = f"""
You are an expert B2B invoicing assistant.
Use Google Search to understand what "{company}" sells.
Propose {n_items} plausible line items for a wholesale B2B order in {currency}.

For EACH item return ONLY these fields (as JSON):
- "item_no": short SKU-like code (e.g., "SKU001")
- "description": product name (include wattage/capacity if relevant)
- "quantity": an integer suggestion (e.g., "10")
- "unit": usually "NOS" (or meter/pack if clearly appropriate)
- "web_price": a realistic unit price in {currency}, based on a similar product online (ex-tax, ex-shipping)
- "source_url": the page where you found the price
- "source_note": a short note like brand/model or dealer page

Strictly return JSON:
{{
  "items": [
    {{
      "item_no": "SKU001",
      "description": "LED Panel Light 2x2 40W Neutral White",
      "quantity": "120",
      "unit": "NOS",
      "web_price": "12.50",
      "source_url": "https://example.com/panel-40w",
      "source_note": "Distributor price; 40W panel"
    }}
  ]
}}
"""

    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    config = types.GenerateContentConfig(tools=[grounding_tool])

    resp = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=config,
    )

    raw = resp.text or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        s, e = raw.find("{"), raw.rfind("}")
        data = json.loads(raw[s:e+1]) if (s != -1 and e != -1 and e > s) else {"items": []}

    raw_items = data.get("items", []) or []

    # --- Build items; set rate = web_price exactly; collect initial qty suggestions ---
    items: List[Dict[str, Any]] = []
    for i, it in enumerate(raw_items, 1):
        rate = _safe_decimal(it.get("web_price", "0"))
        qty_suggested = _safe_decimal(it.get("quantity", "1"))
        qty0 = int(qty_suggested) if qty_suggested > 0 else 1  # ensure >=1
        item = {
            "slno": str(i),
            "item": it,  # keep original for fields
            "rate": rate,                 # use web_price as rate directly (no discount/other mods)
            "qty": qty0,
        }
        items.append(item)

    # If all web_price are zero or missing, avoid division by zero by forcing rate=1, qty=1
    if all(x["rate"] == 0 for x in items):
        for x in items:
            x["rate"] = Decimal("1.00")

    # --- Compute integer quantities to match target_total as closely as possible ---
    # Start from suggested quantities, scale them to target_total, then adjust the last line to absorb rounding.
    if items:
        base_sum = sum(x["rate"] * Decimal(x["qty"]) for x in items)
        if base_sum <= 0:
            # fallback: start with 1 each if all zero
            for x in items: 
                x["qty"] = 1
            base_sum = sum(x["rate"] * Decimal(x["qty"]) for x in items)

        if base_sum > 0:
            scale = (target_total / base_sum) if target_total > 0 else Decimal("1")
            # first pass: floor-scale all but last
            running = Decimal("0.00")
            for x in items[:-1]:
                scaled_q = (Decimal(x["qty"]) * scale).to_integral_value(rounding=ROUND_FLOOR)
                x["qty"] = int(max(1, scaled_q))
                running += x["rate"] * Decimal(x["qty"])

            last = items[-1]
            # choose last qty to get as close as possible to target
            remaining = target_total - running
            if last["rate"] > 0:
                q_last = (remaining / last["rate"]).to_integral_value(rounding=ROUND_HALF_UP)
                last["qty"] = int(max(1, q_last))
            # nothing else changes

    # --- Finalize amounts and output shape; compute achieved total (may differ slightly due to integer qty) ---
    out_items: List[Dict[str, str]] = []
    for i, x in enumerate(items, 1):
        rate = x["rate"]
        qty = max(1, int(x["qty"]))
        amt = (rate * Decimal(qty)).quantize(Decimal("0.01"))
        it = x["item"]
        out_items.append({
            "slno": str(i),
            "item_no": str(it.get("item_no") or f"SKU{i:03d}"),
            "description": str(it.get("description") or "Product"),
            "quantity": str(qty),
            "unit": str(it.get("unit") or "NOS"),
            "rate": f"{rate:.2f}",             # = web_price as-is
            "amount": f"{amt:.2f}",            # qty * rate
            "source_url": it.get("source_url", ""),
            "source_note": it.get("source_note", ""),
        })

    achieved_total = f"{sum(_d2(x['amount']) for x in out_items):.2f}"
    return {"items": out_items, "total": achieved_total}


@app.post("/api/suggest_items")
def api_suggest_items():
    body = request.get_json(force=True)
    company = body.get("company_name", "").strip()
    total = body.get("total", "").strip()
    currency = (body.get("currency") or "USD").upper()
    if not company or not total:
        abort(400, "company_name and total required")
    try:
        out = generate_items_with_grounding(company, total, currency)
        return jsonify(out)
    except Exception as e:
        abort(400, str(e))

@app.post("/api/suggest_items_text")
def api_suggest_items_text():
    body = request.get_json(force=True)
    company = body.get("company_name", "").strip()
    total = body.get("total", "").strip()
    currency = (body.get("currency") or "USD").upper()
    if not company or not total:
        abort(400, "company_name and total required")
    data = generate_items_with_grounding(company, total, currency)
    lines = [
        f"PI No: AUTO-{datetime.now():%Y%m%d}",
    ]
    for it in data["items"]:
        lines.append(f"- {it['item_no']}; {it['description']}; Qty {it['quantity']}; Rate {it['rate']}")
    lines += [
        "Bank:",
    ]
    return jsonify({"text": "\n".join(lines), **data})


# ------------------- Design engine (monochrome + variety, upgraded) -------------------
def pick(seed=None):
    """Random design used ONCE per new company. Thereafter never re-pick."""
    if seed is None:
        seed_env = os.getenv("DESIGN_SEED")
        seed = seed_env if seed_env else os.urandom(8).hex()
    h = int(hashlib.sha256(str(seed).encode()).hexdigest(), 16)
    rnd = random.Random(h)

    FONT_STACKS = [
        "system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
        "Georgia, 'Times New Roman', Times, serif",
        "'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace",
    ]
    TITLE_POS = ["top-left", "top-center", "top-right"]
    HEADER_LAYOUTS = ["single-row", "two-column", "stacked-tight"]
    META_LAYOUTS = ["2x3-grid", "3x2-grid", "single-column", "two-column"]
    BUYER_NOTIFY_LAYOUTS = ["side-by-side", "stacked", "buyer-left-notify-right"]
    TABLE_STYLES = [
        "classic grid with solid borders",
        "striped rows with light gray backgrounds",
        "minimal with only horizontal rules",
        "compact dotted borders",
    ]
    BORDER_STYLES = ["solid", "dashed", "dotted", "none"]
    SPACING_SCALES = ["compact", "standard", "airy"]
    ALIGNMENT = ["left", "center", "justify"]

    # New aesthetic toggles
    ARCHETYPES = [
        "formal-document",      # like a legal/letterhead doc
        "modern-invoice",       # clean posterish header, open whitespace
        "ledger-compact",       # information-dense, narrow leading
        "editorial",            # magazine-like hierarchy
    ]
    HEADLINE_STYLES = ["upper-wide", "smallcaps", "sentence"]
    TABLE_DENSITY = ["cozy", "standard", "roomy"]
    RULE_WEIGHT = ["hairline", "thin", "medium"]  # affects borders/separators

    return {
        "seed": str(seed),
        "archetype": rnd.choice(ARCHETYPES),
        "font_stack": rnd.choice(FONT_STACKS),
        "title_position": rnd.choice(TITLE_POS),
        "header_layout": rnd.choice(HEADER_LAYOUTS),
        "meta_layout": rnd.choice(META_LAYOUTS),
        "buyer_notify_layout": rnd.choice(BUYER_NOTIFY_LAYOUTS),
        "table_style": rnd.choice(TABLE_STYLES),
        "table_density": rnd.choice(TABLE_DENSITY),
        "rule_weight": rnd.choice(RULE_WEIGHT),
        "border_style": rnd.choice(BORDER_STYLES),
        "spacing": rnd.choice(SPACING_SCALES),
        "max_width_px": rnd.choice([720, 800, 860, 900, 960, 1000]),
        "radius_px": rnd.choice([0, 2, 4, 6, 8]),
        "font_base_px": rnd.choice([12, 13, 14, 15]),
        "align": rnd.choice(ALIGNMENT),
        "headline_style": rnd.choice(HEADLINE_STYLES),
        "uppercase_titles": rnd.choice([True, False]),
        "letter_spacing": rnd.choice([0, 0.2, 0.4, 0.6, 1.0]),
        "word_spacing": rnd.choice([0, 1, 2, 3]),
        "zebra_intensity": rnd.choice([0, 5, 8, 12]),
        "tight_numbers": rnd.choice([True, False]),
        "centered": rnd.choice([True, False]),
    }

def build_style_instructions(recipe: Dict[str, Any]) -> str:
    bw_rules = (
        "STRICTLY BLACK & WHITE:\n"
        "- Background: pure white (#fff). Text: pure black (#000).\n"
        "- Borders/rules: grayscale (#111–#999). No colors, gradients, shadows, images, CSS variables."
    )
    return f"""
MONOCHROME DESIGN SPEC (seed {recipe['seed']}):
- {bw_rules}
- Aesthetic archetype: {recipe['archetype']} (interpret and reflect this in proportions and rhythm)
- Header layout: {recipe['header_layout']} | Title position: {recipe['title_position']}
- Meta layout: {recipe['meta_layout']} | Buyer/Notify layout: {recipe['buyer_notify_layout']}
- Table style: {recipe['table_style']} | Density: {recipe['table_density']} | Rule weight: {recipe['rule_weight']}
- Borders: {recipe['border_style']}
- Font: {recipe['font_stack']} base {recipe['font_base_px']}px
- Headline style: {recipe['headline_style']} | uppercase={recipe['uppercase_titles']}
- Letter spacing {recipe['letter_spacing']}px | word spacing {recipe['word_spacing']}px
- Align {recipe['align']}
- Width {recipe['max_width_px']}px | radius {recipe['radius_px']}px
- Zebra rows: {recipe['zebra_intensity']}%
""".strip()

def inject_center_override_if_needed(html: str, force_center: bool) -> str:
    if not force_center:
        return html
    override = """
<style id="center-override">
  body, .invoice, header, main, section, table, th, td { text-align: center !important; }
</style>
""".strip()
    if "</head>" in html.lower():
        return re.sub(r"</head>", override + "\n</head>", html, count=1, flags=re.I)
    return "<!doctype html><html><head>" + override + "</head><body>" + html + "</body></html>"

def apply_design_override(html: str, recipe: Dict[str, Any]) -> str:
    """Global polish applied post-render (monochrome-friendly)."""
    base = int(recipe.get("font_base_px", 14))
    line_h = 1.45 if recipe.get("spacing") == "airy" else (1.35 if recipe.get("spacing") == "standard" else 1.25)
    pad_y = {"cozy": 6, "standard": 9, "roomy": 12}.get(recipe.get("table_density", "standard"), 9)
    border_map = {"hairline": "0.5px", "thin": "1px", "medium": "1.5px"}
    rule_px = border_map.get(recipe.get("rule_weight", "thin"), "1px")
    border_style_map = {
        "solid": f"{rule_px} solid #444",
        "dashed": f"{rule_px} dashed #444",
        "dotted": f"{rule_px} dotted #444",
        "none": "0 none transparent",
    }
    border_css = border_style_map.get(recipe.get("border_style", "solid"), f"{rule_px} solid #444")
    font_stack = recipe.get("font_stack", "system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif")
    ls = float(recipe.get("letter_spacing", 0) or 0)
    ws = float(recipe.get("word_spacing", 0) or 0)
    maxw = int(recipe.get("max_width_px", 900))
    radius = int(recipe.get("radius_px", 0))
    align = recipe.get("align", "left")
    headline_style = recipe.get("headline_style", "upper-wide")
    uppercase = bool(recipe.get("uppercase_titles", False))
    tnum = bool(recipe.get("tight_numbers", False))

    if headline_style == "upper-wide":
        head_transform = "text-transform:uppercase; letter-spacing:0.06em;"
    elif headline_style == "smallcaps":
        head_transform = "font-variant-caps: all-small-caps; letter-spacing: 0.02em;"
    else:
        head_transform = ""

    css = f"""
<style id="design-override">
  /* Global type & rhythm */
  body {{
    font-family:{font_stack} !important;
    font-size:{base}px !important;
    line-height:{line_h} !important;
    letter-spacing:{ls}px !important;
    word-spacing:{ws}px !important;
    color:#000 !important;
    text-align:{align} !important;
    -webkit-font-smoothing:antialiased; -moz-osx-font-smoothing:grayscale;
    {"font-feature-settings:'tnum' 1,'lnum' 1;" if tnum else ""}
  }}
  .invoice {{
    max-width:{maxw}px !important; margin:24px auto !important; padding:8px 16px !important;
    border-radius:{radius}px !important;
  }}

  /* Header */
  .inv-header h1, .inv-header .title {{
    font-weight:700; font-size:{int(base*1.7)}px; {head_transform}
    margin:0 0 6px 0;
  }}
  .inv-header .company, .inv-header address {{ font-size:{int(base*0.95)}px; }}

  /* Meta grid */
  .inv-meta {{
    display:grid; gap:8px 16px; margin:14px 0 10px 0; padding:10px 12px;
    border:{border_css}; border-radius:{max(0, radius-2)}px;
    grid-template-columns: repeat( auto-fit, minmax(180px, 1fr) );
  }}
  .inv-meta .label {{ font-weight:600; }}

  /* Parties */
  .inv-parties {{ display:grid; gap:12px; grid-template-columns: repeat( auto-fit, minmax(280px, 1fr) ); margin:8px 0 6px; }}
  .inv-parties .party {{
    border:{border_css}; border-radius:{max(0, radius-2)}px; padding:10px 12px;
  }}
  .inv-parties .party .heading {{ font-weight:600; margin-bottom:4px; }}

  /* Category */
  .inv-category {{ margin:12px 0 6px; font-weight:700; { "text-transform:uppercase;" if uppercase else "" } }}

  /* Table */
  table.items {{ width:100%; border-collapse:collapse; }}
  table.items thead th {{
    border-bottom:{border_css}; padding:{pad_y}px 10px; font-weight:700; text-align:left;
  }}
  table.items td {{ padding:{pad_y}px 10px; border-bottom:0.5px solid #999; }}
  table.items td.num, table.items th.num {{ text-align:right; }}
  table.items tfoot .total-row th, table.items tfoot .total-row td {{
    border-top:{border_css}; padding:{pad_y+2}px 10px; font-weight:700;
  }}

  /* Bank block */
  .inv-bank {{
    margin-top:16px; padding:10px 12px; border:{border_css}; border-radius:{max(0, radius-2)}px;
  }}
  .inv-bank .title {{ font-weight:700; margin-bottom:6px; }}
  .inv-bank dl {{ display:grid; grid-template-columns:160px 1fr; gap:6px 12px; margin:0; }}
  .inv-bank dt {{ font-weight:600; }}
  .inv-bank dd {{ margin:0; }}

  /* Print */
  @page {{ margin: 14mm 12mm; }}
  @media print {{
    body {{ margin:0 !important; }}
    .invoice {{ box-shadow:none !important; }}
    thead {{ display:table-header-group; }}
    tfoot {{ display:table-footer-group; }}
    tr, img {{ page-break-inside: avoid !important; }}
  }}
</style>
"""
    if "</head>" in html.lower():
        return re.sub(r"</head>", css + "\n</head>", html, count=1, flags=re.I)
    return "<!doctype html><html><head>" + css + "</head><body>" + html + "</body></html>"

def force_monochrome(html: str, zebra_pct: int) -> str:
    """Nuke every path to color and re-assert B&W."""
    # strip gradients/bg images/shadows/filters
    html = re.sub(r"(box|text)-shadow\s*:[^;]+;", "", html, flags=re.I)
    html = re.sub(r"filter\s*:[^;]+;", "", html, flags=re.I)
    html = re.sub(r"background(-image)?\s*:[^;]+;", "", html, flags=re.I)
    # strip CSS vars
    html = re.sub(r"--[a-zA-Z0-9_-]+\s*:\s*[^;]+;", "", html, flags=re.I)
    html = re.sub(r"var\(--[^)]+\)", "#000", html, flags=re.I)
    # normalize colors on properties
    def _to_gray(m):
        prop = m.group(1).lower()
        if prop in ("color", "fill", "stroke"):
            return f"{prop}:#000;"
        if prop.startswith("background"):
            return f"{prop}:#fff;"
        return f"{prop}:#444;"
    color_prop = r"(color|background(?:-color)?|border(?:-color)?|outline|fill|stroke)"
    color_val  = r"(#[0-9a-fA-F]{3,8}|rgba?\([^)]*\)|hsla?\([^)]*\)|oklch\([^)]*\)|oklab\([^)]*\)|hwb\([^)]*\)|[a-zA-Z]+)"
    html = re.sub(rf"{color_prop}\s*:\s*{color_val}\s*;", _to_gray, html, flags=re.I)
    # inject final B&W + zebra
    inject = f"""
<style id="bw-enforcer">
  html,body{{background:#fff!important;color:#000!important;}}
  *{{box-shadow:none!important;text-shadow:none!important;-webkit-print-color-adjust:exact;print-color-adjust:exact;}}
  a, a:visited{{color:#000!important;}}
  svg *{{fill:#000!important;stroke:#000!important;}}
  {'table tr:nth-child(even){background:rgba(0,0,0,' + str(zebra_pct/100.0) + ')!important;}' if zebra_pct>0 else ''}
</style>
""".strip()
    if "</head>" in html.lower():
        return re.sub(r"</head>", inject + "\n</head>", html, count=1, flags=re.I)
    return "<!doctype html><html><head>" + inject + "</head><body>" + html + "</body></html>"

# ------------------- Persistence helpers -------------------
def slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or "company"

def json_template_path(name: str) -> pathlib.Path:
    return DATA_DIR / f"{slugify(name)}.json"

def html_template_path(name: str) -> pathlib.Path:
    return TPL_HTML_DIR / f"{slugify(name)}.html"

def load_template_json(name: str) -> Dict[str, Any] | None:
    p = json_template_path(name)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None

def save_template_json(name: str, data: Dict[str, Any]) -> None:
    p = json_template_path(name)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def load_template_html(name: str) -> str | None:
    p = html_template_path(name)
    if p.exists():
        return p.read_text(encoding="utf-8")
    return None

def save_template_html(name: str, html: str) -> None:
    p = html_template_path(name)
    p.write_text(html, encoding="utf-8")

def list_company_slugs() -> List[str]:
    return sorted([p.stem for p in DATA_DIR.glob("*.json")])

# ------------------- Validation -------------------
REQUIRED_TOP_LEVEL = [
    "company_name","company_addr","title","pi_no","date","shipment",
    "port_loading","port_discharge","buyer_name","buyer_addr",
    "notify_name","notify_addr","category","items","total","bank"
]
REQUIRED_BANK = ["account_name","account_addr","account_no","bank_name","bank_addr","swift"]
REQUIRED_ITEM = ["slno","item_no","description","quantity","unit","rate","amount"]

def validate_invoice_data(d: Dict[str, Any]) -> None:
    missing = [k for k in REQUIRED_TOP_LEVEL if k not in d]
    if missing:
        raise ValueError(f"Missing top-level keys: {missing}")
    if not isinstance(d["items"], list) or len(d["items"]) == 0:
        raise ValueError("`items` must be a non-empty list.")
    for idx, it in enumerate(d["items"], 1):
        imiss = [k for k in REQUIRED_ITEM if k not in it]
        if imiss:
            raise ValueError(f"Item {idx} missing keys: {imiss}")
    if not isinstance(d["bank"], dict):
        raise ValueError("`bank` must be an object.")
    bmiss = [k for k in REQUIRED_BANK if k not in d["bank"]]
    if bmiss:
        raise ValueError(f"Bank missing keys: {bmiss}")

# ------------------- Template normalization (robust) -------------------
REQUIRED_PLACEHOLDERS = [
    "company_name","company_addr","title","pi_no","date","shipment",
    "port_loading","port_discharge","buyer_name","buyer_addr",
    "notify_name","notify_addr","category","total",
    "bank.account_name","bank.account_addr","bank.account_no",
    "bank.bank_name","bank_addr","bank.swift".replace("bank_addr","bank.bank_addr")  # ensure correct key
]
# Fix REQUIRED_PLACEHOLDERS properly:
REQUIRED_PLACEHOLDERS = [
    "company_name","company_addr","title","pi_no","date","shipment",
    "port_loading","port_discharge","buyer_name","buyer_addr",
    "notify_name","notify_addr","category","total",
    "bank.account_name","bank.account_addr","bank.account_no",
    "bank.bank_name","bank.bank_addr","bank.swift"
]

KNOWN_KEYS = [
    "company_name","company_addr","title","pi_no","date","shipment",
    "port_loading","port_discharge","buyer_name","buyer_addr",
    "notify_name","notify_addr","category","total",
    "bank.account_name","bank.account_addr","bank.account_no",
    "bank.bank_name","bank.bank_addr","bank.swift",
]
ITEM_KEYS = ["it.slno","it.item_no","it.description","it.quantity","it.unit","it.rate","it.amount"]

def _single_to_double_placeholders(html: str) -> str:
    keys_alt = "|".join(map(re.escape, KNOWN_KEYS + ITEM_KEYS))
    pattern = rf"(?<!\{{)\{{\s*(?:{keys_alt})\s*\}}(?!\}})"
    def repl(m):
        inner = m.group(0).strip("{} ").strip()
        return "{{ " + inner + " }}"
    return re.sub(pattern, repl, html)

def _fix_braces_and_raw(html: str) -> str:
    html = _html.unescape(html)
    html = re.sub(r"\{\%\s*raw\s*\%\}", "", html, flags=re.I)
    html = re.sub(r"\{\%\s*endraw\s*\%\}", "", html, flags=re.I)
    html = re.sub(r"\{\%\s*verbatim\s*\%\}", "", html, flags=re.I)
    html = re.sub(r"\{\%\s*endverbatim\s*\%\}", "", html, flags=re.I)
    html = re.sub(r"\{\{\{\s*(.*?)\s*\}\}\}", r"{{ \1 }}", html)
    html = re.sub(r"\[\[\s*([a-zA-Z0-9_.]+)\s*\]\]", r"{{ \1 }}", html)
    html = html.replace(r"\{\{", "{{").replace(r"\}\}", "}}")
    return html

def _ensure_items_loop(html: str) -> str:
    if re.search(r"\{\%\s*for\s+it\s+in\s+items\s*\%\}", html):
        return html
    loop_rows = """
<tbody>
{% for it in items %}
  <tr>
    <td>{{ it.slno }}</td>
    <td>{{ it.item_no }}</td>
    <td>{{ it.description }}</td>
    <td class="num">{{ it.quantity }} {{ it.unit }}</td>
    <td class="num">{{ it.rate }}</td>
    <td class="num">{{ it.amount }}</td>
  </tr>
{% endfor %}
  <tr><td colspan="6" style="height: 18px;"></td></tr>
</tbody>
""".strip()
    if re.search(r"<tbody[^>]*>.*?</tbody>", html, flags=re.I|re.S):
        return re.sub(r"<tbody[^>]*>.*?</tbody>", loop_rows, html, flags=re.I|re.S, count=1)
    items_table = f"""
<table class="items" style="width:100%; border-collapse:collapse; margin-top:16px;">
  <thead>
    <tr>
      <th>SL.NO</th><th>ITEM NO.</th><th>DESCRIPTION</th>
      <th>QUANTITY (NOS)</th><th>RATE (CIF CHENNAI)</th><th>TOTAL AMOUNT</th>
    </tr>
  </thead>
  {loop_rows}
</table>
"""
    return re.sub(r"(</body\s*>\s*</html\s*>\s*$)", items_table + r"\n\1", html, flags=re.I) or (html + items_table)

def _ensure_placeholders_present(html: str) -> str:
    for key in REQUIRED_PLACEHOLDERS:
        tag = "{{ " + key + " }}"
        if tag not in html:
            html = html.replace("</body>", f'\n<span style="display:none">{tag}</span>\n</body>')
    return html

def normalize_jinja_template(html: str) -> str:
    html = _single_to_double_placeholders(html)
    html = _fix_braces_and_raw(html)
    html = _ensure_items_loop(html)
    html = _ensure_placeholders_present(html)
    return html

# ------------------- One-time LLM → Jinja2 template -------------------
def build_template_prompt(style_instructions: str) -> str:
    return textwrap.dedent(f"""
        Return ONLY JSON (no prose). Generate ONE professional, print-friendly HTML invoice with inline CSS only,
        as a Jinja2 template with placeholders.

        Must return exactly:
        {{
          "name": "invoice_template.html",
          "html": "<!doctype html>...Jinja2 template..."
        }}

        Base semantic skeleton and class names (you MUST keep these wrappers,
        but you are free to change internal layout using flex/grid, spacing, alignment, etc.):

        <body>
          <div class="invoice">
            <header class="inv-header"> ... company/title ... </header>
            <section class="inv-meta"> ... meta grid ... </section>
            <section class="inv-parties">
              <div class="party buyer"> ... </div>
              <div class="party notify"> ... </div>
            </section>
            <h3 class="inv-category">{{{{ category }}}}</h3>
            <section class="inv-items">
              <table class="items">
                <colgroup>
                  <col style="width:7%"><col style="width:20%"><col style="width:33%">
                  <col style="width:14%"><col style="width:13%"><col style="width:13%">
                </colgroup>
                <thead>
                  <tr>
                    <th>SL.NO</th><th>ITEM NO.</th><th>DESCRIPTION</th>
                    <th>QUANTITY (NOS)</th><th>RATE (CIF CHENNAI)</th><th>TOTAL AMOUNT</th>
                  </tr>
                </thead>
                <tbody>
                  {{% for it in items %}}
                    <tr>
                      <td>{{{{ it.slno }}}}</td>
                      <td>{{{{ it.item_no }}}}</td>
                      <td>{{{{ it.description }}}}</td>
                      <td class="num">{{{{ it.quantity }}}} {{{{ it.unit }}}}</td>
                      <td class="num">{{{{ it.rate }}}}</td>
                      <td class="num">{{{{ it.amount }}}}</td>
                    </tr>
                  {{% endfor %}}
                  <tr><td colspan="6" style="height:18px;"></td></tr>
                </tbody>
                <tfoot>
                  <tr class="total-row"><th colspan="5" class="label">TOTAL</th><th class="num">{{{{ total }}}}</th></tr>
                </tfoot>
              </table>
            </section>
            <section class="inv-bank">
              <div class="title">Bank Details</div>
              <dl>
                <dt>Account Name</dt><dd>{{{{ bank.account_name }}}}</dd>
                <dt>Account Address</dt><dd>{{{{ bank.account_addr }}}}</dd>
                <dt>Account No</dt><dd>{{{{ bank.account_no }}}}</dd>
                <dt>Bank Name</dt><dd>{{{{ bank.bank_name }}}}</dd>
                <dt>Bank Branch Address</dt><dd>{{{{ bank.bank_addr }}}}</dd>
                <dt>SWIFT</dt><dd>{{{{ bank.swift }}}}</dd>
              </dl>
            </section>
          </div>
        </body>

        Placeholders (exact keys):
        - {{{{ company_name }}}}, {{{{ company_addr }}}}, {{{{ title }}}}, {{{{ pi_no }}}}, {{{{ date }}}}, {{{{ shipment }}}}
        - {{{{ port_loading }}}}, {{{{ port_discharge }}}}, {{{{ buyer_name }}}}, {{{{ buyer_addr }}}}, {{{{ notify_name }}}}, {{{{ notify_addr }}}}
        - {{{{ category }}}}, {{{{ total }}}}
        - Bank: {{{{ bank.account_name }}}}, {{{{ bank.account_addr }}}}, {{{{ bank.account_no }}}}, {{{{ bank.bank_name }}}}, {{{{ bank.bank_addr }}}}, {{{{ bank.swift }}}}

        VERY IMPORTANT – INTERPRET THE STYLE RECIPE LITERALLY:

        - If header_layout == "single-row": make .inv-header a single horizontal bar (flex row).
        - If header_layout == "two-column": split company info and invoice meta into two vertical stacks.
        - If header_layout == "stacked-tight": put title on top, company below, with tight spacing.

        - If meta_layout == "2x3-grid": lay out meta fields in 2 columns x 3 rows using CSS grid.
        - If meta_layout == "3x2-grid": 3 columns x 2 rows.
        - If meta_layout == "single-column": stack the fields one per line.
        - If meta_layout == "two-column": two columns using grid or flex-wrap.

        - If buyer_notify_layout == "side-by-side": .inv-parties is two columns.
        - If buyer_notify_layout == "stacked": .party blocks are stacked vertically full-width.
        - If buyer_notify_layout == "buyer-left-notify-right": emphasize buyer on the left (slightly wider).

        - If archetype == "formal-document": make it look like a letterhead (tighter table, minimal decoration).
        - If archetype == "modern-invoice": big bold title, more whitespace, slightly larger headings.
        - If archetype == "ledger-compact": dense table, smaller base font, minimal margins.
        - If archetype == "editorial": strong typographic hierarchy, varied font sizes, more breathing space.

        Requirements:
        - INLINE CSS ONLY; no external assets, images, logos, @import, or CSS variables.
        - STRICTLY BLACK & WHITE: only #000 text, #fff backgrounds, grayscale borders (#111–#999).
          No rgb()/hsl()/oklch()/named colors, gradients, shadows, filters, or background-image.
        - Print styles: @media print + @page margins; crisp rules; system fonts; compact CSS.
        - You MAY adjust padding, font sizes, alignments, and spacing to strongly reflect the style recipe.
        - Include the final blank row after the items loop.

        Style recipe (use this like a config object, not just decoration):
        {style_instructions}
    """).strip()


def ask_llm_for_jinja_template(style_instructions: str) -> str:
    prompt = build_template_prompt(style_instructions)

    # Use a slightly larger, more capable, but still cheap model if you have it
    # Fallback to whatever is available.
    model_name = "gpt-4o-mini"

    # More creative settings → more layout / CSS variety
    resp = client.chat.completions.create(
        model=model_name,
        temperature=0.7,       # was 0.0 → bump for variety
        top_p=0.9,
        presence_penalty=0.4,   # encourage different structures
        frequency_penalty=0.2,
        max_tokens=9000,
        stream=False,
        messages=[
            {
                "role": "system",
                "content": (
                    "Return only a single JSON object with keys 'name' and 'html'. "
                    "No explanations, no markdown, no comments."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    raw = resp.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        s, e = raw.find("{"), raw.rfind("}")
        parsed = json.loads(raw[s:e+1]) if (s != -1 and e != -1 and e > s) else {}
    html = parsed.get("html", "")
    return normalize_jinja_template(html)


# ------------------- Rendering (NO LLM) -------------------
def _manual_render_if_needed(rendered_html: str, ctx: Dict[str, Any]) -> str:
    """If Jinja left tags (rare), do a last-pass simple replacement for scalars and bank.* (also single-brace)."""
    if "{{" not in rendered_html and re.search(r"\{[^{].*?\}", rendered_html) is None:
        return rendered_html

    scalars = [
        "company_name","company_addr","title","pi_no","date","shipment",
        "port_loading","port_discharge","buyer_name","buyer_addr",
        "notify_name","notify_addr","category","total"
    ]
    for k in scalars:
        v = str(ctx.get(k, ""))
        rendered_html = re.sub(r"\{\{\s*"+re.escape(k)+r"\s*\}\}", v, rendered_html)
        rendered_html = re.sub(r"\{\s*"+re.escape(k)+r"\s*\}", v, rendered_html)

    bank = ctx.get("bank", {}) or {}
    for bk in ["account_name","account_addr","account_no","bank_name","bank_addr","swift"]:
        v = str(bank.get(bk, ""))
        rendered_html = re.sub(r"\{\{\s*bank\."+re.escape(bk)+r"\s*\}\}", v, rendered_html)
        rendered_html = re.sub(r"\{\s*bank\."+re.escape(bk)+r"\}", v, rendered_html)

    return rendered_html

def render_company_invoice_from_template(company_name: str, template_html: str, data: Dict[str, Any], design_recipe: Dict[str, Any]) -> str:
    """Render saved Jinja2 template with current data; enforce overrides and hard B&W."""
    ctx = {k: v for k, v in data.items() if k != "design_recipe"}
    template_html = normalize_jinja_template(template_html)
    html = Template(template_html).render(**ctx)
    html = _manual_render_if_needed(html, ctx)
    html = inject_center_override_if_needed(html, force_center=design_recipe.get("centered", False))
    html = apply_design_override(html, design_recipe)
    html = force_monochrome(html, zebra_pct=int(design_recipe.get("zebra_intensity", 0) or 0))

    slug = slugify(company_name)
    target_dir = INVOICE_DIR / slug
    target_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    pi_no = data.get('pi_no', 'PI')
    pi_no_safe = re.sub(r'[^\w\-_.]', '_', str(pi_no))
    filename = f"{slug}-{pi_no_safe}-{ts}.html"

    (target_dir / filename).write_text(html, encoding="utf-8")
    return filename, str(target_dir.resolve())

# ------------------- Parse (regex) + LLM extract endpoints -------------------
def parse_freeform_invoice(text: str) -> Dict[str, Any]:
    t = text.strip()
    lines = [ln.strip() for ln in t.splitlines()]
    joined = "\n".join(lines)
    out: Dict[str, Any] = {
        "company_name": "",
        "company_addr": "",
        "title": "PROFORMA INVOICE",
        "pi_no": "",
        "date": "",
        "shipment": "",
        "port_loading": "",
        "port_discharge": "",
        "buyer_name": "",
        "buyer_addr": "",
        "notify_name": "",
        "notify_addr": "",
        "category": "",
        "items": [],
        "total": "",
        "bank": { "account_name": "", "account_addr": "", "account_no": "", "bank_name": "", "bank_addr": "", "swift": "" }
    }
    def pick(label_variants, after=":"):
        pattern = r"(?im)^(?:\s*(" + "|".join(re.escape(v) for v in label_variants) + r")\s*" + re.escape(after) + r"\s*)(.+?)\s*$"
        m = re.search(pattern, joined);  return m.group(2).strip() if m else ""
    out["pi_no"] = pick(["PI No", "PI Number", "Proforma No", "Proforma Number", "PI#"])
    out["date"] = pick(["Date"])
    ship = pick(["Ship via", "Shipment", "Ship"])
    out["shipment"] = ship or ""
    out["port_loading"] = pick(["POL", "Port of Loading", "Port of Origin"])
    out["port_discharge"] = pick(["POD", "Port of Discharge", "Destination"])
    out["buyer_name"] = pick(["Buyer"])
    out["buyer_addr"] = pick(["Buyer address", "Buyer Address"])
    out["notify_name"] = pick(["Notify"])
    out["notify_addr"] = pick(["Notify address", "Notify Address"])
    out["category"] = pick(["Category"])
    # Items block
    items_block = ""
    if re.search(r"(?im)^Items\s*:\s*$", joined):
        start = re.search(r"(?im)^Items\s*:\s*$", joined).end()
        after = joined[start:]
        m_bank = re.search(r"(?im)^\s*Bank\s*:", after)
        cut = m_bank.start() if m_bank else len(after)
        items_block = after[:cut].strip()
    items = []
    if items_block:
        for ln in [l.strip("-•* \t") for l in items_block.splitlines() if l.strip()]:
            m = re.search(
                r"^\s*(?P<item_no>[^;|]+?)\s*[;|]\s*(?P<description>[^;|]+?)\s*[;|]\s*(?:Qty|QTY)\s*(?P<qty>[\d,\.]+)\s*[;|]\s*(?:Rate|RATE)\s*(?P<rate>[\d,\.]+)\s*$",
                ln, flags=re.I
            )
            if not m:
                parts = [p.strip() for p in re.split(r"[;|]", ln) if p.strip()]
                item_no = parts[0] if parts else ""
                description = parts[1] if len(parts) > 1 else ""
                qty = ""; rate = ""
                for p in parts[2:]:
                    q = re.search(r"(?i)\bqty\b\s*([\d,\.]+)", p)
                    r = re.search(r"(?i)\brate\b\s*([\d,\.]+)", p)
                    if q: qty = q.group(1)
                    if r: rate = r.group(1)
            else:
                gd = m.groupdict()
                item_no = gd["item_no"].strip()
                description = gd["description"].strip()
                qty = gd["qty"].strip()
                rate = gd["rate"].strip()
            def d2(x):
                try: return Decimal(str(x).replace(",", "")).quantize(Decimal("0.01"))
                except InvalidOperation: return Decimal("0.00")
            qd = d2(qty); rd = d2(rate); amt = (qd * rd).quantize(Decimal("0.01"))
            items.append({
                "slno": str(len(items)+1), "item_no": item_no, "description": description,
                "quantity": f"{qd.normalize():f}".rstrip("0").rstrip(".") if qd % 1 != 0 else f"{int(qd)}",
                "unit": "NOS", "rate": f"{rd:.2f}", "amount": f"{amt:.2f}",
            })
    out["items"] = items or out["items"]
    # Simple bank parsing (optional block)
    m_bankstart = re.search(r"(?im)^\s*Bank\s*:\s*$", joined)
    if m_bankstart:
        out["bank"]["account_name"] = pick(["Account Name"], after=":")
        out["bank"]["account_addr"] = pick(["Address", "Account Address"], after=":")
        out["bank"]["account_no"]   = pick(["Account No", "A/c No", "Account Number"], after=":")
        out["bank"]["bank_name"]    = pick(["Bank Name"], after=":")
        out["bank"]["bank_addr"]    = pick(["Bank Address", "Branch Address"], after=":")
        out["bank"]["swift"]        = pick(["SWIFT", "SWIFT Code", "BIC"], after=":")
    if not out["shipment"]:
        m_ship = re.search(r"(?i)\bship\s+via\s+([a-z ]+)", joined)
        if m_ship: out["shipment"] = f"VIA {m_ship.group(1).strip().upper()}"
    return out

@app.post("/api/parse_paste")
def api_parse_paste():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    if not text:
        abort(400, "text required")
    return jsonify(parse_freeform_invoice(text))

def _coerce_decimal(s: str, default="0.00") -> str:
    try:
        d = Decimal(str(s).replace(",", "")).quantize(Decimal("0.01"))
        return f"{d:.2f}"
    except Exception:
        return default

def _normalize_llm_invoice(payload: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "company_name": payload.get("company_name", ""),
        "company_addr": payload.get("company_addr", ""),
        "title": payload.get("title", "PROFORMA INVOICE") or "PROFORMA INVOICE",
        "pi_no": payload.get("pi_no", ""),
        "date": payload.get("date", ""),
        "shipment": payload.get("shipment", ""),
        "port_loading": payload.get("port_loading", ""),
        "port_discharge": payload.get("port_discharge", ""),
        "buyer_name": payload.get("buyer_name", ""),
        "buyer_addr": payload.get("buyer_addr", ""),
        "notify_name": payload.get("notify_name", ""),
        "notify_addr": payload.get("notify_addr", ""),
        "category": payload.get("category", ""),
        "items": [],
        "total": payload.get("total", ""),
        "bank": {
            "account_name": payload.get("bank", {}).get("account_name", ""),
            "account_addr": payload.get("bank", {}).get("account_addr", ""),
            "account_no":   payload.get("bank", {}).get("account_no", ""),
            "bank_name":    payload.get("bank", {}).get("bank_name", ""),
            "bank_addr":    payload.get("bank", {}).get("bank_addr", ""),
            "swift":        payload.get("bank", {}).get("swift", ""),
        }
    }
    items = payload.get("items", []) or []
    norm_items = []
    for i, it in enumerate(items, 1):
        item_no = str(it.get("item_no", "")).strip()
        desc    = str(it.get("description", "")).strip()
        qty_s   = str(it.get("quantity", "")).strip()
        unit    = str(it.get("unit", "NOS")).strip() or "NOS"
        rate_s  = str(it.get("rate", "")).strip()
        amt_s   = str(it.get("amount", "")).strip()
        rate = _coerce_decimal(rate_s, "0.00")
        try:
            qd = Decimal((qty_s or "0").replace(",", ""))
        except Exception:
            qd = Decimal("0")
        if not amt_s:
            amt_s = f"{(qd * Decimal(rate)).quantize(Decimal('0.01')):.2f}"
        norm_items.append({
            "slno": str(i), "item_no": item_no, "description": desc,
            "quantity": (str(int(qd)) if qd % 1 == 0 else str(qd).rstrip("0").rstrip(".")) if qty_s else "0",
            "unit": unit, "rate": rate, "amount": _coerce_decimal(amt_s, "0.00"),
        })
    out["items"] = norm_items
    if not out["total"]:
        try:
            tot = sum(Decimal(i["amount"]) for i in norm_items)
            out["total"] = f"{tot.quantize(Decimal('0.01')):.2f}"
        except Exception:
            out["total"] = ""
    return out

def llm_extract_invoice(text: str) -> Dict[str, Any]:
    schema_hint = {"items":[{"item_no":"string","description":"string","quantity":"number/string","unit":"NOS","rate":"number/string","amount":"optional"}]}
    system = "You extract structured fields from a pasted invoice note. Return ONLY a single JSON object. No prose."
    user = f"""
Extract these keys: company_name, company_addr, title, pi_no, date, shipment, port_loading, port_discharge,
buyer_name, buyer_addr, notify_name, notify_addr, category, items, total, bank (account_name, account_addr, account_no, bank_name, bank_addr, swift).
If a field is missing, use an empty string. Items is an array of objects with item_no, description, quantity, unit (default 'NOS'), rate, amount (optional).
Respond ONLY with JSON. Example schema: {json.dumps(schema_hint)}
TEXT:
{text}
""".strip()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=2000,
        stream=False,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        s, e = raw.find("{"), raw.rfind("}")
        data = json.loads(raw[s:e+1]) if (s!=-1 and e!=-1 and e>s) else {}
    return _normalize_llm_invoice(data)

@app.post("/api/llm_extract")
def api_llm_extract():
    body = request.get_json(force=True)
    text = (body.get("text") or "").strip()
    if not text:
        abort(400, "text required")
    try:
        parsed = llm_extract_invoice(text)
        return jsonify(parsed)
    except Exception as e:
        abort(400, f"Extraction failed: {e}")

# ------------------- Routes for UI/basic API -------------------
# ------------------- Routes for UI/basic API -------------------
@app.get("/")
def index():
    # Renders templates/index.html (place your full HTML there)
    return render_template("index.html")

@app.get("/api/companies")
def api_companies():
    slugs = list_company_slugs()  # uses the helper you defined above
    pretty = [s.replace("-", " ").title() for s in slugs]
    return jsonify({"slugs": slugs, "pretty": pretty})

@app.get("/api/template")
def api_template():
    name = request.args.get("name", "").strip()
    if not name:
        abort(400, "name required")
    t = load_template_json(name)
    if not t:
        abort(404, "template not found")
    return jsonify(t)

def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(a, dict) or not isinstance(b, dict):
        return b if b is not None else a
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

@app.post("/api/save_template")
def api_save_template():
    data = request.get_json(force=True)
    company_name = (data.get("company_name") or "").strip()
    if not company_name:
        abort(400, "company_name required")
    existing = load_template_json(company_name) or {}
    design_recipe = existing.get("design_recipe")
    merged = deep_merge(existing, data)
    if design_recipe:
        merged["design_recipe"] = design_recipe  # keep recipe immutable per company
    save_template_json(company_name, merged)
    return jsonify({"ok": True, "message": f"Template saved for '{company_name}'."})

@app.post("/api/generate")
def api_generate():
    data = request.get_json(force=True)
    company_name = (data.get("company_name") or "").strip()
    if not company_name:
        abort(400, "company_name required")

    tpl_json = load_template_json(company_name)
    tpl_html = load_template_html(company_name)

    if tpl_json is None or tpl_html is None:
        # First time for this company → pick monochrome recipe, ask LLM once, lock it
        design_recipe = pick()
        merged_json = dict(data)
        merged_json["company_name"] = company_name
        merged_json["design_recipe"] = design_recipe
        save_template_json(company_name, merged_json)

        style_instructions = build_style_instructions(design_recipe)
        jinja_html = ask_llm_for_jinja_template(style_instructions)
        jinja_html = normalize_jinja_template(jinja_html)
        save_template_html(company_name, jinja_html)

        tpl_json = merged_json
        tpl_html = jinja_html
    else:
        # Subsequent runs → keep the original recipe/template, only overlay data
        design_recipe = tpl_json.get("design_recipe") or pick()
        merged_json = deep_merge(tpl_json, data)
        merged_json["company_name"] = company_name
        merged_json["design_recipe"] = design_recipe
        save_template_json(company_name, merged_json)
        tpl_json = merged_json
        tpl_html = normalize_jinja_template(tpl_html)

    try:
        validate_invoice_data(tpl_json)
        filename, outdir = render_company_invoice_from_template(
            company_name, tpl_html, tpl_json, tpl_json["design_recipe"]
        )
    except Exception as e:
        abort(400, str(e))

    slug = slugify(company_name)
    download_url = url_for("serve_invoice", company_slug=slug, filename=filename)
    return jsonify({"ok": True, "filename": filename, "dir": outdir, "download_url": download_url})

@app.get("/invoices/<company_slug>/<path:filename>")
def serve_invoice(company_slug, filename):
    d = INVOICE_DIR / company_slug
    if not d.exists():
        abort(404)
    return send_from_directory(d, filename, as_attachment=False)

# ------------------- Main -------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port="5055")
