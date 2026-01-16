import re
import math
import pandas as pd
import spacy
from spacy.matcher import Matcher
from pint import UnitRegistry
from typing import Optional, Tuple, Dict, Any

# --- Setup ---
ureg = UnitRegistry()
# add 'count' as dimensionless quantity for bookkeeping
try:
    ureg.define('count = []')
except Exception:
    # if already defined, ignore
    pass

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

# Patterns to capture: "2 x 14.1 oz", "14.1 oz", "5 lb", "30 Ounce (Pack of 6)"
# Use token patterns for spaCy matcher (numbers + unit token)
unit_tokens = [
    "oz", "ounce", "ounces", "lb", "lbs", "pound", "pounds",
    "fl", "fl oz", "fl. oz", "fluid_ounce", "g", "gram", "grams",
    "kg", "kilogram", "kilograms", "ml", "milliliter", "millilitre",
    "l", "liter", "litre", "pint", "quart", "gallon"
]
# matcher pattern: NUMBER + OPTIONAL 'x/×' + NUMBER + UNIT (word)
matcher.add("QUANTITY_COMBO", [
    [{"LIKE_NUM": True}, {"LOWER": {"IN": ["x", "×"]}}, {"LIKE_NUM": True}, {"LOWER": {"IN": unit_tokens}}],
    [{"LIKE_NUM": True}, {"LOWER": {"IN": unit_tokens}}],
])

# --- Density table for common categories (g per ml) ---
DENSITY_BY_KEYWORD = {
    'water': 1.0,
    'soda': 1.0,
    'milk': 1.03,
    'olive oil': 0.92,
    'oil': 0.92,
    'honey': 1.42,
    'syrup': 1.30,  # generic syrup
    'flour': 0.53,  # approximate bulk density (g/ml) — careful, depends on packing
    # add more domain-specific entries as you validate them
}

# --- Helpers ---
def normalize_unit_token(unit_raw: str) -> str:
    u = unit_raw.strip().lower()
    u = u.replace(".", "")
    u = u.replace("fl oz", "fluid_ounce").replace("floz", "fluid_ounce").replace("fl", "fluid_ounce")
    u = u.replace("oz", "ounce")
    u = u.replace("ounce", "ounce")
    u = u.replace("ounces", "ounce")
    u = u.replace("lb", "pound").replace("lbs", "pound")
    u = u.replace("g", "gram").replace("gram", "gram").replace("grams", "gram")
    u = u.replace("kg", "kilogram").replace("kilograms", "kilogram")
    u = u.replace("ml", "milliliter").replace("milliliter", "milliliter").replace("millilitre", "milliliter")
    u = u.replace("l", "liter").replace("litre", "liter")
    u = u.replace("fl_ounce", "fluid_ounce")
    return u

def parse_pack_count_from_text(text: str) -> Optional[int]:
    text_low = (text or "").lower()
    # "pack of 6", "pack of 6:", "pack of (6)"
    m = re.search(r'pack(?:\sof)?\s*(?:of)?\s*(?:[:\(])?\s*(\d{1,4})', text_low)
    if m:
        return int(m.group(1))
    # "2 x 14.1 oz" style -> pack implied by first number if pattern exists
    m2 = re.search(r'(\d+)\s*[x×]\s*\d+\.?\d*\s*(oz|ounce|g|gram|ml|fl|fl oz|pound|lb)', text_low)
    if m2:
        return int(m2.group(1))
    # "6 pack", "6 ct", "6 count"
    m3 = re.search(r'(\d{1,4})\s*(?:pack|ct|count|pcs|pieces|pk)\b', text_low)
    if m3:
        return int(m3.group(1))
    return None

def extract_quantities_spacy(text: str) -> list:
    """
    Returns list of tuples (raw_text, qty_value(float), unit_str, start, end)
    """
    doc = nlp(text or "")
    matches = matcher(doc)
    out = []
    for _, start, end in matches:
        span = doc[start:end].text
        # normalize "2 x 14.1 oz" -> handle both cases
        # capture numbers + unit via regex in the matched span
        span_low = span.lower()
        # handle "2 x 14.1 oz"
        m = re.search(r'(\d+(?:[.,]\d+)?)\s*[x×]\s*(\d+(?:[.,]\d+)?)\s*(\w[\w\.\s]*\w)?', span_low)
        if m:
            pack = float(m.group(1).replace(",", "."))
            per = float(m.group(2).replace(",", "."))
            unit_raw = (m.group(3) or "").strip()
            out.append((span, pack * per, unit_raw, start, end))
            continue
        # handle single "14.1 oz"
        m2 = re.search(r'(\d+(?:[.,]\d+)?)\s*([a-zA-Z\.\s0-9]+)', span_low)
        if m2:
            val = float(m2.group(1).replace(",", "."))
            unit_raw = (m2.group(2) or "").strip()
            out.append((span, val, unit_raw, start, end))
    return out

def infer_density_from_name(name: str) -> Optional[float]:
    low = (name or "").lower()
    for k, v in DENSITY_BY_KEYWORD.items():
        if k in low:
            return v
    return None

def convert_to_standard(value: float, unit: str, product_type: str) -> Tuple[Optional[float], Optional[float], str, str]:
    """
    Return tuple: (grams_total or None, ml_total or None, used_unit_name, conversion_method)
    """
    unit_norm = normalize_unit_token(unit or "")
    conversion_method = "direct"
    grams = None
    ml = None
    try:
        qty = value * ureg(unit_norm)
        # If unit is mass convertible -> grams
        if qty.check('[mass]') or str(qty.units) in ('gram', 'kilogram', 'pound', 'ounce'):
            grams = qty.to('gram').magnitude
            conversion_method = "direct"
        # If unit is volume convertible -> milliliter
        elif qty.check('[volume]') or 'milliliter' in str(qty.units) or 'liter' in str(qty.units) or 'fluid_ounce' in str(qty.units):
            ml = qty.to('milliliter').magnitude
            conversion_method = "direct"
        # fallback for 'count' or unknown -> keep as count
        else:
            # try to interpret 'count' or numeric-only
            conversion_method = "none"
    except Exception:
        conversion_method = "failed_direct"
    return grams, ml, unit_norm, conversion_method

# --- Main processing function ---
def process_catalog(df: pd.DataFrame,
                    text_fields: Optional[list]=None) -> pd.DataFrame:
    """
    df must have at least columns: 'Item Name', 'Value', 'Unit'
    text_fields: list of other columns to search for quantity info (e.g., Bullet Point 1, Product Description)
    """
    if text_fields is None:
        text_fields = []

    out_rows = []
    for i, row in df.iterrows():
        item_name = str(row.get('Item Name', '') or '')
        value_field = row.get('Value', None)
        unit_field = (row.get('Unit', '') or '').strip()
        combined_text = " ".join([item_name] + [str(row.get(f,"") or "") for f in text_fields])

        # 1) try to parse pack count from item's text (pack, pack of, 2 x 14.1 oz etc)
        pack_from_text = parse_pack_count_from_text(combined_text) or 1

        # 2) Prefer explicit Value+Unit when it's meaningful (not nan, not 'None', not 'Count' unless product is count)
        grams_total = None
        ml_total = None
        conversion_method = None
        confidence = 0.0
        notes = []

        # if Value present numeric and Unit present (non-empty)
        if pd.notna(value_field) and (unit_field and unit_field.lower() not in ['none', 'nan', '']):
            try:
                v = float(value_field)
            except Exception:
                v = None
            if v is not None:
                grams, ml, used_unit, conv = convert_to_standard(v, unit_field, product_type="unknown")
                # adjust for pack-if the Value is per-item but pack mentions multiplicity,
                # We don't try to guess — instead we check common cases:
                # If unit is 'ounce' and item_name contains 'pack of N' but Value equals 30 and "Pack of 6: 30 oz bottles" often is total 180
                # Heuristic: if pack_from_text>1 and value_field * pack_from_text equals some other provided fields -> we'll multiply
                grams_total = grams
                ml_total = ml
                conversion_method = "from_value_field:" + conv
                confidence = 1.0
                # if pack mention exists and the unit in the name has qty that is per-item, detect pattern "2 x 14.1 oz" to override
                extracted = extract_quantities_spacy(combined_text)
                if extracted:
                    # if extracted found an explicit numeric pack * per-unit numeric, prefer it
                    # pick highest-magnitude extracted quantity in text
                    extr = max(extracted, key=lambda x: x[1])
                    extracted_val, extracted_unit = extr[1], extr[2]
                    g_extr, ml_extr, _, _ = convert_to_standard(extracted_val, extracted_unit, "unknown")
                    if g_extr or ml_extr:
                        grams_total = g_extr
                        ml_total = ml_extr
                        conversion_method = "text_extracted_override"
                        confidence = 0.95
                        notes.append(f"override using text-extracted '{extr[0]}'")
                # If Value is small but item_name says "Pack of 6: 30 oz bottles" (i.e. Unit in name different),
                # parse "Pack of" style and if name contains a unit phrase assign pack multiplication
                # If Value seems like per-item and pack_from_text>1 and Value * pack_from_text > value (i.e. likely total),
                if pack_from_text and pack_from_text > 1:
                    # If the item name explicitly contains a unit phrase with a per-item value (like '30 oz bottles'),
                    # prefer the parsed text
                    # Otherwise, if Value is smaller than typical per-pack sizes, leave as is but tag.
                    notes.append(f"pack_mentioned={pack_from_text}")
        else:
            # 3) No reliable Value field — try to extract from text
            extracted = extract_quantities_spacy(combined_text)
            if extracted:
                # choose the largest numeric mention (heuristic)
                extr = max(extracted, key=lambda x: x[1])
                raw_span, numeric_val, unit_raw, _, _ = extr
                grams, ml, used_unit, conv = convert_to_standard(numeric_val, unit_raw, "unknown")
                grams_total = grams
                ml_total = ml
                conversion_method = "text_extracted:" + conv
                confidence = 0.85
                notes.append(f"extracted '{raw_span}'")
            else:
                # 4) nothing found — fallback: if Unit column says 'Count' or value present -> treat as count
                if (unit_field or "").lower() in ['count', 'ct', 'pcs', 'piece', 'pieces', 'ea', 'each']:
                    try:
                        cnt = float(value_field) if pd.notna(value_field) else 1.0
                    except Exception:
                        cnt = 1.0
                    grams_total = None
                    ml_total = None
                    conversion_method = "count"
                    confidence = 0.9
                    notes.append("treated as count")
                else:
                    conversion_method = "missing"
                    confidence = 0.0
                    notes.append("no quantity found")

        # 5) Cross-conversion: if we have ml but want grams or vice versa & product hints a density, try to infer
        density_used = None
        if (grams_total is None and ml_total is not None):
            density = infer_density_from_name(item_name)
            if density:
                grams_total = ml_total * density
                conversion_method += "|inferred_density"
                confidence = min(confidence + 0.1, 0.95)
                density_used = density
                notes.append(f"inferred_density={density}")
        if (ml_total is None and grams_total is not None):
            density = infer_density_from_name(item_name)
            if density:
                ml_total = grams_total / density
                conversion_method += "|inferred_density"
                confidence = min(confidence + 0.1, 0.95)
                density_used = density
                notes.append(f"inferred_density={density}")

        # 6) Pack expansion heuristics: if text contains "2 x 14.1 oz" we already used that; if pack_from_text>1 and we only
        # have per-item grams, multiply to get total; but only do this if convert_method suggests text-extracted per-item or user indicates per-item.
        # We'll multiply if extracted pattern present or unit phrase contains 'x' pattern.
        if pack_from_text and pack_from_text > 1:
            # detect if name explicitly contains an "Nx <unit>" style; if yes and conversion was from text, then grams_total likely already full-pack.
            if re.search(r'\d+\s*[x×]\s*\d', combined_text.lower()):
                # likely handled already by extractor (we set override earlier)
                pass
            else:
                # If the Value field likely represents per-item and pack is present in name,
                # multiply grams_total (or ml_total) by pack count IF conversion_method contains 'from_value_field' and we believe value is per-item.
                if conversion_method and "from_value_field" in conversion_method and pack_from_text > 1:
                    # Heuristic: if item_name includes words 'pack of' and Value is relatively small compared to pack sizes, multiply
                    # We'll multiply but flag.
                    if grams_total:
                        grams_total = grams_total * pack_from_text
                        notes.append(f"multiplied by pack_count={pack_from_text}")
                        conversion_method += "|pack_expanded"
                        confidence = min(confidence, 0.9)
                    if ml_total:
                        ml_total = ml_total * pack_from_text
                        notes.append(f"multiplied by pack_count={pack_from_text}")
                        conversion_method += "|pack_expanded"
                        confidence = min(confidence, 0.9)

        # 7) compute per_item metrics if pack present and totals available
        per_item_g = None
        per_item_ml = None
        if pack_from_text and pack_from_text > 1:
            if grams_total:
                per_item_g = grams_total / pack_from_text
            if ml_total:
                per_item_ml = ml_total / pack_from_text

        # product type (solid/liquid/count/unknown)
        ptype = 'unknown'
        low_text = (item_name + " " + combined_text).lower()
        if any(k in low_text for k in ['syrup', 'sauce', 'marinade', 'juice', 'oil', 'tea', 'drink', 'beverage', 'soda']):
            ptype = 'liquid'
        elif any(k in low_text for k in ['powder', 'beans', 'granola', 'chocolate', 'candy', 'lentil', 'almond', 'salt', 'flour', 'bar', 'snack']):
            ptype = 'solid'
        elif (unit_field or "").lower() in ['count', 'ct', 'pcs', 'piece', 'pieces', 'ea', 'each']:
            ptype = 'count'

        out_rows.append({
            'idx': i,
            'item_name': item_name,
            'value_field': value_field,
            'unit_field': unit_field,
            'pack_from_text': pack_from_text,
            'grams_total': grams_total,
            'ml_total': ml_total,
            'per_item_g': per_item_g,
            'per_item_ml': per_item_ml,
            'conversion_method': conversion_method,
            'confidence': confidence,
            'density_used': density_used,
            'product_type': ptype,
            'notes': "; ".join(notes)
        })

    return pd.DataFrame(out_rows)

# -------------------------
# Example usage:
if __name__ == "__main__":
    # create a sample DataFrame with columns 'Item Name', 'Value', 'Unit', plus optional bullets
    sample_rows = [
        {"Item Name": "Bear Creek Hearty Soup Bowl, Creamy Chicken with Rice, 1.9 Ounce (Pack of 6)",
         "Value": 11.4, "Unit": "Ounce"},
        {"Item Name": "Sprite Bag-In-Box Fountain Syrup 5 gal. A1",
         "Value": 640.0, "Unit": "Fl Oz"},
        {"Item Name": "Dark Chocolate Nonpareils - 5 LB Resealable Stand Up Candy Bag",
         "Value": 80.0, "Unit": "Ounce"},
        {"Item Name": "Walden Farms Calorie Free Syrup Caramel -- 12 fl oz - 3PC",
         "Value": 3.0, "Unit": "Count"}
    ]
    df_sample = pd.DataFrame(sample_rows)
    cleaned = process_catalog(df_sample)
    pd.set_option('display.max_columns', None)
    print(cleaned.to_string(index=False))
