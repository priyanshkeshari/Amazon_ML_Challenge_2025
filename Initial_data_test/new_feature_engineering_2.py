import re
import math
import pandas as pd
import spacy
from spacy.matcher import Matcher
from pint import UnitRegistry
from typing import Optional, Tuple, Dict, Any
import os
from pathlib import Path
from datetime import datetime

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
    "fl", "fl oz", "fl.", "fl. oz", "fluid_ounce", "g", "gram", "grams",
    "kg", "kilogram", "kilograms", "ml", "milliliter", "millilitre",
    "l", "liter", "litre", "pint", "quart", "gallon"
]
# matcher pattern: NUMBER + OPTIONAL 'x/×' + NUMBER + UNIT (word)  OR NUMBER + UNIT
matcher.add("QUANTITY_COMBO", [
    [{"LIKE_NUM": True}, {"LOWER": {"IN": ["x", "×"]}}, {"LIKE_NUM": True}, {"LOWER": {"IN": unit_tokens}}],
    [{"LIKE_NUM": True}, {"LOWER": {"IN": unit_tokens}}],
])

# --- Density table for common categories (g per ml) ---
DENSITY_TABLE = {
    "water": {"d": 1.000, "c": 0.95, "note": "baseline"},
    "soda": {"d": 1.000, "c": 0.90, "note": "approx water"},
    "soft drink": {"d": 1.000, "c": 0.90, "note": "approx water"},
    "milk": {"d": 1.030, "c": 0.90, "note": "whole milk typical"},
    "skim milk": {"d": 1.035, "c": 0.70, "note": "varies"},
    "heavy cream": {"d": 0.994, "c": 0.70, "note": "approx"},
    "olive oil": {"d": 0.920, "c": 0.95, "note": "food-grade olive oil"},
    "vegetable oil": {"d": 0.920, "c": 0.90, "note": "typical edible oils"},
    "honey": {"d": 1.420, "c": 0.95, "note": "very dense"},
    "maple syrup": {"d": 1.330, "c": 0.90, "note": "culinary syrup"},
    "corn syrup": {"d": 1.380, "c": 0.85, "note": "thicker syrup"},
    "generic syrup": {"d": 1.30, "c": 0.80, "note": "generic pancake/sweet syrups"},
    "sugar_granulated": {"d": 0.85, "c": 0.60, "note": "bulk packed sugar; variable"},
    "powdered_sugar": {"d": 0.50, "c": 0.50, "note": "very variable"},
    "brown_sugar_packed": {"d": 0.90, "c": 0.60, "note": "packed brown sugar"},
    "flour": {"d": 0.53, "c": 0.50, "note": "all-purpose, sifted. bulk density, variable"},
    "cocoa_powder": {"d": 0.45, "c": 0.50, "note": "loose powder, variable"},
    "peanut_butter": {"d": 1.07, "c": 0.80, "note": "smooth peanut butter"},
    "jam": {"d": 1.20, "c": 0.80, "note": "fruit preserves"},
    "mayonnaise": {"d": 0.95, "c": 0.80, "note": "emulsion"},
    "ketchup": {"d": 1.12, "c": 0.85, "note": "tomato-based"},
    "tomato_paste": {"d": 1.30, "c": 0.85, "note": "concentrated tomato"},
    "tomato_sauce": {"d": 1.05, "c": 0.80, "note": "canned sauce"},
    "canned_beans": {"d": 1.03, "c": 0.75, "note": "drained cooked beans approx"},
    "rice": {"d": 0.85, "c": 0.50, "note": "dry rice bulk; variable"},
    "lentils": {"d": 0.82, "c": 0.50, "note": "dry lentils bulk"},
    "oats": {"d": 0.45, "c": 0.50, "note": "rolled oats bulk"},
    "salt": {"d": 1.20, "c": 0.90, "note": "table salt fine grain"},
    "butter": {"d": 0.911, "c": 0.90, "note": "solid at room temp approximate"},
    "ethanol": {"d": 0.789, "c": 0.90, "note": "pure ethanol — not for beverages directly"},
    "coffee_ground": {"d": 0.32, "c": 0.50, "note": "fresh ground, loose bulk"},
    "tea_loose": {"d": 0.20, "c": 0.50, "note": "loose tea leaves"}
}

# alias map for matching common phrasings to density keys
DENSITY_KEYWORD_ALIASES = {
    'olive oil': ['olive oil', 'extra virgin olive oil', 'evoo'],
    'vegetable oil': ['vegetable oil', 'canola oil', 'sunflower oil', 'soybean oil'],
    'honey': ['honey'],
    'maple syrup': ['maple syrup'],
    'corn syrup': ['corn syrup', 'glucose syrup'],
    'milk': ['milk', 'whole milk'],
    'skim milk': ['skim milk', 'nonfat milk', 'fat free milk'],
    'ketchup': ['ketchup', 'catsup'],
    'jam': ['jam', 'jelly', 'preserve', 'preserves'],
    'peanut_butter': ['peanut butter'],
    'flour': ['flour', 'all purpose flour', 'ap flour'],
    'sugar_granulated': ['sugar', 'granulated sugar'],
    'powdered_sugar': ['powdered sugar', 'confectioners sugar', 'icing sugar'],
    'salt': ['salt', 'sea salt', 'table salt'],
    'rice': ['rice'],
    'lentils': ['lentil', 'lentils'],
    'oats': ['oat', 'oats'],
    'butter': ['butter'],
    'tomato_paste': ['tomato paste'],
    'tomato_sauce': ['tomato sauce', 'marinara', 'pasta sauce'],
    'canned_beans': ['canned beans', 'baked beans', 'canned bean', "bush's", "bushs"],
    'soda': ['sprite', 'cola', 'soda', 'soft drink', 'fountain syrup'],
    'coffee_ground': ['coffee', 'ground coffee'],
    'tea_loose': ['tea', 'loose leaf', 'teabags', 'tea bags']
    # extend aliases using your catalogue-specific terms/brands
}

# --- Helpers ---
def normalize_unit_token(unit_raw: str) -> str:
    u = (unit_raw or "").strip().lower()
    u = u.replace(".", "")
    u = re.sub(r'\bfl[\s\.]*oz\b', 'fluid_ounce', u)
    u = u.replace("floz", "fluid_ounce")
    u = u.replace("fl", "fluid_ounce")
    u = u.replace("fluid ounce", "fluid_ounce").replace("fluid_ounces", "fluid_ounce")
    u = u.replace("oz", "ounce")
    u = u.replace("ounces", "ounce")
    u = u.replace("lb", "pound").replace("lbs", "pound")
    u = u.replace("pounds", "pound")
    u = re.sub(r'\bgrams?\b', 'gram', u)
    u = u.replace("g", "gram")
    u = u.replace("kg", "kilogram").replace("kilograms", "kilogram")
    u = u.replace("ml", "milliliter").replace("millilitre", "milliliter")
    u = u.replace("l", "liter").replace("litre", "liter")
    u = u.replace("fl_ounce", "fluid_ounce")
    # keep 'count' as-is if present
    if u in ['', 'count', 'ct', 'pcs', 'piece', 'pieces', 'ea', 'each']:
        return u
    return u

def parse_pack_count_from_text(text: str) -> Optional[int]:
    text_low = (text or "").lower()
    # "pack of 6", "pack of 6:", "pack of (6)"
    m = re.search(r'pack(?:\s*of)?\s*(?:[:\(])?\s*(\d{1,4})', text_low)
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
        span_low = span.lower()
        # handle "2 x 14.1 oz"
        m = re.search(r'(\d+(?:[.,]\d+)?)\s*[x×]\s*(\d+(?:[.,]\d+)?)\s*([a-zA-Z\.\s0-9]*)', span_low)
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

def infer_density_from_text(name: str) -> Tuple[Optional[float], float, Optional[str]]:
    """
    Returns (density_g_per_ml or None, confidence 0-1, matched_key)
    Uses DENSITY_KEYWORD_ALIASES then fallback to DENSITY_TABLE keys.
    """
    if not name:
        return None, 0.0, None
    low = name.lower()

    # 1) alias matches (preferred)
    for key, aliases in DENSITY_KEYWORD_ALIASES.items():
        for a in aliases:
            if a in low:
                info = DENSITY_TABLE.get(key) or DENSITY_TABLE.get(a)
                if info:
                    return info['d'], info['c'], key

    # 2) direct key substring match
    for key, info in DENSITY_TABLE.items():
        if key in low:
            return info['d'], info['c'], key

    # 3) fallback by common words
    m = re.search(r'\b(syrup|honey|oil|juice|milk|cream|soda|cola|tea|coffee|ketchup|jam|jelly|butter)\b', low)
    if m:
        word = m.group(1)
        # map common words to sensible keys
        fallback_map = {
            'syrup': 'generic syrup',
            'honey': 'honey',
            'oil': 'vegetable oil',
            'juice': 'water',
            'milk': 'milk',
            'cream': 'heavy cream',
            'soda': 'soda',
            'cola': 'soda',
            'tea': 'tea_loose',
            'coffee': 'coffee_ground',
            'ketchup': 'ketchup',
            'jam': 'jam',
            'jelly': 'jam',
            'butter': 'butter'
        }
        fk = fallback_map.get(word)
        if fk and fk in DENSITY_TABLE:
            info = DENSITY_TABLE[fk]
            return info['d'], info['c'], fk

    return None, 0.0, None

def convert_to_standard(value: float, unit: str, product_type: str) -> Tuple[Optional[float], Optional[float], str, str]:
    """
    Return tuple: (grams_total or None, ml_total or None, used_unit_name, conversion_method)
    """
    unit_norm = normalize_unit_token(unit or "")
    conversion_method = "direct"
    grams = None
    ml = None
    try:
        # handle 'count' manually: UnitRegistry will not have 'count' as a mass/volume; treat as special
        if unit_norm in ['count', 'ct', 'pcs', 'piece', 'pieces', 'ea', 'each', '']:
            conversion_method = "count_or_unknown_unit"
            return None, None, unit_norm, conversion_method

        qty = value * ureg(unit_norm)
        # If unit is mass convertible -> grams
        try:
            if qty.check('[mass]'):
                grams = qty.to('gram').magnitude
                conversion_method = "direct_mass"
            elif qty.check('[volume]'):
                ml = qty.to('milliliter').magnitude
                conversion_method = "direct_volume"
            else:
                # fallback unit name checks
                su = str(qty.units)
                if su in ('gram', 'kilogram', 'pound', 'ounce'):
                    grams = qty.to('gram').magnitude
                    conversion_method = "direct_mass"
                elif 'milliliter' in su or 'liter' in su or 'fluid_ounce' in su:
                    ml = qty.to('milliliter').magnitude
                    conversion_method = "direct_volume"
                else:
                    conversion_method = "unknown_unit_type"
        except Exception:
            # if pint unit introspection failed, attempt manual common conversions
            su = unit_norm
            if su in ['ounce', 'pound', 'gram', 'kilogram']:
                try:
                    grams = (value * ureg(su)).to('gram').magnitude
                    conversion_method = "direct_mass_fallback"
                except Exception:
                    conversion_method = "failed_direct"
            elif su in ['milliliter', 'liter', 'fluid_ounce']:
                try:
                    ml = (value * ureg(su)).to('milliliter').magnitude
                    conversion_method = "direct_volume_fallback"
                except Exception:
                    conversion_method = "failed_direct"
            else:
                conversion_method = "failed_direct"
    except Exception:
        conversion_method = "failed_direct"
    return grams, ml, unit_norm, conversion_method

# --- Main processing function ---
def process_catalog(df: pd.DataFrame,
                    text_fields: Optional[list]=None,
                    density_conversion_threshold: float = 0.6) -> pd.DataFrame:
    """
    df must have at least columns: 'Item Name', 'Value', 'Unit'
    text_fields: list of other columns to search for quantity info (e.g., 'Bullet Point 1', 'Product Description')
    density_conversion_threshold: only auto-convert mass<->volume if density confidence >= threshold
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
        density_used = None
        density_confidence = None
        density_key = None

        # if Value present numeric and Unit present (non-empty)
        if pd.notna(value_field) and (unit_field and unit_field.lower() not in ['none', 'nan', '']):
            try:
                v = float(value_field)
            except Exception:
                v = None
            if v is not None:
                grams, ml, used_unit, conv = convert_to_standard(v, unit_field, product_type="unknown")
                grams_total = grams
                ml_total = ml
                conversion_method = "from_value_field:" + conv
                confidence = 1.0 if conv.startswith("direct") else 0.7
                # if pack mention exists and the unit in the name has qty that is per-item, detect pattern "2 x 14.1 oz" to override
                extracted = extract_quantities_spacy(combined_text)
                if extracted:
                    # prefer explicit textual phrase representing full pack or pack*per-item
                    extr = max(extracted, key=lambda x: x[1])
                    extracted_val, extracted_unit = extr[1], extr[2]
                    g_extr, ml_extr, _, _ = convert_to_standard(extracted_val, extracted_unit, "unknown")
                    if g_extr or ml_extr:
                        grams_total = g_extr
                        ml_total = ml_extr
                        conversion_method = "text_extracted_override"
                        confidence = 0.95
                        notes.append(f"override using text-extracted '{extr[0]}'")
                notes.append(f"pack_mentioned={pack_from_text}" if pack_from_text and pack_from_text>1 else "no_pack")
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

        # 5) Cross-conversion: if we have ml but not grams or vice versa & product hints a density, try to infer
        if (grams_total is None and ml_total is not None):
            d, d_conf, d_key = infer_density_from_text(item_name)
            if d and d_conf >= density_conversion_threshold:
                grams_total = ml_total * d
                conversion_method = (conversion_method or "") + "|inferred_density"
                confidence = min(confidence + d_conf * 0.1, 0.95)
                density_used = d
                density_confidence = d_conf
                density_key = d_key
                notes.append(f"inferred_density={d} (key={d_key})")
            else:
                # mark that we found something but confidence too low - do not convert
                if d:
                    notes.append(f"density_low_confidence={d_conf} for key={d_key}; no auto-convert")
        if (ml_total is None and grams_total is not None):
            d, d_conf, d_key = infer_density_from_text(item_name)
            if d and d_conf >= density_conversion_threshold:
                ml_total = grams_total / d
                conversion_method = (conversion_method or "") + "|inferred_density"
                confidence = min(confidence + d_conf * 0.1, 0.95)
                density_used = d
                density_confidence = d_conf
                density_key = d_key
                notes.append(f"inferred_density={d} (key={d_key})")
            else:
                if d:
                    notes.append(f"density_low_confidence={d_conf} for key={d_key}; no auto-convert")

        # 6) Pack expansion heuristics: if text contains "2 x 14.1 oz" we already used that; if pack_from_text>1 and we only
        # have per-item grams, multiply to get total; but only do this if convert_method suggests from_value_field and we believe value is per-item.
        if pack_from_text and pack_from_text > 1:
            if re.search(r'\d+\s*[x×]\s*\d', combined_text.lower()):
                # likely handled already by extractor (we set override earlier)
                pass
            else:
                # If the Value field likely represents per-item and pack is present in name,
                # multiply grams_total (or ml_total) by pack count IF conversion_method indicates value was from value_field
                if conversion_method and "from_value_field" in conversion_method and pack_from_text > 1:
                    if grams_total:
                        grams_total = grams_total * pack_from_text
                        notes.append(f"multiplied by pack_count={pack_from_text}")
                        conversion_method = (conversion_method or "") + "|pack_expanded"
                        confidence = min(confidence, 0.9)
                    if ml_total:
                        ml_total = ml_total * pack_from_text
                        notes.append(f"multiplied by pack_count={pack_from_text}")
                        conversion_method = (conversion_method or "") + "|pack_expanded"
                        confidence = min(confidence, 0.9)

        # 7) compute per_item metrics if pack present and totals available
        per_item_g = None
        per_item_ml = None
        if pack_from_text and pack_from_text > 1:
            if grams_total:
                per_item_g = grams_total / pack_from_text
            if ml_total:
                per_item_ml = ml_total / pack_from_text

        # product type (solid/liquid/count/unknown) - simple heuristics
        ptype = 'unknown'
        low_text = (item_name + " " + combined_text).lower()
        if any(k in low_text for k in ['syrup', 'sauce', 'marinade', 'juice', 'oil', 'tea', 'drink', 'beverage', 'soda', 'milk']):
            ptype = 'liquid'
        elif any(k in low_text for k in ['powder', 'beans', 'granola', 'chocolate', 'candy', 'lentil', 'almond', 'salt', 'flour', 'bar', 'snack', 'rice', 'oats']):
            ptype = 'solid'
        elif (unit_field or "").lower() in ['count', 'ct', 'pcs', 'piece', 'pieces', 'ea', 'each']:
            ptype = 'count'

        out_rows.append({
            'idx': i,
            'item_name': item_name,
            'value_field': value_field,
            'unit_field': unit_field,
            'pack_from_text': pack_from_text,
            'grams_total': None if grams_total is None else float(grams_total),
            'ml_total': None if ml_total is None else float(ml_total),
            'per_item_g': None if per_item_g is None else float(per_item_g),
            'per_item_ml': None if per_item_ml is None else float(per_item_ml),
            'conversion_method': conversion_method,
            'confidence': float(confidence),
            'density_used': density_used,
            'density_confidence': density_confidence,
            'density_key': density_key,
            'product_type': ptype,
            'notes': "; ".join(notes)
        })

    return pd.DataFrame(out_rows)


def save_cleaned_catalog(df: pd.DataFrame,
                         out_dir: str = "extracted_features",
                         filename: Optional[str] = None) -> str:
    """
    Save cleaned DataFrame to CSV inside out_dir (creates dir if needed).
    Returns the full path to the saved file.
    """
    # ensure directory exists
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # build filename if not provided
    if not filename:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cleaned_catalog_{ts}_1.csv"

    out_path = Path(out_dir) / filename

    # save CSV (no index)
    df.to_csv(out_path, index=False)

    print(f"Saved cleaned catalog to: {out_path}")
    return str(out_path)


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
         "Value": 3.0, "Unit": "Count"},
        {"Item Name": "BulkSupplements Trehalose Powder 5kg (11 lbs) (Pack of 5)",
         "Value": 176.37, "Unit": "Ounce"},
        {"Item Name": "Colour Mill Oil-Based Food Coloring, 100 Milliliters (Navy)",
         "Value": 3.38, "Unit": "Fl Oz"},
        {"Item Name": "Tetley - Original Tea Bags 240 - 750g",
         "Value": 26.25, "Unit": "Ounce"},
    ]
    df_sample = pd.DataFrame(sample_rows)
    cleaned = process_catalog(df_sample, text_fields=None)
    pd.set_option('display.max_columns', None)
    print(cleaned.to_string(index=False))

    saved_path = save_cleaned_catalog(cleaned)
    print("saved csv file ")
