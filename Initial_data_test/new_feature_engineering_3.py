# catalog_cleaner_from_catalog_content.py
import re
import math
import pandas as pd
import spacy
from spacy.matcher import Matcher
from pint import UnitRegistry
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
ureg = UnitRegistry()
# add a simple 'count' unit if not already present (dimensionless bookkeeping)
try:
    ureg.define('count = []')
except Exception:
    pass

# spaCy model (make sure you ran: python -m spacy download en_core_web_sm)
nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

# Units tokens for matcher patterns
unit_tokens = [
    "oz", "ounce", "ounces", "lb", "lbs", "pound", "pounds",
    "fl", "fl oz", "fl.", "fl. oz", "fluid_ounce", "g", "gram", "grams",
    "kg", "kilogram", "kilograms", "ml", "milliliter", "millilitre",
    "l", "liter", "litre", "pint", "quart", "gallon"
]
# Add matcher patterns
matcher.add("QUANTITY_COMBO", [
    [{"LIKE_NUM": True}, {"LOWER": {"IN": ["x", "×"]}}, {"LIKE_NUM": True}, {"LOWER": {"IN": unit_tokens}}],
    [{"LIKE_NUM": True}, {"LOWER": {"IN": unit_tokens}}],
])

# ---------------------------------------------------------------------------
# Density table & aliases (conservative starter table)
# ---------------------------------------------------------------------------
DENSITY_TABLE = {
    "water": {"d": 1.000, "c": 0.95},
    "soda": {"d": 1.000, "c": 0.90},
    "milk": {"d": 1.030, "c": 0.90},
    "skim milk": {"d": 1.035, "c": 0.70},
    "heavy cream": {"d": 0.994, "c": 0.70},
    "olive oil": {"d": 0.920, "c": 0.95},
    "vegetable oil": {"d": 0.920, "c": 0.90},
    "honey": {"d": 1.420, "c": 0.95},
    "maple syrup": {"d": 1.330, "c": 0.90},
    "corn syrup": {"d": 1.380, "c": 0.85},
    "generic syrup": {"d": 1.30, "c": 0.80},
    "sugar_granulated": {"d": 0.85, "c": 0.60},
    "powdered_sugar": {"d": 0.50, "c": 0.50},
    "flour": {"d": 0.53, "c": 0.50},
    "cocoa_powder": {"d": 0.45, "c": 0.50},
    "peanut_butter": {"d": 1.07, "c": 0.80},
    "jam": {"d": 1.20, "c": 0.80},
    "ketchup": {"d": 1.12, "c": 0.85},
    "tomato_paste": {"d": 1.30, "c": 0.85},
    "tomato_sauce": {"d": 1.05, "c": 0.80},
    "canned_beans": {"d": 1.03, "c": 0.75},
    "rice": {"d": 0.85, "c": 0.50},
    "lentils": {"d": 0.82, "c": 0.50},
    "oats": {"d": 0.45, "c": 0.50},
    "salt": {"d": 1.20, "c": 0.90},
    "butter": {"d": 0.911, "c": 0.90},
    "ethanol": {"d": 0.789, "c": 0.90},
    "coffee_ground": {"d": 0.32, "c": 0.50},
    "tea_loose": {"d": 0.20, "c": 0.50}
}

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
}

# ---------------------------------------------------------------------------
# Parsing catalog_content (extract Item Name, Value, Unit, bullets, description)
# ---------------------------------------------------------------------------
def parse_catalog_content(text: Optional[str]) -> Dict[str, Any]:
    """
    Parse a single 'catalog_content' text blob.
    Returns dict: item_name, value (float or None), unit (str or ''), bullets (joined), product_description (str)
    """
    txt = (text or "")
    # Normalize line endings
    txt = txt.replace("\r\n", "\n").replace("\r", "\n")
    out = {
        "item_name": "",
        "value": None,
        "unit": "",
        "bullets": "",
        "product_description": ""
    }

    # Item Name:
    m_name = re.search(r'Item Name:\s*(.+?)(?:\n|$)', txt, flags=re.I)
    if m_name:
        out["item_name"] = m_name.group(1).strip()
    else:
        # fallback: try first line as name
        first_line = txt.split("\n", 1)[0].strip()
        out["item_name"] = first_line

    # Value:
    m_val = re.search(r'Value:\s*([0-9]+(?:[.,][0-9]+)?|nan)', txt, flags=re.I)
    if m_val:
        val_raw = m_val.group(1).strip().lower()
        if val_raw == 'nan':
            out["value"] = None
        else:
            out["value"] = float(val_raw.replace(",", "."))
    else:
        # try to infer from unit-style phrases like "50.7oz" inside text (rare)
        m_val2 = re.search(r'(\d+(?:[.,]\d+)?)\s*(?:oz|ounce|g|gram|ml|fl oz)', txt, flags=re.I)
        if m_val2:
            out["value"] = float(m_val2.group(1).replace(",", "."))

    # Unit:
    m_unit = re.search(r'Unit:\s*([A-Za-z0-9\.\s/%\-]+)', txt, flags=re.I)
    if m_unit:
        out["unit"] = m_unit.group(1).strip()
    else:
        # try unit inline with item name e.g. "1.9 Ounce"
        m_unit2 = re.search(r'(\d+(?:[.,]\d+)?)\s*(oz|ounce|ounces|fl oz|fl\. oz|g|gram|grams|kg|kilogram|ml|l|pound|lb)\b', txt, flags=re.I)
        if m_unit2:
            out["unit"] = m_unit2.group(2).strip()

    # Bullet points: capture all "Bullet Point" lines
    bullets = re.findall(r'Bullet Point\s*\d*\s*:\s*(.+)', txt, flags=re.I)
    out["bullets"] = " ".join([b.strip() for b in bullets])

    # Product Description (capture text after 'Product Description:' up to 'Value:' or end)
    m_desc = re.search(r'Product Description:\s*(.+?)(?:\nValue:|\Z)', txt, flags=re.I | re.S)
    if m_desc:
        out["product_description"] = m_desc.group(1).strip()
    else:
        # fallback: try after 'Product Description' without colon
        m_desc2 = re.search(r'Product Description\s*(.+?)(?:\nValue:|\Z)', txt, flags=re.I | re.S)
        if m_desc2:
            out["product_description"] = m_desc2.group(1).strip()

    return out

# ---------------------------------------------------------------------------
# Quantity extraction helpers (same as earlier but updated to accept None safely)
# ---------------------------------------------------------------------------
def normalize_unit_token(unit_raw: Optional[str]) -> str:
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
    if u in ['', 'count', 'ct', 'pcs', 'piece', 'pieces', 'ea', 'each']:
        return u
    return u

def parse_pack_count_from_text(text: Optional[str]) -> Optional[int]:
    text_low = (text or "").lower()
    m = re.search(r'pack(?:\s*of)?\s*(?:[:\(])?\s*(\d{1,4})', text_low)
    if m:
        return int(m.group(1))
    m2 = re.search(r'(\d+)\s*[x×]\s*\d+\.?\d*\s*(oz|ounce|g|gram|ml|fl|fl oz|pound|lb)', text_low)
    if m2:
        return int(m2.group(1))
    m3 = re.search(r'(\d{1,4})\s*(?:pack|ct|count|pcs|pieces|pk)\b', text_low)
    if m3:
        return int(m3.group(1))
    return None

def extract_quantities_spacy(text: Optional[str]) -> list:
    doc = nlp(text or "")
    matches = matcher(doc)
    out = []
    for _, start, end in matches:
        span = doc[start:end].text
        span_low = span.lower()
        m = re.search(r'(\d+(?:[.,]\d+)?)\s*[x×]\s*(\d+(?:[.,]\d+)?)\s*([a-zA-Z\.\s0-9]*)', span_low)
        if m:
            pack = float(m.group(1).replace(",", "."))
            per = float(m.group(2).replace(",", "."))
            unit_raw = (m.group(3) or "").strip()
            out.append((span, pack * per, unit_raw, start, end))
            continue
        m2 = re.search(r'(\d+(?:[.,]\d+)?)\s*([a-zA-Z\.\s0-9]+)', span_low)
        if m2:
            val = float(m2.group(1).replace(",", "."))
            unit_raw = (m2.group(2) or "").strip()
            out.append((span, val, unit_raw, start, end))
    return out

def infer_density_from_text(name: Optional[str]) -> Tuple[Optional[float], float, Optional[str]]:
    if not name:
        return None, 0.0, None
    low = name.lower()
    for key, aliases in DENSITY_KEYWORD_ALIASES.items():
        for a in aliases:
            if a in low:
                info = DENSITY_TABLE.get(key) or DENSITY_TABLE.get(a)
                if info:
                    return info['d'], info['c'], key
    for key, info in DENSITY_TABLE.items():
        if key in low:
            return info['d'], info['c'], key
    m = re.search(r'\b(syrup|honey|oil|juice|milk|cream|soda|cola|tea|coffee|ketchup|jam|jelly|butter)\b', low)
    if m:
        word = m.group(1)
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

def convert_to_standard(value: float, unit: Optional[str], product_type: Optional[str]) -> Tuple[Optional[float], Optional[float], str, str]:
    unit_norm = normalize_unit_token(unit or "")
    conversion_method = "direct"
    grams = None
    ml = None
    try:
        if unit_norm in ['count', 'ct', 'pcs', 'piece', 'pieces', 'ea', 'each', '']:
            conversion_method = "count_or_unknown_unit"
            return None, None, unit_norm, conversion_method
        qty = value * ureg(unit_norm)
        try:
            if qty.check('[mass]'):
                grams = qty.to('gram').magnitude
                conversion_method = "direct_mass"
            elif qty.check('[volume]'):
                ml = qty.to('milliliter').magnitude
                conversion_method = "direct_volume"
            else:
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

# ---------------------------------------------------------------------------
# Main processing: adapt for df with catalog_content, sample_id, image_link, price
# ---------------------------------------------------------------------------
def process_catalog_from_catalog_content(df: pd.DataFrame,
                                         density_conversion_threshold: float = 0.6) -> pd.DataFrame:
    """
    Input df columns: must contain 'sample_id', 'catalog_content', 'image_link', 'price' (price optional)
    Returns DataFrame of extracted features similar to previous outputs.
    """

    print("process_catalog_from_catalog_content function running")

    out_rows = []
    for i, row in df.iterrows():
        sample_id = row.get('sample_id', None)
        catalog_content = row.get('catalog_content', None)
        image_link = row.get('image_link', None)
        price_val = row.get('price', None)

        parsed = parse_catalog_content(catalog_content)
        item_name = parsed.get('item_name', '')
        value_field = parsed.get('value', None)
        unit_field = parsed.get('unit', '')
        bullets = parsed.get('bullets', '')
        prod_desc = parsed.get('product_description', '')

        combined_text = " ".join([item_name, bullets, prod_desc])

        # pack count
        pack_from_text = parse_pack_count_from_text(combined_text) or 1

        # initialize
        grams_total = None
        ml_total = None
        conversion_method = None
        confidence = 0.0
        notes = []
        density_used = None
        density_confidence = None
        density_key = None

        # prefer explicit parsed value+unit
        if value_field is not None and (unit_field and unit_field.lower() not in ['none', 'nan', '']):
            v = value_field
            grams, ml, used_unit, conv = convert_to_standard(v, unit_field, product_type=None)
            grams_total = grams
            ml_total = ml
            conversion_method = "from_parsed_value:" + conv
            confidence = 1.0 if conv.startswith("direct") else 0.7
            # try override using explicit textual "2 x 14.1 oz" in item_name or bullets
            extracted = extract_quantities_spacy(combined_text)
            if extracted:
                extr = max(extracted, key=lambda x: x[1])
                extracted_val, extracted_unit = extr[1], extr[2]
                g_extr, ml_extr, _, _ = convert_to_standard(extracted_val, extracted_unit, None)
                if g_extr or ml_extr:
                    grams_total = g_extr
                    ml_total = ml_extr
                    conversion_method = "text_extracted_override"
                    confidence = 0.95
                    notes.append(f"override using text-extracted '{extr[0]}'")
            notes.append(f"pack_mentioned={pack_from_text}" if pack_from_text and pack_from_text>1 else "no_pack")
        else:
            # fallback: try extract from text
            extracted = extract_quantities_spacy(combined_text)
            if extracted:
                extr = max(extracted, key=lambda x: x[1])
                raw_span, numeric_val, unit_raw, _, _ = extr
                grams, ml, used_unit, conv = convert_to_standard(numeric_val, unit_raw, None)
                grams_total = grams
                ml_total = ml
                conversion_method = "text_extracted:" + conv
                confidence = 0.85
                notes.append(f"extracted '{raw_span}'")
            else:
                # handle count-like items if price or words indicate count
                if re.search(r'\b(count|ct|pcs|pieces|pack of|pack)\b', combined_text.lower()) or (unit_field or "").lower() in ['count', 'ct', 'pcs', 'piece', 'pieces', 'ea', 'each']:
                    conversion_method = "count"
                    confidence = 0.9
                    notes.append("treated as count")
                else:
                    conversion_method = "missing"
                    confidence = 0.0
                    notes.append("no quantity found")

        # Cross-conversion using density if needed and confident
        if (grams_total is None and ml_total is not None):
            d, d_conf, d_key = infer_density_from_text(item_name + " " + combined_text)
            if d and d_conf >= density_conversion_threshold:
                grams_total = ml_total * d
                conversion_method = (conversion_method or "") + "|inferred_density"
                confidence = min(confidence + d_conf * 0.1, 0.95)
                density_used = d
                density_confidence = d_conf
                density_key = d_key
                notes.append(f"inferred_density={d} (key={d_key})")
            else:
                if d:
                    notes.append(f"density_low_confidence={d_conf} for key={d_key}; no auto-convert")
        if (ml_total is None and grams_total is not None):
            d, d_conf, d_key = infer_density_from_text(item_name + " " + combined_text)
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

        # Pack handling: multiply per-item by pack count if appropriate
        if pack_from_text and pack_from_text > 1:
            if not re.search(r'\d+\s*[x×]\s*\d', combined_text.lower()):
                if conversion_method and "from_parsed_value" in conversion_method:
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

        # Per-item metrics
        per_item_g = None
        per_item_ml = None
        if pack_from_text and pack_from_text > 1:
            if grams_total:
                per_item_g = grams_total / pack_from_text
            if ml_total:
                per_item_ml = ml_total / pack_from_text

        # product type heuristic
        ptype = 'unknown'
        low_text = (item_name + " " + combined_text).lower()
        if any(k in low_text for k in ['syrup', 'sauce', 'marinade', 'juice', 'oil', 'tea', 'drink', 'beverage', 'soda', 'milk']):
            ptype = 'liquid'
        elif any(k in low_text for k in ['powder', 'beans', 'granola', 'chocolate', 'candy', 'lentil', 'almond', 'salt', 'flour', 'bar', 'snack', 'rice', 'oats']):
            ptype = 'solid'
        elif (unit_field or "").lower() in ['count', 'ct', 'pcs', 'piece', 'pieces', 'ea', 'each']:
            ptype = 'count'

        # price-normalized features
        price_per_100g = None
        price_per_100ml = None
        try:
            if pd.notna(price_val) and price_val is not None:
                price_num = float(price_val)
                if grams_total and grams_total > 0:
                    price_per_100g = price_num / (grams_total / 100.0)
                if ml_total and ml_total > 0:
                    price_per_100ml = price_num / (ml_total / 100.0)
        except Exception:
            price_per_100g = None
            price_per_100ml = None

        out_rows.append({
            'sample_id': sample_id,
            'item_name': item_name,
            'catalog_content': catalog_content,
            'image_link': image_link,
            'price': price_val,
            'value_field_parsed': value_field,
            'unit_field_parsed': unit_field,
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
            'price_per_100g': price_per_100g,
            'price_per_100ml': price_per_100ml,
            'notes': "; ".join(notes)
        })

    return pd.DataFrame(out_rows)

# ---------------------------------------------------------------------------
# Save cleaned catalog CSV
# ---------------------------------------------------------------------------
def save_cleaned_catalog(df: pd.DataFrame,
                         out_dir: str = "extracted_features",
                         filename: Optional[str] = None) -> str:
    
    print("saving")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if filename is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cleaned_catalog_{ts}_1.csv"
    out_path = Path(out_dir) / filename
    df.to_csv(out_path, index=False)
    print(f"Saved cleaned catalog to: {out_path}")
    return str(out_path)

# ---------------------------------------------------------------------------
# Example usage: run on sample rows or read your real CSV
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("starting")
    # Example: small in-memory sample (replace with reading your CSV)
    # sample_rows = [
    #     {"sample_id": "s1",
    #      "catalog_content": "Item Name: Bear Creek Hearty Soup Bowl, Creamy Chicken with Rice, 1.9 Ounce (Pack of 6)\nBullet Point 1: Loaded with hearty long grain wild rice and vegetables\nValue: 11.4\nUnit: Ounce",
    #      "image_link": "", "price": 3.99},
    #     {"sample_id": "s2",
    #      "catalog_content": "Item Name: Sprite Bag-In-Box Fountain Syrup 5 gal. A1\nProduct Description: ...\nValue: 640.0\nUnit: Fl Oz",
    #      "image_link": "", "price": 20.0},
    # ]
    # df_sample = pd.DataFrame(sample_rows)

    # If you have a CSV file: uncomment and use this
    df_input = pd.read_csv(r"C:\Users\acer\Desktop\Programming\ML\Hackathons\Amazon_ML_Challenge_2025\68e8d1d70b66d_student_resource\student_resource\dataset\train.csv", encoding="utf-8", low_memory=False)

    # Choose the input DataFrame to process:
    # df_input = df_sample  # replace with df_input from CSV if available

    cleaned = process_catalog_from_catalog_content(df_input)
    pd.set_option('display.max_columns', None)
    print(cleaned.to_string(index=False))

    saved_path = save_cleaned_catalog(cleaned)
    print("Saved to:", saved_path)
