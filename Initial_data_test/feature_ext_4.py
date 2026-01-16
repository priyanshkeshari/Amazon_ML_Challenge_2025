import pandas as pd
import re
import numpy as np # Import numpy for better handling of numerical missing values
import os # Import the os module for directory handling
import sys

# --- Configuration ---
OUTPUT_DIR = "extracted_features"
OUTPUT_FILENAME = "extracted_features_4_test.csv"

# --- Keyword Dictionaries for Feature Flagging (UPDATED FOR E-COMMERCE PLATFORM) ---
# Keywords are converted to lowercase and checked against product text
KEYWORD_CONFIG = {
    # Premium / Quality / Luxury — impacts pricing perception across all categories
    'is_premium': [
        'gourmet', 'artisan', 'premium', 'select', 'reserve', 'deluxe', 'finest', 'luxury',
        'handcrafted', 'small batch', 'signature', 'exclusive', 'limited', 'high-end',
        'ultra', 'heritage', 'crafted', 'superior', 'platinum', 'elite', 'professional',
        'pro-grade', 'custom', 'bespoke', 'collectors', 'designer', 'flagship', 'prestige'
    ],
    
    # Health / Eco / Certifications / Ingredients — affects demand & sustainability score
    'is_health_eco': [
        'organic', 'sustainable', 'eco-friendly', 'non-gmo', 'recyclable', 'vegan',
        'gluten-free', 'natural', 'pure', 'certified', 'fiber', 'protein', 'biodegradable',
        'cruelty-free', 'fair trade', 'plant-based', 'paraben-free', 'bpa-free',
        'hypoallergenic', 'eco', 'renewable', 'chemical-free', 'additive-free', 'eco-safe',
        'low-fat', 'low-sugar', 'whole grain', 'healthy', 'probiotic', 'antioxidant'
    ],
    
    # Electronics / Tech Goods — used in high-price-variance categories
    'is_electronic': [
        'usb', 'bluetooth', 'wifi', 'wireless', 'battery', 'electric', 'smart', 'hd', '4k',
        '8k', 'led', 'oled', 'charger', 'adapter', 'compatible', 'app', 'processor', 'display',
        'camera', 'sensor', 'ai', 'touchscreen', 'hdmi', 'portable', 'gaming', 'console',
        'monitor', 'pc', 'laptop', 'tablet', 'smartwatch', 'wearable', 'headphone', 'earbud',
        'speaker', 'soundbar', 'microphone', 'amplifier', 'router', 'antenna', 'drone',
        'gadget', 'powerbank', 'projector', 'graphics', 'motherboard', 'ssd', 'ram'
    ],
    
    # Apparel / Accessories / Fashion
    'is_apparel': [
        'size', 'color', 'fit', 'leather', 'cotton', 'wool', 'unisex', 'shoe', 'shirt',
        'jacket', 'material', 'fabric', 'accessory', 'jewelry', 'dress', 'pant', 't-shirt',
        'hoodie', 'sneaker', 'sock', 'scarf', 'watch', 'bag', 'belt', 'cap', 'glove',
        'sandal', 'heel', 'outerwear', 'sportswear', 'denim', 'kurta', 'saree', 'ethnic',
        'fashion', 'style', 'casual', 'formal', 'fit', 'stretchable', 'trendy', 'printed'
    ],
    
    # Consumables / Groceries / Cleaning / Personal Care
    'is_consumable': [
        'food', 'drink', 'beverage', 'snack', 'mix', 'spice', 'sauce', 'oil', 'canned',
        'fresh', 'paper', 'detergent', 'perishable', 'soap', 'shampoo', 'toothpaste',
        'supplement', 'vitamin', 'nut', 'chocolate', 'biscuit', 'juice', 'powder', 'coffee',
        'tea', 'flour', 'rice', 'grain', 'sugar', 'salt', 'cleaner', 'disinfectant', 'wipe',
        'lotion', 'cream', 'gel', 'toiletry', 'toilet paper', 'dishwash', 'sanitizer'
    ],
    
    # Latest / Model / Launch — indicates newness, model upgrade, and affects base price
    'is_latest_model': [
        'new', 'latest', 'updated', 'model', '2024', '2025', 'generation', 'gen', 'series',
        'edition', 'version', 'revamp', 'refresh', 'v2', 'v3', 'mark ii', 'mark iii',
        'enhanced', 'improved', 'next-gen', 'release', 'launch', 'upgraded', 'smart-gen',
        'flagship', 'reissue', 'special edition', 'modern', 'advanced'
    ],
    
    # Home / Furniture / Kitchen / Lifestyle
    'is_home_kitchen': [
        'furniture', 'sofa', 'bed', 'mattress', 'chair', 'table', 'lamp', 'curtain', 'carpet',
        'utensil', 'cookware', 'pan', 'pot', 'bottle', 'storage', 'organizer', 'cleaning',
        'kitchen', 'home', 'decor', 'fryer', 'blender', 'mixer', 'vacuum', 'iron', 'fan',
        'air purifier', 'humidifier', 'bedding', 'pillow', 'sheet', 'towel'
    ],
    
    # Baby / Pet / Toys — specific categories on Amazon
    'is_baby_pet': [
        'diaper', 'toy', 'stroller', 'crib', 'bottle', 'formula', 'pacifier', 'baby oil',
        'pet', 'dog', 'cat', 'leash', 'collar', 'treat', 'kennel', 'litter', 'grooming'
    ],
    
    # Sports / Outdoor / Automotive
    'is_sports_auto': [
        'fitness', 'gym', 'sports', 'ball', 'racket', 'cycle', 'bike', 'helmet', 'car',
        'automotive', 'engine', 'tire', 'tool', 'equipment', 'outdoor', 'tent', 'camping',
        'hiking', 'fishing', 'golf', 'yoga', 'skate', 'board', 'bat', 'jersey'
    ]
}
# ---------------------

# --- Sample Data Creation (for demonstration, including a potential NaN/missing value) ---
# NOTE: Using a generic path to ensure script portability
try:
    df = pd.read_csv(r'C:\Users\acer\Desktop\Programming\ML\Hackathons\Amazon_ML_Challenge_2025\68e8d1d70b66d_student_resource\student_resource\dataset\test.csv')
except FileNotFoundError:
    print("Error: train.csv not found. Please place the file in the same directory.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading or processing CSV: {e}")
    sys.exit(1)
# ---------------------------------------------


def extract_catalog_features(row):
    # 🌟 FIX: Ensure content is always a string. If it's NaN (float type in pandas), use an empty string.
    content = str(row['catalog_content']) if pd.notna(row['catalog_content']) else ""
    
    features = {}
    
    # --- 1. Item Name (Pre-processing) ---
    item_name_match = re.search(r"Item Name: (.*?)\n", content)
    # Applied .lower() for normalization
    features['Item_Name_Full'] = item_name_match.group(1).strip().lower() if item_name_match else ""
    
    # --- 2. Pack Size ---
    pack_match = re.search(r"\((?:P|p)ack of (\d+)\)", features['Item_Name_Full'])
    features['Pack_Size'] = int(pack_match.group(1)) if pack_match else 1
    
    # --- 3. Brand Name ---
    if features['Item_Name_Full']:
        # This is already lowercased, as it's derived from Item_Name_Full
        features['Brand_Name'] = features['Item_Name_Full'].split()[0].replace(',', '')
    else:
        features['Brand_Name'] = None

    # --- 4. Net Quantity (Value) ---
    value_match = re.search(r"Value: ([\d.]+)", content)
    # Use NaN for missing numerical values
    features['Pack_Quantity'] = float(value_match.group(1)) if value_match else np.nan 
    # modified from previous (Pack_Quantity is quantity per unit)
    features['Pack_Quantity'] /= features['Pack_Size']

    # --- 5. Unit Type ---
    unit_match = re.search(r"Unit: (.*?)\n", content)
    # Applied .lower() for normalization
    features['Unit_Type'] = unit_match.group(1).strip().lower() if unit_match else None
    
    # --- 6. Number of Bullet Points ---
    features['Num_Bullet_Points'] = len(re.findall(r"Bullet Point \d+:", content))

    # --- 7. Full Bullet Points Text ---
    bullet_points_text = re.findall(r"Bullet Point \d+: (.*?)\n", content)
    # Applied .lower() for normalization
    features['Bullet_Points_Text'] = " ".join(bullet_points_text).lower()
    
    # --- 8. Product Name/Category (Simple cleanup of Item Name) ---
    temp_name = features['Item_Name_Full']
    # Removes the brand and quantity description for a cleaner description (operates on lowercased string)
    temp_name = re.sub(r"^\w+\s*|,\s*\d+\s*(?:Ounce|Fl Oz).*", "", temp_name).strip() 
    features['Product_Description'] = temp_name
    
    return pd.Series(features)

# Apply the extraction function
df_features = df.apply(extract_catalog_features, axis=1)

# Merge the extracted features back into the main DataFrame
df = pd.concat([df, df_features], axis=1)

# --- 9. Unit Price (Derived Feature - Existing Logic) ---
# Calculate the price per unit of measure (e.g., price per Fl Oz)
# Use .fillna(1) for Pack_Size in case of missing data, and handle division by zero/NaN
df['Total_Quantity'] = df['Pack_Quantity'] * df['Pack_Size']
# Use .where() to prevent division by zero, resulting in NaN where Total_Quantity is 0 or NaN


# was not commented on train, for test commented
# df['Unit_Price_Per_Unit_Calculated'] = df['price'] / df['Total_Quantity'].where(df['Total_Quantity'] != 0)
# df['Unit_Price_Per_Pack_Calculated'] = df['price'] / df['Pack_Size'].where(df['Pack_Size'] != 0)


# --- 10. Keyword Feature Flagging (UPDATED LOGIC) ---
# Combine relevant text columns for comprehensive keyword search
text_source = df['Product_Description'].fillna('') + ' ' + df['Bullet_Points_Text'].fillna('')
# NOTE: This line was already present, but explicit lowercasing of source features (above)
# ensures consistency even if this line was removed or changed.
text_source = text_source.str.lower()

for feature_name, keywords in KEYWORD_CONFIG.items():
    # Create a single regex pattern for all keywords in the category, using word boundaries (\b)
    pattern = r'\b(' + '|'.join(map(re.escape, keywords)) + r')\b'
    
    # Check if any keyword matches in the combined text source (returns True/False)
    # Convert boolean result to integer (1 for match, 0 for no match)
    df[feature_name] = text_source.str.contains(pattern, na=False, regex=True).astype(int)
# --- END UPDATED LOGIC ---


# Display the final, relevant features (UPDATED to include all 9 new keyword flags)
final_features = df[['sample_id', 'Brand_Name', 'Product_Description', 
                     'Pack_Size', 'Pack_Quantity', 'Unit_Type', 'Total_Quantity', 
                    #  'Unit_Price_Per_Unit_Calculated', 'Unit_Price_Per_Pack_Calculated', 
                     'Num_Bullet_Points', 
                    #  'price', 
                     'is_premium', 'is_health_eco', 'is_electronic', 'is_apparel', 'is_consumable', 'is_latest_model',
                     'is_home_kitchen', 'is_baby_pet', 'is_sports_auto']]

print("--- Extracted Features with Keyword Flags ---")
# print(final_features.to_string(index=False))

# 11. Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Created directory: '{OUTPUT_DIR}'")

# 12. Define the full path for the CSV file
output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

# 13. Save the DataFrame to CSV
final_features.to_csv(output_path, index=False) # index=False prevents writing the DataFrame index to the CSV

print(f"\n✅ Successfully saved extracted features, including keyword flags, to: '{output_path}'")
