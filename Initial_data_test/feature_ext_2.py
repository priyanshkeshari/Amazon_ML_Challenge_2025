import pandas as pd
import re
import numpy as np # Import numpy for better handling of numerical missing values
import os # Import the os module for directory handling

# --- Configuration ---
OUTPUT_DIR = "extracted_features"
OUTPUT_FILENAME = "extracted_features_2.csv"
# ---------------------

# --- Sample Data Creation (for demonstration, including a potential NaN/missing value) ---
# data = {
#     'sample_id': [33127, 198967, 261251, 999999],
#     'catalog_content': [
#         "Item Name: La Victoria Green Taco Sauce Mild, 12 Ounce (Pack of 6)\nValue: 72.0\nUnit: Fl Oz\n",
#         "Item Name: Salerno Cookies, The Original Butter Cookies, 8 Ounce (Pack of 4)\nBullet Point 1: Original Butter Cookies: Classic butter cookies made with real butter\nBullet Point 2: Variety Pack: Includes 4 boxes with 32 cookies total\nBullet Point 3: Occasion Perfect: Delicious cookies for birthdays, weddings, anniversaries\nBullet Point 4: Shareable Treats: Fun to give and enjoy with friends and family\nBullet Point 5: Salerno Brand: Trusted brand of delicious butter cookies since 1925\nValue: 32.0\nUnit: Ounce\n",
#         "Item Name: Bear Creek Hearty Soup Bowl, Creamy Chicken with Rice, 1.9 Ounce (Pack of 6)\nBullet Point 1: Loaded with hearty long grain wild rice and vegetables\nBullet Point 2: Full of hearty goodness\nBullet Point 3: Single serve bowls\nBullet Point 4: Easy to prepare mix\nBullet Point 5: 0 grams trans fat\nValue: 11.4\nUnit: Ounce\n",
#         np.nan # Introducing a missing value to test the fix
#     ],
#     'image_link': [
#         "https://m.media-amazon.com/images/I/51mo8htwTHL.jpg",
#         "https://m.media-amazon.com/images/I/71YtriIHAAL.jpg",
#         "https://m.media-amazon.com/images/I/51+PFEe-w-L.jpg",
#         "https://m.media-amazon.com/images/I/example.jpg"
#     ],
#     'price': [4.89, 13.12, 1.97, 5.00]
# }
# df = pd.DataFrame(data)
df = pd.read_csv(r'C:\Users\acer\Desktop\Programming\ML\Hackathons\Amazon_ML_Challenge_2025\68e8d1d70b66d_student_resource\student_resource\dataset\train.csv')
# ---------------------------------------------


def extract_catalog_features(row):
    # 🌟 FIX: Ensure content is always a string. If it's NaN (float type in pandas), use an empty string.
    content = str(row['catalog_content']) if pd.notna(row['catalog_content']) else ""
    
    features = {}
    
    # --- 1. Item Name (Pre-processing) ---
    item_name_match = re.search(r"Item Name: (.*?)\n", content)
    # Use empty string if no match to prevent error in subsequent steps
    features['Item_Name_Full'] = item_name_match.group(1).strip() if item_name_match else ""
    
    # --- 2. Pack Size ---
    pack_match = re.search(r"\((?:P|p)ack of (\d+)\)", features['Item_Name_Full'])
    features['Pack_Size'] = int(pack_match.group(1)) if pack_match else 1
    
    # --- 3. Brand Name ---
    if features['Item_Name_Full']:
        # Assumes the first word of the item name is often the brand.
        features['Brand_Name'] = features['Item_Name_Full'].split()[0].replace(',', '')
    else:
        features['Brand_Name'] = None

    # --- 4. Net Quantity (Value) ---
    value_match = re.search(r"Value: ([\d.]+)", content)
    # Use NaN for missing numerical values
    features['Unit_Quantity'] = float(value_match.group(1)) if value_match else np.nan 
    # modified from previous
    features['Unit_Quantity'] /= features['Pack_Size']

    # --- 5. Unit Type ---
    unit_match = re.search(r"Unit: (.*?)\n", content)
    features['Unit_Type'] = unit_match.group(1).strip() if unit_match else None
    
    # --- 6. Number of Bullet Points ---
    features['Num_Bullet_Points'] = len(re.findall(r"Bullet Point \d+:", content))

    # --- 7. Full Bullet Points Text ---
    bullet_points_text = re.findall(r"Bullet Point \d+: (.*?)\n", content)
    features['Bullet_Points_Text'] = " ".join(bullet_points_text)
    
    # --- 8. Product Name/Category (Simple cleanup of Item Name) ---
    temp_name = features['Item_Name_Full']
    # Removes the brand and quantity description for a cleaner description
    temp_name = re.sub(r"^\w+\s*|,\s*\d+\s*(?:Ounce|Fl Oz).*", "", temp_name).strip() 
    features['Product_Description'] = temp_name
    
    return pd.Series(features)

# Apply the extraction function
df_features = df.apply(extract_catalog_features, axis=1)

# Merge the extracted features back into the main DataFrame
df = pd.concat([df, df_features], axis=1)

# --- 9. Unit Price (Derived Feature) ---
# Calculate the price per unit of measure (e.g., price per Fl Oz)
# Use .fillna(1) for Pack_Size in case of missing data, and handle division by zero/NaN
df['Total_Quantity'] = df['Unit_Quantity'] * df['Pack_Size']
# Use .where() to prevent division by zero, resulting in NaN where Total_Quantity is 0 or NaN
df['Unit_Price_Per_Unit_Calculated'] = df['price'] / df['Total_Quantity'].where(df['Total_Quantity'] != 0)
df['Unit_Price_Pack_Calculated'] = df['price'] / df['Pack_Size'].where(df['Pack_Size'] != 0)

# Display the final, relevant features
final_features = df[['sample_id', 'Brand_Name', 'Product_Description', 
                     'Pack_Size', 'Unit_Quantity', 'Unit_Type', 'Total_Quantity', 
                     'Unit_Price_Per_Unit_Calculated', 'Unit_Price_Pack_Calculated', 'Num_Bullet_Points', 'price']]

print("--- Extracted Features with Fix ---")
# print(final_features.to_string(index=False))

# 10. Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Created directory: '{OUTPUT_DIR}'")

# 11. Define the full path for the CSV file
output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

# 12. Save the DataFrame to CSV
final_features.to_csv(output_path, index=False) # index=False prevents writing the DataFrame index to the CSV

print(f"\n✅ Successfully saved extracted features to: '{output_path}'")