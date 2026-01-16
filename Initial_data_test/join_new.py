import pandas as pd
import os

# --- Step 1: Load your CSV files ---
# Replace these paths with your actual CSV filenames
file1 = r'C:\Users\acer\Desktop\Programming\ML\Hackathons\Amazon_ML_Challenge_2025\Initial_data_test\extracted_features\using_kaggle\new\cleaned_catalog__test_20251013_081926_1.csv'
file2 = r'C:\Users\acer\Desktop\Programming\ML\Hackathons\Amazon_ML_Challenge_2025\Initial_data_test\extracted_features\processed_data_joined_2_test.csv'

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# --- Step 2: Select the columns you want to keep ---
# Keep 'sample_id' (the join key) plus the columns you need
# Example: we keep 'price' and 'Brand_Name' from df1, and 'predicted_price' from df2

df1_drop_cols = ['item_name', 'catalog_content', 'image_link', 'price_per_100g', 'price_per_100ml', 'notes']
df2_selected_cols = ['sample_id', 'Brand_Name', 'Pack_Quantity', 'max_num_in_name', 'min_num_in_name', 'avg_num_in_name', 'num_count_in_name']

df1_selected = df1.drop(columns=df1_drop_cols)
df2_selected = df2[df2_selected_cols]



# --- Step 3: Merge (join) both CSVs on 'sample_id' ---
# 'inner' keeps only matching rows, 'left' keeps all from df1, 'outer' keeps all from both
merged_df = pd.merge(df1_selected, df2_selected, on='sample_id', how='inner')

merged_df = merged_df.fillna(0)

# --- Step 4: Create output directory if not exists ---
output_dir = os.path.join(os.getcwd(), 'extracted_features')
os.makedirs(output_dir, exist_ok=True)

# --- Step 6: Save merged file ---
output_path = os.path.join(output_dir, 'new_merged_output_train_2.csv')
merged_df.to_csv(output_path, index=False)

print(f"✅ Merged CSV saved successfully at: {output_path}")
print(merged_df.head())
