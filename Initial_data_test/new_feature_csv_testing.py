import pandas as pd
import os

df = pd.read_csv(r'C:\Users\acer\Desktop\Programming\ML\Hackathons\Amazon_ML_Challenge_2025\Initial_data_test\extracted_features\using_kaggle\new\new_merged_output_train_1.csv')

df = df.fillna(0)

# --- Step 4: Create output directory if not exists ---
output_dir = os.path.join(os.getcwd(), 'extracted_features')
os.makedirs(output_dir, exist_ok=True)

# --- Step 6: Save merged file ---
output_path = os.path.join(output_dir, 'new_merged_output_train_2.csv')
df.to_csv(output_path, index=False)

print(f"✅ Merged CSV saved successfully at: {output_path}")
print(df.head())