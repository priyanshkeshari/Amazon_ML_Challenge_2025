import pandas as pd
import os

# Read the three CSV files
df1 = pd.read_csv(r'C:\Users\acer\Desktop\Programming\ML\Hackathons\Amazon_ML_Challenge_2025\Submission\5th\h_tuned_test_out_new_feat_cnn.csv')
df2 = pd.read_csv(r'C:\Users\acer\Desktop\Programming\ML\Hackathons\Amazon_ML_Challenge_2025\Submission\4th\test_out_new_feat_cnn.csv')
df3 = pd.read_csv(r'C:\Users\acer\Desktop\Programming\ML\Hackathons\Amazon_ML_Challenge_2025\Submission\3rd\test_out_new_feat.csv')

# Merge all three on 'sample_id'
merged = df1.merge(df2, on='sample_id', suffixes=('_1', '_2'))
merged = merged.merge(df3, on='sample_id')
merged.rename(columns={'price': 'price_3'}, inplace=True)

# Calculate average price
merged['average_price'] = merged[['price_1', 'price_2', 'price_3']].mean(axis=1)

# Keep only the desired columns
final_df = merged[['sample_id', 'average_price']]

# === Step 5: Create directory (if not exists) ===
output_dir = 'extracted_features'
os.makedirs(output_dir, exist_ok=True)

# === Step 6: Save the new CSV ===
output_path = os.path.join(output_dir, 'averaged_prices.csv')
final_df.to_csv(output_path, index=False)

print(f"✅ New CSV file saved to: {output_path}")
