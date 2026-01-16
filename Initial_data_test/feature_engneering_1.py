import pandas as pd
from typing import Literal, Optional
import os


df_left = pd.read_csv(r'C:\Users\acer\Desktop\Programming\ML\Hackathons\Amazon_ML_Challenge_2025\Initial_data_test\extracted_features\extracted_features_4_test.csv')
df_right = pd.read_csv(r'C:\Users\acer\Desktop\Programming\ML\Hackathons\Amazon_ML_Challenge_2025\Initial_data_test\extracted_features\test_features_competitive.csv')

columns_to_drop_left_extracted_features_4 = ['Product_Description', 'Unit_Type', 'Num_Bullet_Points',]
columns_to_drop_right_train_features_competitive = ['brand_name', 'pack_size', 'total_quantity', 'is_premium', 
                                                    # 'unit_price_calculated', 
                                                    'quantity_value', 'price']

df_left_dropped = df_left.drop(columns=columns_to_drop_left_extracted_features_4, axis=1)
df_right_dropped = df_right.drop(columns=columns_to_drop_right_train_features_competitive, axis=1)

df_joined = pd.merge(df_left_dropped, df_right_dropped, on='sample_id', how='inner')



# 2. Define the directory and file path
directory_name = 'extracted_features'
file_name = 'processed_data_joined_2_test.csv'
full_path = os.path.join(directory_name, file_name)

# 3. Create the directory if it doesn't exist
# The os.makedirs() function with exist_ok=True prevents an error
# if the directory already exists.
try:
    os.makedirs(directory_name, exist_ok=True)
    print(f"Directory '{directory_name}' ensured to exist.")
except Exception as e:
    print(f"Error creating directory: {e}")
    # You might want to exit or handle the error gracefully here

# 4. Save the DataFrame to the CSV file within the directory
# index=False prevents pandas from writing the DataFrame index as a column in the CSV.
try:
    df_joined.to_csv(full_path, index=False)
    print(f"\nSuccessfully saved DataFrame to: {full_path}")
except Exception as e:
    print(f"Error saving file: {e}")
