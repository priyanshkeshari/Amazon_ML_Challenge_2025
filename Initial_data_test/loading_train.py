import pandas as pd
import numpy as np

# Define the file path for the training dataset
TRAIN_FILE_PATH = r'C:\Users\acer\Desktop\Programming\ML\Hackathons\Amazon_ML_Challenge_2025\68e8d1d70b66d_student_resource\student_resource\dataset\train.csv'

try:
    # 1. Load the training dataset
    df_train = pd.read_csv(TRAIN_FILE_PATH)

    # 2. Check for missing values (NaN, None) in each column
    missing_counts = df_train.isnull().sum()

    print("--- Missing Value Counts per Column (dataset/train.csv) ---")
    print(missing_counts)
    print("\n-------------------------------------------------------------")

    # Optional: Calculate and display the percentage of missing values
    total_samples = len(df_train)
    missing_percentages = (missing_counts / total_samples) * 100

    # Filter to show only columns with at least one missing value
    missing_info = pd.DataFrame({
        'Missing Count': missing_counts,
        'Missing Percentage': missing_percentages.round(2)
    })
    missing_info = missing_info[missing_info['Missing Count'] > 0]

    if not missing_info.empty:
        print("--- Columns with Missing Data and their Percentages ---")
        print(missing_info)
    else:
        print("No missing values detected in the training dataset.")

except FileNotFoundError:
    print(f"Error: File not found at {TRAIN_FILE_PATH}. Please ensure the dataset is downloaded and the path is correct.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")