import pandas as pd
import numpy as np

# --- ACTION REQUIRED: PLEASE VERIFY THIS PATH ---
# Using the path structure implied by your file system setup.
# If your project structure is different, update this variable accordingly.
TEST_FILE_PATH = r'C:\Users\acer\Desktop\Programming\ML\Hackathons\Amazon_ML_Challenge_2025\68e8d1d70b66d_student_resource\student_resource\dataset\test.csv'

try:
    # 1. Load the test dataset
    df_test = pd.read_csv(TEST_FILE_PATH)

    # 2. Check for missing values (NaN, None) in each column
    missing_counts = df_test.isnull().sum()

    print(f"--- Missing Value Counts per Column ({TEST_FILE_PATH}) ---")
    print(missing_counts)
    print("\n-------------------------------------------------------------")

    # Calculate and display the percentage of missing values
    total_samples = len(df_test)
    missing_percentages = (missing_counts / total_samples) * 100

    # Filter information to display
    missing_info = pd.DataFrame({
        'Missing Count': missing_counts,
        'Missing Percentage': missing_percentages.round(2)
    })

    # Note on the 'price' column in the test set
    if 'price' in missing_info.index and missing_info.loc['price', 'Missing Count'] == total_samples:
        print("Note: The 'price' column is expected to be entirely missing (NaN) as it is the target variable you are required to predict on the test set.")

    # Filter to show only missing input features (excluding the price target)
    missing_input_features = missing_info[(missing_info['Missing Count'] > 0) & (missing_info.index!= 'price')]

    if not missing_input_features.empty:
        print("\n--- Missing Data Detected in Input Features ---")
        print("These features (catalog_content or image_link) will require specific handling, such as imputation or using placeholder vectors, before multimodal feature extraction.")
        print(missing_input_features)
    else:
        print("\nNo missing values detected in the input features (sample_id, catalog_content, image_link) of the test dataset.")

except FileNotFoundError:
    print(f"Error: File not found at the configured path: {TEST_FILE_PATH}. Please double-check your local file structure and update the TEST_FILE_PATH variable.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")