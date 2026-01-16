# # import pandas as pd
# # import re
# # import numpy as np
# # from sklearn.preprocessing import OneHotEncoder, StandardScaler
# # from transformers import BertTokenizer, BertModel
# # import torch

# # # --- 1. SETUP AND MODEL INITIALIZATION ---

# # # 1.1. BERT Model Selection (Best Model under Constraints)
# # # BERT-Base Uncased is selected for its performance, size (340M params), and Apache 2.0 license.[2]
# # MODEL_NAME = 'bert-base-uncased'
# # MAX_LEN = 128
# # BERT_OUTPUT_DIM = 768 # Standard output dimension for BERT Base

# # # Initialize Tokenizer and Encoder
# # tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
# # # Note: For efficiency in a code competition, you would load the model
# # # and freeze its weights (top-tuning) if computational resources are limited.[4]
# # try:
# #     bert_encoder = BertModel.from_pretrained(MODEL_NAME)
# #     # Freeze weights for feature extraction (Transfer Learning approach)
# #     for param in bert_encoder.parameters():
# #         param.requires_grad = False
# #     bert_encoder.eval() 
# # except Exception as e:
# #     print(f"BERT model initialization failed (Likely no internet or resource issue): {e}")
# #     # Placeholder for running offline
# #     bert_encoder = None


# # # --- 2. HIGH-PRECISION STRUCTURED ATTRIBUTE EXTRACTION ---
# # # This simulates the Hybrid Information Extraction (IE) component (NER + Rules).[5, 6]

# # def extract_structured_attributes(content):
# #     """
# #     Extracts high-precision, price-driving features: Brand and Item Pack Quantity (IPQ).
# #     """
# #     # 2.1. Brand Extraction (Simulating NER using a simple dictionary/lookup)
# #     # In a full solution, this would use a dedicated fine-tuned NER model.[7]
# #     known_brands =
# #     brand = 'Other'
# #     for b in known_brands:
# #         if content.startswith(b):
# #             brand = b
# #             break
            
# #     # 2.2. IPQ Extraction (Rule-Based Parsing)
# #     # Uses Regular Expressions for high precision on common product quantity patterns.[8]
# #     pack_match = re.search(r'Pack of (\d+)', content, re.I)
# #     pack_count = int(pack_match.group(1)) if pack_match else 1

# #     unit_match = re.search(r'(\d+(?:\.\d+)?)\s*(ounce|oz|lbs|lb|count)', content, re.I)
# #     unit_size = float(unit_match.group(1)) if unit_match else 0.0

# #     # 2.3. Residual Text Cleaning
# #     # Remove the extracted entities (Brand, IPQ terms) to isolate true descriptive text
# #     cleaned_text = content
# #     for b in known_brands:
# #         cleaned_text = cleaned_text.replace(b, ' ', 1)
    
# #     # Remove all quantity/unit mentions and redundant symbols
# #     cleaned_text = re.sub(r'\(\s*Pack of \d+\s*\)', ' ', cleaned_text, flags=re.I)
# #     cleaned_text = re.sub(r'\d+(?:\.\d+)?\s*(ounce|oz|lbs|lb|count)', ' ', cleaned_text, flags=re.I)
# #     cleaned_text = re.sub(r'[^\w\s]', ' ', cleaned_text) # Remove punctuation
# #     cleaned_text = ' '.join(cleaned_text.split()).lower() # Normalize spaces and lowercase

# #     return brand, pack_count, unit_size, cleaned_text

# # # --- 3. DEEP TEXT EMBEDDING PIPELINE (BERT) ---

# # def generate_bert_embeddings(texts):
# #     """
# #     Tokenizes text and generates contextual embeddings using the BERT Encoder.
# #     """
# #     if not bert_encoder:
# #         print("Using zero vector placeholder for BERT embeddings.")
# #         return np.zeros((len(texts), BERT_OUTPUT_DIM))

# #     # Tokenization: handles padding and attention mask creation
# #     inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LEN)
    
# #     with torch.no_grad():
# #         # Pass tokens through the frozen BERT encoder
# #         outputs = bert_encoder(**inputs)
    
# #     # Use the output of the token (first token) as the sentence embedding
# #     # This vector represents the entire input text context.[9]
# #     cls_embeddings = outputs.last_hidden_state[:, 0, :]
    
# #     return cls_embeddings.numpy()

# # # --- 4. FEATURE EXTRACTION EXECUTION ---

# # # Load Data (Replace with your actual data loading)
# # # Example data simulation based on the provided image
# # data = {'catalog_content':}
# # df = pd.DataFrame(data)

# # # 4.1. Structured and Residual Text Extraction
# # df[['brand', 'ipq_pack_count', 'ipq_unit_size', 'residual_text']] = df['catalog_content'].apply(
# #     lambda x: pd.Series(extract_structured_attributes(x))
# # )

# # # 4.2. Deep Text Feature Generation
# # bert_embeddings_array = generate_bert_embeddings(df['residual_text'].tolist())
# # bert_df = pd.DataFrame(bert_embeddings_array, 
# #                        columns=)


# # # --- 5. ENCODING AND SCALING FINAL TEXT FEATURES ---

# # # 5.1. One-Hot Encode (OHE) Brand (Categorical Feature)
# # ohe_brand = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
# # brand_encoded = ohe_brand.fit_transform(df[['brand']])
# # brand_df = pd.DataFrame(brand_encoded, 
# #                         columns=ohe_brand.get_feature_names_out(['brand']))

# # # 5.2. Scale Numerical Features (IPQ)
# # scaler_ipq = StandardScaler()
# # ipq_scaled = scaler_ipq.fit_transform(df[['ipq_pack_count', 'ipq_unit_size']])
# # ipq_df = pd.DataFrame(ipq_scaled, 
# #                       columns=['ipq_pack_count_scaled', 'ipq_unit_size_scaled'])

# # # 5.3. Final Text Feature Fusion (Structured + Deep)
# # X_text_fused = pd.concat(, axis=1)

# # print("\n--- Summary of Text Feature Extraction ---")
# # print(f"Total features extracted: {X_text_fused.shape[1]}")
# # print("Example of Extracted Features (First 2 Samples):")
# # print(X_text_fused.iloc[0:2, 0:10]) # Displaying first 10 columns for brevity
# # print("\n--- Example of Residual Text (Input to BERT) ---")
# # print(df[['catalog_content', 'residual_text']].head())










# import pandas as pd
# import re
# import numpy as np
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from transformers import BertTokenizer, BertModel
# import torch
# from io import StringIO
# import os # <-- Added for file management

# # --- 1. CONFIGURATION AND DIRECTORY SETUP ---
# OUTPUT_DIR = 'extracted_features'
# FUSED_FILE_NAME = 'fused_text_features.csv'
# FUSED_FILE_PATH = os.path.join(OUTPUT_DIR, FUSED_FILE_NAME)

# # Create the output directory if it does not exist
# os.makedirs(OUTPUT_DIR, exist_ok=True)
# print(f"Output directory '{OUTPUT_DIR}' ensured.")

# # --- 2. BERT MODEL INITIALIZATION ---

# # 2.1. BERT Model Selection (Best Model under Constraints: Apache 2.0, 340M params)
# MODEL_NAME = 'bert-base-uncased'
# MAX_LEN = 128
# BERT_OUTPUT_DIM = 768

# # Initialize Tokenizer and Encoder
# tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# try:
#     bert_encoder = BertModel.from_pretrained(MODEL_NAME)
#     # Freeze weights for feature extraction (Transfer Learning approach)
#     for param in bert_encoder.parameters():
#         param.requires_grad = False
#     bert_encoder.eval() 
# except Exception as e:
#     print(f"BERT model initialization failed (Likely no internet or resource issue): {e}")
#     bert_encoder = None


# # --- 3. HIGH-PRECISION STRUCTURED ATTRIBUTE EXTRACTION ---

# def extract_structured_attributes(content):
#     """
#     Extracts high-precision, price-driving features: Brand and Item Pack Quantity (IPQ).
#     """
    
#     #!!! ACTION REQUIRED: REPLACE THIS PLACEHOLDER LIST!!!
#     # Use the full, refined list generated from running the heuristic on your entire train.csv.
#     known_brands =
#     brand = 'Other'
    
#     # 3.1. Brand Extraction (Requires using the heuristic function on the Item Name field)
#     # NOTE: The provided startswith check is simplistic; for actual results, replace with your robust brand extraction logic.
#     for b in known_brands:
#         if content.startswith(b):
#             brand = b
#             break
            
#     # 3.2. IPQ Extraction (Rule-Based Parsing)
#     # Uses Regular Expressions for high precision on common product quantity patterns.
#     pack_match = re.search(r'Pack of (\d+)', content, re.I)
#     pack_count = int(pack_match.group(1)) if pack_match else 1

#     unit_match = re.search(r'(\d+(?:\.\d+)?)\s*(ounce|oz|lbs|lb|count)', content, re.I)
#     unit_size = float(unit_match.group(1)) if unit_match else 0.0

#     # 3.3. Residual Text Cleaning
#     # Remove the extracted entities (Brand, IPQ terms) to isolate true descriptive text for BERT
#     cleaned_text = content
#     for b in known_brands:
#         cleaned_text = cleaned_text.replace(b, ' ', 1)
    
#     cleaned_text = re.sub(r'\(\s*Pack of \d+\s*\)', ' ', cleaned_text, flags=re.I)
#     cleaned_text = re.sub(r'\d+(?:\.\d+)?\s*(ounce|oz|lbs|lb|count)', ' ', cleaned_text, flags=re.I)
#     cleaned_text = re.sub(r'[^\w\s]', ' ', cleaned_text) # Remove punctuation
#     cleaned_text = ' '.join(cleaned_text.split()).lower() # Normalize spaces and lowercase

#     return brand, pack_count, unit_size, cleaned_text

# # --- 4. DEEP TEXT EMBEDDING PIPELINE (BERT) ---

# def generate_bert_embeddings(texts):
#     """
#     Tokenizes text and generates contextual embeddings using the BERT Encoder.
#     """
#     if not bert_encoder:
#         # Fallback if the model failed to load
#         print("Using zero vector placeholder for BERT embeddings.")
#         return np.zeros((len(texts), BERT_OUTPUT_DIM))

#     inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LEN)
    
#     with torch.no_grad():
#         outputs = bert_encoder(**inputs)
    
#     # Use the token output as the sentence embedding
#     cls_embeddings = outputs.last_hidden_state[:, 0, :]
    
#     return cls_embeddings.numpy()

# # --- 5. FEATURE EXTRACTION EXECUTION ---

# # Load Data (Replace with your actual data loading command: pd.read_csv('.../train.csv'))
# data = {'sample_id': ,
#         'catalog_content':}
# df = pd.DataFrame(data)

# # 5.1. Structured and Residual Text Extraction
# df[['brand', 'ipq_pack_count', 'ipq_unit_size', 'residual_text']] = df['catalog_content'].apply(
#     lambda x: pd.Series(extract_structured_attributes(x))
# )

# # 5.2. Deep Text Feature Generation
# bert_embeddings_array = generate_bert_embeddings(df['residual_text'].tolist())
# bert_df = pd.DataFrame(bert_embeddings_array, index=df.index, 
#                        columns=)


# # --- 6. ENCODING AND SCALING FINAL TEXT FEATURES ---

# # Preserve sample_id for merging
# df_ids = df[['sample_id']].copy()

# # 6.1. One-Hot Encode (OHE) Brand (Categorical Feature)
# ohe_brand = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
# brand_encoded = ohe_brand.fit_transform(df[['brand']])
# brand_df = pd.DataFrame(brand_encoded, index=df.index, 
#                         columns=ohe_brand.get_feature_names_out(['brand']))

# # 6.2. Scale Numerical Features (IPQ)
# scaler_ipq = StandardScaler()
# ipq_scaled = scaler_ipq.fit_transform(df[['ipq_pack_count', 'ipq_unit_size']])
# ipq_df = pd.DataFrame(ipq_scaled, index=df.index,
#                       columns=['ipq_pack_count_scaled', 'ipq_unit_size_scaled'])

# # 6.3. Final Text Feature Fusion (Structured + Deep) - X_text_fused is the final output
# X_text_fused = pd.concat(, axis=1)

# print("\n--- Summary of Text Feature Extraction ---")
# print(f"Total features extracted (excluding sample_id): {X_text_fused.shape[1] - 1}")
# print(f"Shape of Fused Feature Matrix: {X_text_fused.shape}")

# # --- 7. SAVE RESULTS TO SEPARATE FILE ---

# try:
#     X_text_fused.to_csv(FUSED_FILE_PATH, index=False)
#     print(f"\n--- SUCCESS: Fused Text Feature Matrix Saved ---")
#     print(f"File created successfully at: {FUSED_FILE_PATH}")
#     print(f"Columns saved: sample_id, {X_text_fused.columns.tolist()[1][:20]}..., etc.")

# except Exception as e:
#     print(f"Error writing file: {e}") 






import pandas as pd
import re
import numpy as np
import torch
import os
from io import StringIO
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from transformers import BertTokenizer, BertModel
import 

# --- 1. CONFIGURATION AND DIRECTORY SETUP ---
OUTPUT_DIR = 'extracted_features'
FUSED_FILE_NAME = 'fused_text_features.csv'
FUSED_FILE_PATH = os.path.join(OUTPUT_DIR, FUSED_FILE_NAME)

# Create the output directory if it does not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory '{OUTPUT_DIR}' ensured.")

# --- 2. BERT MODEL INITIALIZATION ---

# BERT-Base Uncased is selected (340M parameters, Apache 2.0 license) [2]
MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 128
BERT_OUTPUT_DIM = 768

# Initialize Tokenizer and Encoder
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

try:
    bert_encoder = BertModel.from_pretrained(MODEL_NAME)
    # Freeze weights for feature extraction (Transfer Learning approach) [3]
    for param in bert_encoder.parameters():
        param.requires_grad = False
    bert_encoder.eval() 
except Exception as e:
    print(f"BERT model initialization failed (Check internet/resources): {e}")
    bert_encoder = None


# --- 3. HIGH-PRECISION STRUCTURED ATTRIBUTE EXTRACTION ---

# Fallback Dictionary: Contains the most unambiguous brands from the provided samples.
# This list should be replaced by the user's comprehensive list after running the heuristic on the full dataset.[4]
KNOWN_BRANDS = 

def extract_structured_attributes(content):
    """
    Extracts high-precision, price-driving features: Brand and Item Pack Quantity (IPQ).
    """
    
    # Use Item Name field for extraction
    item_name_match = re.search(r"Item Name:\s*(.*?)(?:\n|$)", content, re.IGNORECASE)
    core_line = item_name_match.group(1).strip() if item_name_match else content.strip()
    
    brand = 'Unknown'
    
    # 3.1. Brand Extraction (Dictionary Lookup Simulation)
    # Uses longest prefix match against the KNOWN_BRANDS list [4]
    sorted_brands = sorted(KNOWN_BRANDS, key=len, reverse=True)
    for b in sorted_brands:
        if core_line.startswith(b):
            brand = b
            break
            
    # 3.2. IPQ Extraction (Rule-Based Parsing) [5]
    pack_match = re.search(r'Pack of (\d+)', content, re.I)
    pack_count = int(pack_match.group(1)) if pack_match else 1

    # Extracting the explicit 'Value' and 'Unit' from the structured fields for consistency
    value_match = re.search(r"Value:\s*(\d+(?:\.\d+)?)", content, re.IGNORECASE)
    unit_size = float(value_match.group(1)) if value_match else 0.0

    # 3.3. Residual Text Cleaning
    cleaned_text = content
    # Remove extracted brands from residual text
    for b in KNOWN_BRANDS:
        cleaned_text = cleaned_text.replace(b, ' ', 1)
    
    # Remove all quantity/unit mentions and punctuation to prepare text for BERT [6]
    cleaned_text = re.sub(r'Item Name:.*?\n', '', cleaned_text, flags=re.DOTALL)
    cleaned_text = re.sub(r'Bullet Point [0-9]:', '', cleaned_text, flags=re.I)
    cleaned_text = re.sub(r'\(\s*Pack of \d+\s*\)', ' ', cleaned_text, flags=re.I)
    cleaned_text = re.sub(r'Value:\s*.*?\n|Unit:\s*.*?\n', ' ', cleaned_text, flags=re.DOTALL)
    cleaned_text = re.sub(r'[^\w\s]', ' ', cleaned_text)
    cleaned_text = ' '.join(cleaned_text.split()).lower()

    return brand, pack_count, unit_size, cleaned_text

# --- 5. DEEP TEXT EMBEDDING PIPELINE (BERT) ---

def generate_bert_embeddings(texts):
    """ Generates contextual embeddings using the BERT Encoder. """
    if not bert_encoder:
        return np.zeros((len(texts), BERT_OUTPUT_DIM))

    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LEN)
    
    with torch.no_grad():
        outputs = bert_encoder(**inputs)
    
    cls_embeddings = outputs.last_hidden_state[:, 0, :]
    return cls_embeddings.numpy()

# --- 6. FEATURE EXTRACTION EXECUTION ---

# Load Data (Replace with your actual data loading command)
data = {'sample_id': ,
        'catalog_content':}
df = pd.DataFrame(data)
df.set_index('sample_id', inplace=True) # Set index for concatenation

# 6.1. Structured and Residual Text Extraction
results = df['catalog_content'].apply(
    lambda x: pd.Series(extract_structured_attributes(x), 
                        index=['brand', 'ipq_pack_count', 'ipq_unit_size', 'residual_text'])
)
df = pd.join(results, how='left')

# 6.2. Deep Text Feature Generation
bert_embeddings_array = generate_bert_embeddings(df['residual_text'].tolist())
bert_df = pd.DataFrame(bert_embeddings_array, index=df.index, 
                       columns=)


# --- 7. ENCODING AND SCALING FINAL TEXT FEATURES ---

# 7.1. One-Hot Encode (OHE) Brand [6]
ohe_brand = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
brand_encoded = ohe_brand.fit_transform(df[['brand']])
brand_df = pd.DataFrame(brand_encoded, index=df.index, 
                        columns=ohe_brand.get_feature_names_out(['brand']))

# 7.2. Scale Numerical Features (IPQ and Unit Size)
scaler_ipq = StandardScaler()
ipq_scaled = scaler_ipq.fit_transform(df[['ipq_pack_count', 'ipq_unit_size']])
ipq_df = pd.DataFrame(ipq_scaled, index=df.index,
                      columns=['ipq_pack_count_scaled', 'ipq_unit_size_scaled'])

# 7.3. Final Text Feature Fusion (Structured + Deep)
X_text_fused = pd.concat(, axis=1).reset_index(drop=True)

# Rename sample_id column back
X_text_fused.rename(columns={'sample_id': 'sample_id'}, inplace=True)


# --- 8. SAVE RESULTS TO SEPARATE FILE ---

try:
    X_text_fused.to_csv(FUSED_FILE_PATH, index=False)
    
    # Save the feature column names (important details) to a separate file for tracking
    COLUMNS_FILE_PATH = os.path.join(OUTPUT_DIR, 'feature_columns.txt')
    with open(COLUMNS_FILE_PATH, 'w') as f:
        f.write(f"Total Features: {X_text_fused.shape[1] - 1}\n")
        f.write("Features List (excluding sample_id):\n")
        for col in X_text_fused.columns[1:]:
            f.write(f"- {col}\n")

    print(f"\n--- SUCCESS: Fused Text Feature Matrix Saved ---")
    print(f"File created successfully at: {FUSED_FILE_PATH}")
    print(f"Feature list saved to: {COLUMNS_FILE_PATH}")

except Exception as e:
    print(f"Error writing file: {e}")