import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import os
import time

# --- Configuration ---
FEATURE_TRAIN_PATH = r'C:\Users\acer\Desktop\Programming\ML\Hackathons\Amazon_ML_Challenge_2025\Initial_data_test\extracted_features\train_features_v5.csv'
FEATURE_TEST_PATH = r'C:\Users\acer\Desktop\Programming\ML\Hackathons\Amazon_ML_Challenge_2025\Initial_data_test\extracted_features\test_features_v5.csv'
OUTPUT_DIR = 'extracted_features'

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # 768-dimensional embeddings per text
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_EMBEDDINGS_PATH = os.path.join(OUTPUT_DIR, 'train_v5_sbert_embeddings_multi_text.npy')
TEST_EMBEDDINGS_PATH = os.path.join(OUTPUT_DIR, 'test_v5_sbert_embeddings_multi_text.npy')

# Columns to generate embeddings for
TEXT_FEATURES = ['item_name', 'description', 'cleaned_text']

# ---------------------------
# Function to generate embeddings for a list of texts
# ---------------------------
def generate_embeddings(texts, tokenizer, model, batch_size=128, device="cuda"):
    all_embeddings = []
    model.eval()
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(**encoded_input)
            # Mean pooling over token embeddings
            batch_embeddings = output.last_hidden_state.mean(dim=1)
            all_embeddings.append(batch_embeddings.cpu().numpy())
    return np.vstack(all_embeddings)

# ---------------------------
# Function to process a DataFrame with multiple text columns
# ---------------------------
def process_dataframe(df, text_features, tokenizer, model, batch_size=128, device="cuda"):
    embeddings_list = []
    for col in text_features:
        print(f"Generating embeddings for column: {col}")
        texts = df[col].astype(str).fillna('').tolist()
        col_embeddings = generate_embeddings(texts, tokenizer, model, batch_size=batch_size, device=device)
        embeddings_list.append(col_embeddings)
    # Concatenate embeddings for all text columns
    return np.hstack(embeddings_list)

# ---------------------------
# Main function
# ---------------------------
def main():
    start_time = time.time()
    print(f"--- Starting multi-text embedding extraction using {MODEL_NAME} on {DEVICE} ---")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load data
    print("\n[1/3] Loading CSV files...")
    try:
        train_df = pd.read_csv(FEATURE_TRAIN_PATH)
        test_df = pd.read_csv(FEATURE_TEST_PATH)
    except FileNotFoundError as e:
        print(f"[FATAL ERROR] {e}")
        return

    # 2. Load tokenizer & model
    print("\n[2/3] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    print("Model loaded successfully.")

    # 3. Generate embeddings
    print("\n[3/3] Generating train embeddings...")
    train_embeddings = process_dataframe(train_df, TEXT_FEATURES, tokenizer, model, batch_size=BATCH_SIZE, device=DEVICE)
    np.save(TRAIN_EMBEDDINGS_PATH, train_embeddings)
    print(f"Train embeddings saved at {TRAIN_EMBEDDINGS_PATH} | Shape: {train_embeddings.shape}")

    print("\nGenerating test embeddings...")
    test_embeddings = process_dataframe(test_df, TEXT_FEATURES, tokenizer, model, batch_size=BATCH_SIZE, device=DEVICE)
    np.save(TEST_EMBEDDINGS_PATH, test_embeddings)
    print(f"Test embeddings saved at {TEST_EMBEDDINGS_PATH} | Shape: {test_embeddings.shape}")

    end_time = time.time()
    print(f"\n--- Finished in {end_time - start_time:.2f} seconds ---")

# ---------------------------
# Run script
# ---------------------------
if __name__ == "__main__":
    main()
