# -*- coding: utf-8 -*-

# ===============================================================
# 📘 SBERT Embedding Generator (.npy)
# Generates embeddings for specified text columns and saves as .npy
# ===============================================================

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch

# ------------------ CONFIG ------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CSV_PATH = r"C:\Users\acer\Desktop\Programming\ML\Hackathons\Amazon_ML_Challenge_2025\Initial_data_test\extracted_features\train_features_v5.csv"
TEXT_FEATURES = ['item_name', 'description', 'cleaned_text']
SAVE_PATH = r"C:\Users\acer\Desktop\Programming\ML\Hackathons\Amazon_ML_Challenge_2025\Initial_data_test\extracted_features\train_features_v5_sbert_embeddings.npy"
BATCH_SIZE = 128

# ------------------ DEVICE SETUP ------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🚀 Using device: {device}")

# ------------------ LOAD MODEL ------------------
print(f"🚀 Loading SBERT model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME, device=device)

# ------------------ LOAD DATA ------------------
print(f"📂 Loading data from: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(f"✅ Data loaded | Shape: {df.shape}")

# ------------------ GENERATE EMBEDDINGS ------------------
all_embeddings = []

for text_col in TEXT_FEATURES:
    if text_col not in df.columns:
        print(f"⚠️ Column '{text_col}' not found. Skipping...")
        continue

    print(f"\n⚙️ Generating embeddings for column: '{text_col}'")
    texts = df[text_col].fillna('').astype(str).tolist()
    embeddings_list = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch = texts[i:i+BATCH_SIZE]
        emb = model.encode(
            batch,
            batch_size=BATCH_SIZE,
            convert_to_numpy=True,
            show_progress_bar=False,
            device=device
        )
        embeddings_list.append(emb)

    embeddings_col = np.vstack(embeddings_list)
    print(f"✅ Shape of embeddings for '{text_col}': {embeddings_col.shape}")

    all_embeddings.append(embeddings_col)

# ------------------ CONCATENATE EMBEDDINGS ------------------
# Concatenate embeddings from all text columns horizontally
final_embeddings = np.hstack(all_embeddings)
print(f"\n✅ Final embeddings shape: {final_embeddings.shape}")

# ------------------ SAVE TO .npy ------------------
np.save(SAVE_PATH, final_embeddings)
print(f"\n💾 Embeddings saved to: {SAVE_PATH}")
