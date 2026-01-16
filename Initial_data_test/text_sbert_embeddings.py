# ===============================================================
# 📘 SBERT Embedding Generator (Local-ready)
# Generates text embeddings using "sentence-transformers/all-MiniLM-L6-v2"
# Saves output to ./extracted_features/sbert_embeddings.csv
# ===============================================================

# ------------------ INSTALL DEPENDENCIES ------------------
# Uncomment below if not already installed
# !pip install -q sentence-transformers huggingface_hub tqdm pandas numpy

# ------------------ IMPORTS ------------------
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download

# ------------------ CONFIG ------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_MODEL_DIR = "./local_sbert_model"
DATA_PATH = r"C:\Users\acer\Desktop\Programming\ML\Hackathons\Amazon_ML_Challenge_2025\Initial_data_test\extracted_features\AMLC\train_features_v5.csv"  # 👈 change this if your CSV is elsewhere
TEXT_FEATURES = ['item_name', 'description', 'cleaned_text']
COLUMNS_TO_DROP = ['image_link', 'full_text']
SAVE_DIR = r"C:\Users\acer\Desktop\Programming\ML\Hackathons\Amazon_ML_Challenge_2025\Initial_data_test\extracted_features"
SAVE_PATH = os.path.join(SAVE_DIR, "train_features_v5_sbert_embeddings.npy")
BATCH_SIZE = 128

# ------------------ SETUP OUTPUT DIR ------------------
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------ LOAD MODEL ------------------
print(f"🚀 Attempting to load model: {MODEL_NAME}")
try:
    model = SentenceTransformer(MODEL_NAME)
    print("✅ Model loaded directly from Hugging Face")
except Exception as e:
    print(f"⚠️ Direct load failed: {e}")
    print("⬇️ Downloading manually...")
    snapshot_download(repo_id=MODEL_NAME, local_dir=LOCAL_MODEL_DIR)
    model = SentenceTransformer(LOCAL_MODEL_DIR)
    print("✅ Model loaded from local cache")

# ------------------ LOAD DATA ------------------
print(f"\n📂 Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

df = df.drop(columns=COLUMNS_TO_DROP, errors="ignore")
print(f"🧹 Dropped columns: {COLUMNS_TO_DROP}")
print(f"✅ Data ready for embedding | Shape: {df.shape}")

# ------------------ GENERATE EMBEDDINGS ------------------
df_embedded = df.copy()

for text_col in TEXT_FEATURES:
    if text_col not in df_embedded.columns:
        print(f"⚠️ Column '{text_col}' not found in dataframe, skipping.")
        continue

    print(f"\n⚙️ Generating embeddings for column: '{text_col}'")
    texts = df_embedded[text_col].fillna('').astype(str).tolist()
    embeddings = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch = texts[i:i+BATCH_SIZE]
        emb = model.encode(batch, batch_size=BATCH_SIZE, convert_to_numpy=True, show_progress_bar=False)
        embeddings.append(emb)

    embeddings = np.vstack(embeddings)
    print(f"✅ Embeddings shape for '{text_col}': {embeddings.shape}")

    # Add embeddings to dataframe with column prefixes
    emb_cols = [f"{text_col}_emb_{i}" for i in range(embeddings.shape[1])]
    emb_df = pd.DataFrame(embeddings, columns=emb_cols, index=df_embedded.index)
    df_embedded = pd.concat([df_embedded, emb_df], axis=1)

# ------------------ DROP ORIGINAL TEXT COLUMNS ------------------
df_embedded = df_embedded.drop(columns=TEXT_FEATURES)
print(f"\n✅ Final embedded dataframe shape: {df_embedded.shape}")

# ------------------ SAVE TO CSV ------------------
df_embedded.to_csv(SAVE_PATH, index=False)
print(f"\n💾 Saved all embeddings to: {SAVE_PATH}")
print("🎉 Done! You can now use these features for modeling.")