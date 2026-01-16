# ==============================================================
# 🚀 GPU-Enabled Ensemble Training: LightGBM + XGBoost + CatBoost
# Updated for your new dataset & embeddings
# ==============================================================

import os
import time
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import hstack, load_npz

from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import mean_absolute_error

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# ------------------ CONFIG ------------------
TABULAR_TRAIN_PATH = "features/train_features_competitive.csv"
IMAGE_EMBEDDINGS_PATH = "features/train_vit_embeddings.npy"
TEXT_EMBEDDINGS_PATH = "extracted_features/sbert_embeddings.csv"

MODEL_OUTPUT_DIR = "models_ensemble"
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

N_SPLITS = 5
RANDOM_SEED = 42

# ------------------ METRIC ------------------
def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / (denominator + 1e-8)) * 100

# ------------------ MAIN TRAINING PIPELINE ------------------
def main():
    start_time = time.time()
    print("--- Starting GPU Ensemble Training Pipeline ---")

    # 1️⃣ Load Data & Embeddings
    print("\n[1/4] Loading feature sources...")
    df = pd.read_csv(TABULAR_TRAIN_PATH)
    image_embeddings = np.load(IMAGE_EMBEDDINGS_PATH)
    text_embeddings = pd.read_csv(TEXT_EMBEDDINGS_PATH).values
    print(f"Tabular: {df.shape}, Image Embeddings: {image_embeddings.shape}, Text Embeddings: {text_embeddings.shape}")

    y = np.log1p(df['price'])

    # 2️⃣ Tabular Features
    print("\n[2/4] Preparing tabular features...")
    categorical_cols = ['quantity_unit']  # Update as per new dataset
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    cat_features = ohe.fit_transform(df[categorical_cols])

    numerical_cols = [
        'pack_size_agg', 'unit_quantity_oz_agg', 'total_quantity_agg',
        'num_quantity_mentions', 'quantity_consistency', 'word_count'
    ]
    scaler = MinMaxScaler()
    num_features = scaler.fit_transform(df[numerical_cols].fillna(0))

    # 3️⃣ Combine All Features
    print("\n[3/4] Combining features into final matrix...")
    X = hstack([
        num_features,
        cat_features,
        image_embeddings,
        text_embeddings
    ]).tocsr()
    print(f"Final feature matrix shape: {X.shape}")

    # 4️⃣ K-Fold CV Ensemble Training
    print("\n[4/4] Training ensemble models with 5-Fold CV...")
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    oof_preds = np.zeros(len(df))
    lgb_models, xgb_models, cat_models = [], [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # ----- LightGBM -----
        lgb_params = {
            'objective': 'regression_l1',
            'n_estimators': 3000,
            'learning_rate': 0.02,
            'num_leaves': 50,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 1,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'verbose': -1,
            'n_jobs': -1,
            'boosting_type': 'gbdt',
            'device': 'gpu',
            'seed': RANDOM_SEED
        }
        lgb_model = lgb.LGBMRegressor(**lgb_params)
        lgb_model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      eval_metric='mae',
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        lgb_models.append(lgb_model)
        print("✅ LightGBM trained")

        # ----- XGBoost -----
        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=3000,
            learning_rate=0.02,
            max_depth=10,
            subsample=0.7,
            colsample_bytree=0.7,
            tree_method='gpu_hist',
            predictor='gpu_predictor',
            n_jobs=-1,
            seed=RANDOM_SEED
        )
        xgb_model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      eval_metric='mae',
                      early_stopping_rounds=150,
                      verbose=False)
        xgb_models.append(xgb_model)
        print("✅ XGBoost trained")

        # ----- CatBoost -----
        cat_model = CatBoostRegressor(
            iterations=3000,
            learning_rate=0.02,
            depth=10,
            loss_function='MAE',
            eval_metric='MAE',
            task_type='GPU',
            verbose=0,
            random_seed=RANDOM_SEED
        )
        cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=150)
        cat_models.append(cat_model)
        print("✅ CatBoost trained")

        # ----- Ensemble Prediction -----
        val_pred = (lgb_model.predict(X_val) +
                    xgb_model.predict(X_val) +
                    cat_model.predict(X_val)) / 3
        oof_preds[val_idx] = val_pred

    # 5️⃣ Final Evaluation
    oof_preds_rescaled = np.expm1(oof_preds)
    oof_preds_rescaled[oof_preds_rescaled < 0] = 0
    final_smape = smape(df['price'], oof_preds_rescaled)
    print("\n" + "="*50)
    print(f"🎯 Final CV SMAPE Score: {final_smape:.4f}%")
    print("="*50)

    # 6️⃣ Save Models & Preprocessors
    print("\n💾 Saving models and preprocessing artifacts...")
    joblib.dump(lgb_models, os.path.join(MODEL_OUTPUT_DIR, 'lgb_models.pkl'))
    joblib.dump(xgb_models, os.path.join(MODEL_OUTPUT_DIR, 'xgb_models.pkl'))
    joblib.dump(cat_models, os.path.join(MODEL_OUTPUT_DIR, 'cat_models.pkl'))
    joblib.dump(ohe, os.path.join(MODEL_OUTPUT_DIR, 'ohe_categorical.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_OUTPUT_DIR, 'scaler_numeric.pkl'))
    print("✅ All artifacts saved successfully!")

    end_time = time.time()
    print(f"\n⏱ Script finished in {end_time - start_time:.2f} seconds")

# ------------------ RUN ------------------
if __name__ == "__main__":
    main()




# ------------------ Save Models & Weights ------------------
print("\n💾 Saving models and preprocessing artifacts...")

# Save joblib objects (full models)
joblib.dump(lgb_models, os.path.join(MODEL_OUTPUT_DIR, 'lgb_model_1.pkl'))
joblib.dump(xgb_models, os.path.join(MODEL_OUTPUT_DIR, 'xgb_model_1.pkl'))
joblib.dump(cat_models, os.path.join(MODEL_OUTPUT_DIR, 'cat_model_1.pkl'))

# Save preprocessing artifacts
joblib.dump(ohe, os.path.join(MODEL_OUTPUT_DIR, 'ohe_categorical.pkl'))
joblib.dump(scaler, os.path.join(MODEL_OUTPUT_DIR, 'scaler_numeric.pkl'))

# ------------------ Save Native Model Weights ------------------
# LightGBM
for i, model in enumerate(lgb_models):
    model.booster_.save_model(os.path.join(MODEL_OUTPUT_DIR, f"lgb_model_fold{i+1}.txt"))

# XGBoost
for i, model in enumerate(xgb_models):
    model.save_model(os.path.join(MODEL_OUTPUT_DIR, f"xgb_model_fold{i+1}.json"))

# CatBoost
for i, model in enumerate(cat_models):
    model.save_model(os.path.join(MODEL_OUTPUT_DIR, f"cat_model_fold{i+1}.cbm"))

print("✅ All artifacts and native model weights saved successfully!")
