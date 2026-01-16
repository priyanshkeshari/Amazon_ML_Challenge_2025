import os
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, load_npz
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from scipy.optimize import minimize

# ---------------- CONFIG ----------------
MODEL_DIR = 'models'  # directory where models and artifacts are saved
TABULAR_TRAIN_PATH = 'features/train_features_competitive.csv'
IMAGE_EMBEDDINGS_PATH = 'features/train_vit_embeddings.npy'
TEXT_FEATURES_PATH = 'features/train_sbert_embeddings.npz'  # updated
NUMERICAL_COLS = [
    'text_length', 'word_count', 'num_bullet_points', 'pack_size_agg',
    'unit_quantity_oz_agg', 'total_quantity_agg', 'num_quantity_mentions',
    'quantity_consistency'
]
KEYWORD_COLS_PREFIX = 'is_'  # any column starting with 'is_'
UNIT_COL = 'quantity_unit'  # categorical column for one-hot encoding

# ---------------- SMAPE ----------------
def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / (denominator + 1e-8)) * 100

# ---------------- Ensemble Functions ----------------
def ensemble_smape(weights, preds_list, y_true):
    ensemble_pred = np.zeros_like(preds_list[0])
    for w, pred in zip(weights, preds_list):
        ensemble_pred += w * pred
    ensemble_pred /= np.sum(weights)
    return smape(y_true, ensemble_pred)

def find_best_weights(preds_list, y_true):
    n_models = len(preds_list)
    init_weights = np.ones(n_models) / n_models
    bounds = [(0,1)] * n_models
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w)-1}
    result = minimize(
        ensemble_smape,
        x0=init_weights,
        args=(preds_list, y_true),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    if result.success:
        best_weights = result.x
        best_smape = ensemble_smape(best_weights, preds_list, y_true)
        print(f"✅ Optimized Weights: {best_weights}")
        print(f"✅ SMAPE of Weighted Ensemble: {best_smape:.4f}%")
        return best_weights
    else:
        raise RuntimeError("Optimization failed: "+result.message)

def weighted_ensemble_predict(preds_list, weights):
    ensemble_pred = np.zeros_like(preds_list[0])
    for w, pred in zip(weights, preds_list):
        ensemble_pred += w * pred
    ensemble_pred /= np.sum(weights)
    return ensemble_pred

# ---------------- Main Pipeline ----------------
def main():
    # 1. Load tabular, image, and text features
    df = pd.read_csv(TABULAR_TRAIN_PATH)
    image_embeddings = np.load(IMAGE_EMBEDDINGS_PATH)
    text_features = load_npz(TEXT_FEATURES_PATH)

    y_true = df['price'].values

    # 2. Preprocess tabular features
    ohe_unit = joblib.load(os.path.join(MODEL_DIR, 'ohe_unit.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))

    unit_features = ohe_unit.transform(df[[UNIT_COL]])
    numerical_features = scaler.transform(df[NUMERICAL_COLS].fillna(0))
    keyword_cols = [col for col in df.columns if col.startswith(KEYWORD_COLS_PREFIX)]

    X = hstack([
        numerical_features,
        df[keyword_cols].values,
        unit_features,
        image_embeddings,
        text_features
    ]).tocsr()
    print(f"Final combined feature matrix shape: {X.shape}")

    # 3. Load trained models
    lgb_models = joblib.load(os.path.join(MODEL_DIR, 'lgbm_models_all_folds.pkl'))
    xgb_models = joblib.load(os.path.join(MODEL_DIR, 'xgb_models_all_folds.pkl'))
    cat_models = joblib.load(os.path.join(MODEL_DIR, 'cat_models_all_folds.pkl'))

    # 4. Generate predictions from each model (averaging over folds)
    def avg_fold_predictions(models, X):
        preds = []
        for model in models:
            preds.append(model.predict(X))
        return np.mean(preds, axis=0)

    preds_lgb = avg_fold_predictions(lgb_models, X)
    preds_xgb = avg_fold_predictions(xgb_models, X)
    preds_cat = avg_fold_predictions(cat_models, X)

    preds_list = [preds_lgb, preds_xgb, preds_cat]

    # 5. Optimize ensemble weights
    best_weights = find_best_weights(preds_list, y_true)

    # 6. Final ensemble prediction
    final_preds = weighted_ensemble_predict(preds_list, best_weights)
    final_smape = smape(y_true, final_preds)
    print(f"\n✅ Final SMAPE after Weighted Voting Ensemble: {final_smape:.4f}%")

    # 7. Save ensemble weights for future inference
    np.save(os.path.join(MODEL_DIR, 'ensemble_weights.npy'), best_weights)
    print("✅ Ensemble weights saved successfully.")

if __name__ == "__main__":
    main()
