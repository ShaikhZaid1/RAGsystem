"""
train.py
--------
Phase 1: Full ML pipeline.
  - Loads & preprocesses data (handles missing values, scales features, encodes categoricals)
  - Trains Random Forest, XGBoost, and Logistic Regression
  - Compares models on Accuracy, Precision, Recall, F1, and ROC-AUC
  - Exports the best model + preprocessing pipeline via joblib
"""

import warnings
warnings.filterwarnings("ignore")

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_PATH  = BASE_DIR / "customer_churn.csv"
MODEL_PATH = BASE_DIR / "best_model.joblib"
META_PATH  = BASE_DIR / "model_meta.json"


# ── 1. Load Data ─────────────────────────────────────────────────────────────
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load dataset and perform basic sanity checks."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Run `python generate_data.py` first, or place your CSV at that path."
        )
    df = pd.read_csv(path)
    print(f"✅  Loaded {len(df):,} rows × {df.shape[1]} columns")
    print(f"    Churn rate : {df['Churn'].mean():.1%}")
    print(f"    Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}\n")
    return df


# ── 2. Feature Engineering ────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame):
    """Split features/target and define column groups."""
    df = df.copy()

    # Derived feature: average charge per month of tenure
    df["Avg_Charge_Per_Month"] = (
        df["Total_Charges"] / df["Tenure_Months"].replace(0, 1)
    ).round(2)

    target = "Churn"
    numeric_cols = [
        "Age", "Tenure_Months", "Monthly_Charges", "Total_Charges",
        "Num_Products", "Support_Calls", "Satisfaction_Score",
        "Avg_Charge_Per_Month",
    ]
    categorical_cols = ["Contract_Type", "Payment_Method", "Internet_Service"]

    X = df[numeric_cols + categorical_cols]
    y = df[target]
    return X, y, numeric_cols, categorical_cols


# ── 3. Preprocessing Pipeline ─────────────────────────────────────────────────
def build_preprocessor(numeric_cols, categorical_cols) -> ColumnTransformer:
    """
    Numeric  → impute (median) → standard scale
    Categorical → impute (most_frequent) → one-hot encode
    """
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])


# ── 4. Model Definitions ──────────────────────────────────────────────────────
def get_models():
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            scale_pos_weight=3, eval_metric="logloss",
            random_state=42, n_jobs=-1, verbosity=0,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced",
            C=0.5, random_state=42,
        ),
    }


# ── 5. Train & Evaluate ───────────────────────────────────────────────────────
def evaluate_model(name, pipeline, X_train, X_test, y_train, y_test) -> dict:
    """Fit a pipeline and return a dict of metrics."""
    pipeline.fit(X_train, y_train)
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "Model":     name,
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC":   round(roc_auc_score(y_test, y_proba), 4),
    }
    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")
    for k, v in metrics.items():
        if k != "Model":
            print(f"  {k:<12}: {v:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Stay','Churn'])}")
    return metrics


def compare_models(X, y, preprocessor):
    """Train all models, return comparison DataFrame + pipelines dict."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results  = []
    pipelines = {}

    for name, clf in get_models().items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier",   clf),
        ])
        metrics = evaluate_model(name, pipe, X_train, X_test, y_train, y_test)
        results.append(metrics)
        pipelines[name] = pipe

    df_results = pd.DataFrame(results).set_index("Model")
    print("\n" + "═"*55)
    print("  MODEL COMPARISON SUMMARY")
    print("═"*55)
    print(df_results.to_string())
    return df_results, pipelines, X_train, X_test, y_train, y_test


# ── 6. Export Best Model ──────────────────────────────────────────────────────
def export_best(df_results, pipelines, numeric_cols, categorical_cols):
    """Pick the model with the highest ROC-AUC and save it."""
    best_name = df_results["ROC-AUC"].idxmax()
    best_pipe = pipelines[best_name]

    joblib.dump(best_pipe, MODEL_PATH)

    meta = {
        "best_model":      best_name,
        "numeric_cols":    numeric_cols,
        "categorical_cols": categorical_cols,
        "metrics":         df_results.loc[best_name].to_dict(),
        "all_metrics":     df_results.reset_index().to_dict(orient="records"),
    }
    META_PATH.write_text(json.dumps(meta, indent=2))

    print(f"\n🏆  Best model : {best_name}  (ROC-AUC: {df_results.loc[best_name,'ROC-AUC']:.4f})")
    print(f"💾  Saved → {MODEL_PATH}")
    print(f"📋  Metadata → {META_PATH}")
    return best_name, best_pipe


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "═"*55)
    print("  CUSTOMER CHURN — ML TRAINING PIPELINE")
    print("═"*55 + "\n")

    df = load_data()
    X, y, numeric_cols, categorical_cols = engineer_features(df)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    df_results, pipelines, X_train, X_test, y_train, y_test = compare_models(
        X, y, preprocessor
    )
    export_best(df_results, pipelines, numeric_cols, categorical_cols)

    print("\n✅  Training complete. Run `streamlit run app.py` to launch the UI.\n")


if __name__ == "__main__":
    main()
