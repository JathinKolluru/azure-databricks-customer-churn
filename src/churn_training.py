
import argparse, os, joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

def load_data(path):
    df = pd.read_csv(path)
    # Standardize column names
    df.columns = [c.strip() for c in df.columns]
    return df

def build_pipeline(cat_cols, num_cols):
    cat = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    num = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median"))
    ])
    pre = ColumnTransformer([
        ("cat", cat, cat_cols),
        ("num", num, num_cols)
    ])
    model = LogisticRegression(max_iter=1000, solver="liblinear")
    pipe = Pipeline([("prep", pre), ("clf", model)])
    return pipe

def plot_confusion(y_true, y_pred, outdir):
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    out = os.path.join(outdir, "confusion_matrix.png")
    fig.savefig(out)
    plt.close(fig)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="reports/figures")
    ap.add_argument("--model", default="reports/churn_model.pkl")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.dirname(args.model), exist_ok=True)

    df = load_data(args.data)
    # Expect 'Churn' target with 'Yes'/'No'
    y = (df["Churn"].astype(str).str.lower() == "yes").astype(int)

    # Drop id-like fields from features
    X = df.drop(columns=["Churn", "CustomerID"], errors="ignore")

    # Identify columns
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number)]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    pipe = build_pipeline(cat_cols, num_cols)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:,1] if hasattr(pipe, "predict_proba") else None

    print("=== Classification Report (Test) ===")
    print(classification_report(y_test, y_pred, digits=4))

    if y_proba is not None:
        auc = roc_auc_score(y_test, y_proba)
        print(f"ROC-AUC: {auc:.4f}")

    # Save confusion matrix
    cm_path = plot_confusion(y_test, y_pred, args.out)
    print(f"Saved confusion matrix to: {cm_path}")

    # Save model
    joblib.dump(pipe, args.model)
    print(f"Saved trained model to: {args.model}")

if __name__ == "__main__":
    main()
