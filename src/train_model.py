import json
import pickle
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preprocessing import TitanicFeatureEngineer

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "titanic.csv"
PLOTS_DIR = ROOT / "plots"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
for d in [PLOTS_DIR, MODELS_DIR, OUTPUTS_DIR]:
    d.mkdir(exist_ok=True)

RANDOM_STATE = 42


def make_preprocessor():
    numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FareLog", "FamilySize", "IsAlone", "CabinPresent"]
    categorical_features = ["Sex", "Embarked", "Title", "Deck"]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features)
    ])


def make_pipeline(model):
    return Pipeline([
        ("feature_engineering", TitanicFeatureEngineer()),
        ("preprocessor", make_preprocessor()),
        ("model", model)
    ])


def main():
    df = pd.read_csv(DATA_PATH)
    target = "Survived"
    X = df.drop(columns=[target])
    y = df[target]

    # Basic EDA plots
    plt.figure(figsize=(6, 4))
    y.value_counts().sort_index().plot(kind="bar")
    plt.title("Survival Class Distribution")
    plt.xlabel("Survived (0 = No, 1 = Yes)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "survival_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    df.groupby("Sex")[target].mean().plot(kind="bar")
    plt.title("Survival Rate by Sex")
    plt.xlabel("Sex")
    plt.ylabel("Survival Rate")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "survival_by_sex.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    df.groupby("Pclass")[target].mean().plot(kind="bar")
    plt.title("Survival Rate by Passenger Class")
    plt.xlabel("Passenger Class")
    plt.ylabel("Survival Rate")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "survival_by_pclass.png", dpi=150)
    plt.close()

    fe_preview = TitanicFeatureEngineer().fit_transform(df)
    plt.figure(figsize=(7, 4))
    fe_preview.groupby("Title")[target].mean().sort_values(ascending=False).plot(kind="bar")
    plt.title("Survival Rate by Extracted Title")
    plt.xlabel("Title")
    plt.ylabel("Survival Rate")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "survival_by_title.png", dpi=150)
    plt.close()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=5, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE)
    }

    rows = []
    fitted = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for name, model in models.items():
        pipe = make_pipeline(model)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="accuracy")
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        rows.append({
            "model": name,
            "cv_accuracy_mean": float(cv_scores.mean()),
            "cv_accuracy_std": float(cv_scores.std()),
            "test_accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "f1": float(f1_score(y_test, preds, zero_division=0))
        })
        fitted[name] = pipe

    metrics_df = pd.DataFrame(rows).sort_values(["f1", "test_accuracy"], ascending=False)
    metrics_df.to_csv(OUTPUTS_DIR / "model_comparison_metrics.csv", index=False)

    best_name = metrics_df.iloc[0]["model"]
    best_model = fitted[best_name]
    y_pred = best_model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    with open(OUTPUTS_DIR / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Survived", "Survived"])
    disp.plot(values_format="d")
    plt.title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix_best_model.png", dpi=150)
    plt.close()

    # Feature importance / explainability for tree-based model if selected; otherwise coefficient importance.
    try:
        preprocessor = best_model.named_steps["preprocessor"]
        feature_names = preprocessor.get_feature_names_out()
        final_model = best_model.named_steps["model"]
        if hasattr(final_model, "feature_importances_"):
            importance = final_model.feature_importances_
        elif hasattr(final_model, "coef_"):
            importance = np.abs(final_model.coef_[0])
        else:
            importance = np.zeros(len(feature_names))
        imp = pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values("importance", ascending=False).head(15)
        imp.to_csv(OUTPUTS_DIR / "feature_importance.csv", index=False)
        plt.figure(figsize=(8, 6))
        plt.barh(imp["feature"][::-1], imp["importance"][::-1])
        plt.title(f"Top Feature Importance - {best_name}")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "feature_importance_best_model.png", dpi=150)
        plt.close()
    except Exception as e:
        with open(OUTPUTS_DIR / "feature_importance_error.txt", "w") as f:
            f.write(str(e))

    joblib.dump(best_model, MODELS_DIR / "best_titanic_survival_model.joblib")
    with open(MODELS_DIR / "best_titanic_survival_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    summary = {
        "best_model": best_name,
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "model_file": "models/best_titanic_survival_model.joblib"
    }
    with open(OUTPUTS_DIR / "best_model_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
