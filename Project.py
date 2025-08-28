import warnings
warnings.filterwarnings("ignore")

import os
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, precision_recall_curve,
    roc_auc_score, average_precision_score, matthews_corrcoef, cohen_kappa_score,
    ConfusionMatrixDisplay, jaccard_score
)
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, AdaBoostClassifier
)
from xgboost import XGBClassifier, XGBRFClassifier
from lightgbm import LGBMClassifier

from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
import joblib
from scipy.stats import chi2_contingency


# ------------------------------ Config ------------------------------
RANDOM_STATE = 42
TARGET = "Diabetes_012"
DATA_PATH = "diabetes_012_health_indicators_BRFSS2015.csv"
PLOTS_DIR = "plots"

EDA_SAVESETS = {
    # match your uploaded figure names
    "HighBP":        ["dataset_overview.png", "info.png"],        # two variants of HP plots you shared
    "CholCheck":     ["infocholcheck.png"],                        # cholcheck figure
    "HighChol":      ["infocol.png"]                               # highchol figure
}

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

ensure_dir(PLOTS_DIR)



def trim_outliers_zscore(X: pd.DataFrame, y: pd.Series, z_thresh: float = 3.0):
    num_cols = X.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        return X.reset_index(drop=True), y.reset_index(drop=True)
    z = (X[num_cols] - X[num_cols].mean()) / X[num_cols].std(ddof=0)
    mask = (z.abs() <= z_thresh).all(axis=1)
    return X.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True)


def ecdf(yvals):
    y = np.sort(yvals)
    x = np.arange(1, len(y) + 1) / len(y)
    return y, x


def plot_triplet_ecdf_qq(df, col, suptitle):
    """Histogram (+kde), Box, Violin; ECDF; Q-Q for a binary/near-binary column."""
    fig = plt.figure(figsize=(24, 12))

    # 1) Histogram
    ax1 = plt.subplot2grid((2, 3), (0, 0))
    sns.histplot(df[col], bins=30, kde=True, ax=ax1)
    ax1.set_title(f"Histogram of {col}")
    ax1.set_xlabel(col)
    ax1.set_ylabel("Count")

    # 2) Boxplot
    ax2 = plt.subplot2grid((2, 3), (0, 1))
    sns.boxplot(x=df[col], ax=ax2)
    ax2.set_title(f"Boxplot of {col}")
    ax2.set_xlabel(col)

    # 3) Violin
    ax3 = plt.subplot2grid((2, 3), (0, 2))
    sns.violinplot(x=df[col], inner=None, ax=ax3)
    ax3.set_title(f"Violin Plot of {col}")
    ax3.set_xlabel(col)

    # 4) ECDF
    ax4 = plt.subplot2grid((2, 3), (1, 0))
    y_sorted, x = ecdf(df[col].values)
    ax4.plot(y_sorted, x)
    ax4.set_title(f"ECDF of {col}")
    ax4.set_xlabel(col)
    ax4.set_ylabel("ECDF")

    # 5) QQ plot vs Normal (for display only; binary will show bands)
    from scipy import stats
    ax5 = plt.subplot2grid((2, 3), (1, 1))
    (osm, osr), (slope, intercept, r) = stats.probplot(df[col], dist="norm")
    ax5.scatter(osm, osr, s=10)
    ax5.plot(osm, slope*osm + intercept, 'r-')
    ax5.set_title(f"Q-Q Plot of {col}")
    ax5.set_xlabel("Theoretical quantiles")
    ax5.set_ylabel("Ordered Values")

    plt.suptitle(suptitle, y=1.02)
    plt.tight_layout()
    return fig


def save_fig(fig, filename):
    out_path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def evaluate_and_report(name, model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    acc  = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"\n=== {name} | Proper Evaluation (original test distribution) ===")
    print(f"Accuracy            : {acc:.4f}")
    print(f"Balanced Accuracy   : {bacc:.4f}")
    print(f"Macro Precision     : {prec:.4f}")
    print(f"Macro Recall        : {rec:.4f}")
    print(f"Macro F1            : {f1:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    print("Row totals (true-class counts in test):", cm.sum(axis=1).tolist())

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    ax.set_title(f"Confusion Matrix – {name} (Proper Eval)")
    plt.tight_layout()
    save_fig(fig, f"cm_proper_{name}.png")

    return {
        "acc": acc, "bacc": bacc, "prec_macro": prec,
        "rec_macro": rec, "f1_macro": f1, "cm": cm
    }


def diagnostic_balanced_cm_per_model(name, base_model, X_full, y_full):
    scaler_diag = StandardScaler()
    X_scaled_all = scaler_diag.fit_transform(X_full)
    X_res, y_res = ADASYN(random_state=RANDOM_STATE, n_neighbors=5).fit_resample(X_scaled_all, y_full)
    Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(
        X_res, y_res, test_size=0.2, random_state=RANDOM_STATE, shuffle=True
    )
    model_new = type(base_model)(**base_model.get_params())
    model_new.fit(Xd_tr, yd_tr)

    yd_pred = model_new.predict(Xd_te)
    cm_diag = confusion_matrix(yd_te, yd_pred)
    print(f"\n=== Diagnostic Balanced Confusion Matrix – {name} ===")
    print("Row totals (should be ~equal):", cm_diag.sum(axis=1).tolist())

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_diag)
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    ax.set_title(f"Confusion Matrix – {name} (Diagnostic Balanced)")
    plt.tight_layout()
    save_fig(fig, f"cm_diagnostic_balanced_{name}.png")


# ------------------------------ Main Flow ------------------------------
def main():
    # 1) Load
    df = pd.read_csv(DATA_PATH)
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])

    # 2) EDA (your exact style/combination)
    for col, files in EDA_SAVESETS.items():
        if col in df.columns:
            title = f"{col} – Distribution Suite"
            fig = plot_triplet_ecdf_qq(df, col, title)
            # Save duplicates where requested by your filenames list
            for fname in files:
                save_fig(fig, fname)

    # A few quick overview plots you used (BMI distribution + counts + heatmap)
    sns.set(style="whitegrid")
    if "BMI" in df.columns:
        fig = plt.figure(figsize=(8, 5))
        ax = plt.gca()
        sns.histplot(df["BMI"], bins=30, kde=True, ax=ax)
        ax.set_title("Distribution of BMI")
        ax.set_xlabel("BMI"); ax.set_ylabel("Frequency")
        plt.tight_layout()
        save_fig(fig, "bmi_hist.png")

    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()
    sns.countplot(x=y, ax=ax)
    ax.set_title("Count of Diabetes Cases")
    ax.set_xlabel(TARGET); ax.set_ylabel("Count")
    plt.tight_layout()
    save_fig(fig, "diabetes_count.png")

    fig = plt.figure(figsize=(16, 12))
    ax = plt.gca()
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True,
                cbar_kws={"label": "Correlation Coefficient"}, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation Heatmap of Diabetes Dataset")
    plt.tight_layout()
    save_fig(fig, "correlation_heatmap.png")

    # 3) Optional: light outlier trim (single pass)
    X_trim, y_trim = trim_outliers_zscore(X, y, z_thresh=3.0)

    # 4) Proper split on ORIGINAL data (keeps real class imbalance in test)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_trim, y_trim, test_size=0.2, random_state=RANDOM_STATE, stratify=y_trim
    )
    print("Train distribution:", Counter(y_tr))
    print("Test distribution :", Counter(y_te))

    # 5) Scale -> ADASYN on TRAIN ONLY (no leakage)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    X_tr_bal, y_tr_bal = ADASYN(random_state=RANDOM_STATE, n_neighbors=5).fit_resample(X_tr_s, y_tr)
    print("Balanced TRAIN distribution (ADASYN):", Counter(y_tr_bal))

    # 6) Models
    models = {
        "ExtraTrees": ExtraTreesClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "RandomForest": RandomForestClassifier(max_depth=7, random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "AdaBoost": AdaBoostClassifier(random_state=RANDOM_STATE),
        "XGBClassifier": XGBClassifier(random_state=RANDOM_STATE, eval_metric="mlogloss", use_label_encoder=False),
        "XGBRFClassifier": XGBRFClassifier(random_state=RANDOM_STATE),
        "LGBMClassifier": LGBMClassifier(n_estimators=150, random_state=RANDOM_STATE)
    }

    metrics_summary = {}
    trained_models = {}
    best_name, best_f1 = None, -1.0

    # 7) Train + proper evaluation (+ per-model CM)
    for name, model in models.items():
        model.fit(X_tr_bal, y_tr_bal)
        trained_models[name] = model
        m = evaluate_and_report(name, model, X_te_s, y_te)
        metrics_summary[name] = m
        if m["f1_macro"] > best_f1:
            best_f1 = m["f1_macro"]
            best_name = name

    print("\n=== Summary (sorted by Macro F1) ===")
    for name, m in sorted(metrics_summary.items(), key=lambda kv: kv[1]["f1_macro"], reverse=True):
        print(f"{name:18s} | Acc {m['acc']:.4f} | BAcc {m['bacc']:.4f} | "
              f"Prec {m['prec_macro']:.4f} | Rec {m['rec_macro']:.4f} | F1 {m['f1_macro']:.4f}")

    # 8) Diagnostic balanced CM per model
    print("\nGenerating Diagnostic Balanced Confusion Matrices (per model)...")
    for name, base_model in models.items():
        diagnostic_balanced_cm_per_model(name, base_model, X_trim, y_trim)

    # 9) Metrics comparison bar chart (your layout)
    metrics_to_plot = [
        ("acc", "Accuracy"),
        ("bacc", "Balanced Accuracy"),
        ("prec_macro", "Precision (Macro)"),
        ("rec_macro", "Recall (Macro)"),
        ("f1_macro", "F1-Score (Macro)"),
    ]
    fig = plt.figure(figsize=(20, 10))
    ax = plt.gca()
    names = list(metrics_summary.keys())
    width = 0.12
    x = np.arange(len(names))
    for i, (k, label) in enumerate(metrics_to_plot):
        vals = [metrics_summary[n][k] for n in names]
        ax.bar(x + i*width, vals, width, label=label)
    ax.set_xticks(x + width*2)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_title("Model Comparison Across Metrics")
    ax.set_ylabel("Score")
    ax.legend(loc="upper right", ncol=1, frameon=True)
    plt.tight_layout()
    save_fig(fig, "metrics_comparison.png")

    # 10) PR and ROC curves (your style)
    fig_pr = plt.figure(figsize=(12, 9))
    axpr = plt.gca()
    for name, model in trained_models.items():
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(X_te_s)
            # micro-average by flattening one-vs-rest probs for multi-class
            y_true = pd.get_dummies(y_te).values
            precision, recall, _ = precision_recall_curve(y_true.ravel(), probas.ravel())
            ap = average_precision_score(y_true, probas, average="micro")
            axpr.plot(recall, precision, label=f"{name} (AP = {ap:.2f})")
    axpr.set_title("Precision-Recall Curves")
    axpr.set_xlabel("Recall"); axpr.set_ylabel("Precision")
    axpr.legend(loc="lower left")
    plt.tight_layout()
    save_fig(fig_pr, "pr_curves.png")

    fig_roc = plt.figure(figsize=(12, 9))
    axroc = plt.gca()
    axroc.plot([0,1],[0,1],'k--',linewidth=1)
    for name, model in trained_models.items():
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(X_te_s)
            y_true = pd.get_dummies(y_te).values
            fpr, tpr, _ = roc_curve(y_true.ravel(), probas.ravel())
            auc = roc_auc_score(y_true, probas, average="micro", multi_class="ovr")
            axroc.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")
    axroc.set_title("ROC Curves")
    axroc.set_xlabel("False Positive Rate"); axroc.set_ylabel("True Positive Rate")
    axroc.legend(loc="lower right")
    plt.tight_layout()
    save_fig(fig_roc, "roc_curves.png")

    # 11) Feature importance (ExtraTrees) + save
    if "ExtraTrees" in trained_models:
        et = trained_models["ExtraTrees"]
        importances = et.feature_importances_
        if importances.shape[0] == X_trim.shape[1]:
            fi = pd.DataFrame({"Feature": X_trim.columns, "Importance": importances}) \
                   .sort_values("Importance", ascending=False)
            print("\nTop 15 Feature Importances (ExtraTrees):\n", fi.head(15).to_string(index=False))
            top = fi.head(15)
            fig = plt.figure(figsize=(8, 6))
            ax = plt.gca()
            ax.barh(top["Feature"][::-1], top["Importance"][::-1])
            ax.set_title("Top 15 Feature Importances – ExtraTrees")
            plt.tight_layout()
            save_fig(fig, "feature_importance_extratrees_top15.png")

    # 12) Save best model + scaler
    best_model = trained_models[best_name]
    joblib.dump(best_model, f"diabetes_best_{best_name}.pkl")
    joblib.dump(scaler, "diabetes_scaler.pkl")
    print(f"\nSaved best model: diabetes_best_{best_name}.pkl")
    print("Saved scaler: diabetes_scaler.pkl")

    # 13) Chi-Square test on original test predictions
    y_hat = best_model.predict(X_te_s)
    contingency = pd.crosstab(y_te, y_hat)
    chi2, p, dof, ex = chi2_contingency(contingency)
    print("\nChi-Square Test (y_test vs predictions):")
    print(f"  Chi2  : {chi2:.4f}")
    print(f"  p-val : {p:.6f}")
    print(f"  dof   : {dof}")
    print("  Decision:", "Reject H0 (dependence)" if p < 0.05 else "Fail to Reject H0 (independence)")

    # 14) LGBM GridSearch (quick) + RandomizedSearch (richer) on original (STRATIFIED) split
    lgb = LGBMClassifier(random_state=RANDOM_STATE, class_weight=None)
    param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [5, 10],
        'learning_rate': [0.05, 0.1],
        'num_leaves': [31, 50]
    }
    gs = GridSearchCV(lgb, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1, verbose=1)
    gs.fit(X_tr, y_tr)
    print("✅ LGBM GridSearch Best Params:", gs.best_params_)
    print("✅ LGBM GridSearch Best CV Acc:", gs.best_score_)
    joblib.dump(gs.best_estimator_, 'diabetes_012_LGB_Tuned.pkl')

    # Randomized with class_weight=balanced
    lgb_bal = LGBMClassifier(class_weight='balanced', random_state=RANDOM_STATE)
    param_dist = {
        'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15],
        'n_estimators': [100, 150, 200, 250],
        'max_depth': [3, 5, 7, 10],
        'num_leaves': [20, 31, 40, 50, 70, 100],
        'min_child_samples': [10, 20, 30],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    rs = RandomizedSearchCV(
        estimator=lgb_bal,
        param_distributions=param_dist,
        n_iter=30,
        scoring='f1_macro',
        cv=3,
        verbose=1,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    rs.fit(X_tr, y_tr)
    joblib.dump(rs.best_estimator_, 'diabetes_012_LGB_Final_Tuned.pkl')
    print("✅ LGBM Randomized Best Params:", rs.best_params_)
    print("✅ LGBM Randomized Best F1-Macro (CV):", rs.best_score_)

    # Confusion matrix for the tuned LGBM (on test)
    y_pred_final = rs.best_estimator_.predict(X_te)
    cm_final = confusion_matrix(y_te, y_pred_final)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_final)
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    ax.set_title("Confusion Matrix for Final Tuned LGBM")
    plt.tight_layout()
    save_fig(fig, "cm_lgbm_final_tuned.png")

    # 15) Small sample prediction demo (3 rows from test)
    sample = X_te.sample(3, random_state=RANDOM_STATE)
    print("\n=== Sample Input (3 rows) ===")
    print(sample)
    et_best = trained_models["ExtraTrees"]
    predicted = et_best.predict(scaler.transform(sample))
    probs = et_best.predict_proba(scaler.transform(sample))
    print("\nPredicted Output:", predicted)
    print("\nPrediction Probabilities:\n", probs)


if __name__ == "__main__":
    main()