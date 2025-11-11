import warnings
warnings.filterwarnings("ignore")

import os
from collections import Counter
from math import pi

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, precision_recall_curve,
    roc_auc_score, average_precision_score
)
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, AdaBoostClassifier
)
from xgboost import XGBClassifier, XGBRFClassifier
from lightgbm import LGBMClassifier

from imblearn.over_sampling import ADASYN
import joblib
from scipy.stats import chi2_contingency


# ============================== basic setup / "constants" ==============================

RANDOM_STATE = 42

# target column for the prediction task
TARGET = "Diabetes_012"

# local path to the BRFSS2015 diabetes indicators dataset
DATA_PATH = "C:/Users/91910/OneDrive/Desktop/FinalProject/diabetes_012_health_indicators_BRFSS2015.csv"

# all plots will be exported here so I can just drop them into the report
PLOTS_DIR = "f"

# plotting style tweaks:

sns.set_theme(style="whitegrid", context="talk")
mpl.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "axes.titlesize": 18,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.autolayout": False,
})

PALETTE = sns.color_palette("colorblind")
ACCENT = PALETTE[0]


def ensure_dir(path: str):
    """
    Small helper: create the folder if it's not there.
    I use this to avoid manual "mkdir" every time I rerun.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

ensure_dir(PLOTS_DIR)


def save_fig(fig, filename, outdir=PLOTS_DIR):
    """
    Centralised figure saver so I don't repeat dpi/tight/bbox everywhere.
    Also prints where it saved so I can quickly open it.
    """
    out_path = os.path.join(outdir, filename)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[plot saved] {out_path}")


def trim_outliers_zscore(X: pd.DataFrame, y: pd.Series, z_thresh: float = 3.0):
    """
    Very light outlier trimming using z-score.
    I'm not doing anything fancy here. The idea is just:
    if something is >3 std devs away for ALL numeric cols, I drop that row.
    This is mainly to calm down extreme BMI / BP style values.

    returns new_X, new_y (indexes reset)
    """
    num_cols = X.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        return X.reset_index(drop=True), y.reset_index(drop=True)

    z = (X[num_cols] - X[num_cols].mean()) / X[num_cols].std(ddof=0)
    mask = (z.abs() <= z_thresh).all(axis=1)

    return X.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True)


def _plot_confusion_matrix_pretty(cm, title):
    """
    Confusion Matrix with raw counts and row-wise %.
    This style is the same idea I've been using in code1,
    and I'm keeping it because it's actually readable in the report.

    cm: confusion_matrix(y_true, y_pred)
    title: figure title
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Count")

    n_classes = cm.shape[0]
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))

    row_sums = cm.sum(axis=1, keepdims=True)

    for i in range(n_classes):
        for j in range(n_classes):
            pct = (cm[i, j] / row_sums[i, 0]) * 100 if row_sums[i, 0] else 0.0
            ax.text(
                j, i,
                f"{cm[i, j]}\n({pct:.1f}%)",
                ha="center", va="center",
                color=("white" if im.norm(cm[i, j]) > 0.5 else "black"),
                fontsize=11, weight="bold"
            )
    plt.tight_layout()
    return fig


def _plot_confusion_matrix_normalized(y_true, y_pred, filename):
    """
    Same confusion matrix idea but normalised per true class (recall-style).
    I generate this in grayscale because in the screenshots it looked cleaner.
    """
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, cmap="Greys", vmin=0, vmax=1)

    ax.set_title("Normalized Confusion Matrix\n(Recall per True Class)", weight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(cm_norm.shape[1]))
    ax.set_yticks(range(cm_norm.shape[0]))
    ax.set_xticklabels(range(cm_norm.shape[1]))
    ax.set_yticklabels(range(cm_norm.shape[0]))

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            pct = cm_norm[i, j] * 100
            ax.text(
                j, i,
                f"{pct:.1f}%",
                ha="center", va="center",
                color="black" if pct < 60 else "white",
                fontsize=10, weight="bold"
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall (%)")

    plt.tight_layout()
    save_fig(fig, filename)


def evaluate_and_report(name, model, X_test_scaled, y_test):
    """
    Core evaluation step.
    I'm intentionally evaluating on the *original* (imbalanced) test set here,
    because that's the actual "real world" distribution.

    Returns a dict of metrics for later comparison and radar plotting.
    """
    y_pred = model.predict(X_test_scaled)

    acc  = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"\n--------------------------------------------------------------------------------")
    print(f"[FINAL TEST EVAL] {name}")
    print("Note: test set keeps the natural class imbalance, on purpose.")
    print("--------------------------------------------------------------------------------")
    print(f"Accuracy            : {acc:.4f}")
    print(f"Balanced Accuracy   : {bacc:.4f}")
    print(f"Macro Precision     : {prec:.4f}")
    print(f"Macro Recall        : {rec:.4f}")
    print(f"Macro F1            : {f1:.4f}")
    print("\nDetailed per-class metrics:\n")
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    print("True-class row totals in test set:", cm.sum(axis=1).tolist())

    # save confusion matrices
    fig_cm = _plot_confusion_matrix_pretty(
        cm, f"Confusion Matrix – {name} (Proper Eval)"
    )
    save_fig(fig_cm, f"cm_proper_{name}.png")

    _plot_confusion_matrix_normalized(
        y_true=y_test,
        y_pred=y_pred,
        filename=f"cm_normalized_{name}.png"
    )

    return {
        "acc": acc,
        "bacc": bacc,
        "prec_macro": prec,
        "rec_macro": rec,
        "f1_macro": f1,
        "cm": cm
    }


def diagnostic_balanced_cm_per_model(name, base_model, X_full, y_full):
    """
    Side experiment / sanity check.

    Here I'm forcing a balanced training scenario using ADASYN
    on the WHOLE dataset (not just train). Then I split again and see
    if the model can treat all classes more fairly.

    This is NOT the main evaluation we report in results,
    it's more like "let me just check if the minority class is even learnable".
    """
    scaler_diag = StandardScaler()
    X_scaled_all = scaler_diag.fit_transform(X_full)

    # Oversample with ADASYN to attack the class imbalance
    X_res, y_res = ADASYN(random_state=RANDOM_STATE, n_neighbors=5).fit_resample(
        X_scaled_all, y_full
    )

    Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(
        X_res, y_res, test_size=0.2, random_state=RANDOM_STATE, shuffle=True
    )

    # Recreate the same model class so I don't contaminate the main trained model
    model_new = type(base_model)(**base_model.get_params())
    model_new.fit(Xd_tr, yd_tr)

    yd_pred = model_new.predict(Xd_te)
    cm_diag = confusion_matrix(yd_te, yd_pred)

    print(f"\n[DIAGNOSTIC BALANCED CHECK] {name}")
    print("Row totals here should be ~equal because of ADASYN balancing:")
    print(cm_diag.sum(axis=1).tolist())

    fig = _plot_confusion_matrix_pretty(
        cm_diag,
        f"Confusion Matrix – {name} (Diagnostic Balanced)"
    )
    save_fig(fig, f"cm_diagnostic_balanced_{name}.png")

    _plot_confusion_matrix_normalized(
        y_true=yd_te,
        y_pred=yd_pred,
        filename=f"cm_diagnostic_balanced_normalized_{name}.png"
    )


# ============================== main code ==============================
def main():
    # ---------------------------------------------------------------------------------
    # 1) Load data
    # ---------------------------------------------------------------------------------
    df = pd.read_csv(DATA_PATH)

    # Target column: Diabetes_012 (0=no diabetes, 1=prediabetes, 2=diabetes)
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])

    # ---------------------------------------------------------------------------------
    # 2) Exploratory "what's actually going on?" (EDA)
    # ---------------------------------------------------------------------------------
    print("\n[EDA] Basic distribution checks and sanity plots...\n")

    # 2.1 Class distribution of Diabetes_012, with both counts and %
    fig = plt.figure(figsize=(7, 4))
    ax = plt.gca()
    bar = sns.countplot(x=y, palette="Blues_r", ax=ax)
    ax.set_title("     Diabetes Category Distribution", weight="bold")
    ax.set_xlabel("Diabetes_012 (0: No Diabetes, 1: Prediabetes, 2: Diabetes)")
    ax.set_ylabel("Count")
    total = len(y)
    for p in bar.patches:
        height = p.get_height()
        pct = (height / total) * 100 if total else 0
        ax.annotate(
            f"{height:,}\n({pct:.1f}%)",
            (p.get_x() + p.get_width()/2., height),
            ha="center", va="bottom", fontsize=11, weight="bold",
            xytext=(0, 5), textcoords="offset points"
        )
    plt.tight_layout()
    save_fig(fig, "diabetes_class_distribution.png")

    # 2.2 BMI distribution across diabetes classes (boxplot)
    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()
    sns.boxplot(x=y, y=df["BMI"], palette="pastel", ax=ax)
    ax.set_title("BMI Distribution by Diabetes Category", weight="bold")
    ax.set_xlabel("Diabetes_012")
    ax.set_ylabel("BMI")
    # manually annotate median for each group so it's easy to talk about in the write-up
    for i, grp in enumerate(sorted(df[TARGET].unique())):
        median_val = df.loc[df[TARGET] == grp, "BMI"].median()
        ax.text(
            i, median_val + 0.3,
            f"Median={median_val:.1f}",
            ha="center", va="bottom", fontsize=10, color="black"
        )
    plt.tight_layout()
    save_fig(fig, "bmi_boxplot_by_diabetes.png")

    # 2.3 BMI density (KDE) for each diabetes class, plus marking the approx peak
    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()
    class_labels_ordered = sorted(df[TARGET].unique())
    label_map = {0: "No Diabetes", 1: "Prediabetes", 2: "Diabetes"}
    for cls in class_labels_ordered:
        subset = df[df[TARGET] == cls]["BMI"].dropna()
        if len(subset) == 0:
            continue
        sns.kdeplot(subset, ax=ax, linewidth=2, label=label_map.get(cls, f"class {cls}"))

        # mark a representative mode-ish point, just to show where it's peaking
        peak = subset.mode().iloc[0]
        ax.axvline(peak, ls="--", alpha=0.4)
        ax.text(
            peak, 0.002,
            f"Peak≈{peak:.1f}",
            rotation=90,
            va="bottom",
            fontsize=9
        )
    ax.set_title("BMI Density by Diabetes Category", weight="bold")
    ax.set_xlabel("BMI")
    ax.legend(title="Group")
    plt.tight_layout()
    save_fig(fig, "bmi_density_by_diabetes.png")

    # 2.4 Diabetes rate by Age group (only looking at class==2 i.e. diagnosed diabetes)
    if "Age" in df.columns:
        age_stats = (
            df.groupby("Age")[TARGET]
              .value_counts(normalize=True)
              .rename("Rate")
              .reset_index()
        )
        age_stats = age_stats[age_stats[TARGET] == 2]  # keep only '2' (diabetes)

        fig = plt.figure(figsize=(9, 5))
        ax = plt.gca()
        sns.barplot(
            x="Age", y="Rate", data=age_stats,
            color="skyblue", edgecolor="black", linewidth=1, ax=ax
        )
        ax.set_title("Diabetes (2) Prevalence by Age Group", weight="bold")
        ax.set_xlabel("Age Category (1=18–24 ... 13=80+)")
        ax.set_ylabel("Diabetes Rate")
        for i, v in enumerate(age_stats["Rate"]):
            ax.text(
                i, v + 0.002,
                f"{v*100:.1f}%",
                ha="center", fontsize=9, weight="bold"
            )
        plt.tight_layout()
        save_fig(fig, "age_diabetes_prevalence.png")

    # 2.5 Self-reported general health vs diabetes rate
    if "GenHlth" in df.columns:
        hlth_stats = (
            df.groupby("GenHlth")[TARGET]
              .value_counts(normalize=True)
              .rename("Rate")
              .reset_index()
        )
        hlth_stats = hlth_stats[hlth_stats[TARGET] == 2]

        fig = plt.figure(figsize=(8, 5))
        ax = plt.gca()
        sns.barplot(
            x="GenHlth", y="Rate", data=hlth_stats,
            color="salmon", edgecolor="black", linewidth=1, ax=ax
        )
        ax.set_title("Diabetes Rate vs Self-Reported Health", weight="bold")
        ax.set_xlabel("General Health (1=Excellent ... 5=Poor)")
        ax.set_ylabel("Diabetes Rate")
        for i, v in enumerate(hlth_stats["Rate"]):
            ax.text(
                i, v + 0.002,
                f"{v*100:.1f}%",
                ha="center", fontsize=9, weight="bold"
            )
        plt.tight_layout()
        save_fig(fig, "general_health_vs_diabetes.png")

    # 2.6 full numeric correlation heatmap
    # (This is mainly here for the appendix / methodology section.)
    fig = plt.figure(figsize=(18, 13))
    ax = plt.gca()
    corr = df.corr(numeric_only=True)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        linewidths=0.2,
        cbar_kws={"label": "Correlation Coefficient"},
        vmin=-1,
        vmax=1,
        ax=ax,
        annot_kws={"size": 8}
    )
    ax.set_title("Correlation Heatmap of Diabetes Dataset", weight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_fig(fig, "correlation_heatmap_academic.png")

    # ---------------------------------------------------------------------------------
    # 3) Light outlier trimming
    # ---------------------------------------------------------------------------------
    X_trim, y_trim = trim_outliers_zscore(X, y, z_thresh=3.0)

    # ---------------------------------------------------------------------------------
    # 4) Train/test split (stratified) on the TRIMMED data
    # ---------------------------------------------------------------------------------
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_trim, y_trim,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_trim
    )
    print("\n[split] Class distribution after split:")
    print("Train:", Counter(y_tr))
    print("Test :", Counter(y_te))

    # ---------------------------------------------------------------------------------
    # 5) Scale features and balance ONLY the training set with ADASYN
    # ---------------------------------------------------------------------------------
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    X_tr_bal, y_tr_bal = ADASYN(
        random_state=RANDOM_STATE,
        n_neighbors=5
    ).fit_resample(X_tr_s, y_tr)

    print("\n[ADASYN] Class distribution AFTER balancing the training set:")
    print(Counter(y_tr_bal))

    # Instead of a boring single bar chart, I'm plotting "before vs after" horizontally.
    # (filename stays the same as older version to avoid breaking anything downstream)
    before_counts = pd.Series(y_tr).value_counts().sort_index()
    after_counts  = pd.Series(y_tr_bal).value_counts().sort_index()

    classes_sorted = sorted(before_counts.index.union(after_counts.index))
    before_vals = [before_counts.get(c, 0) for c in classes_sorted]
    after_vals  = [after_counts.get(c, 0)  for c in classes_sorted]

    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()
    y_pos = np.arange(len(classes_sorted))

    bar_before = ax.barh(
        y_pos - 0.2,
        before_vals,
        height=0.4,
        color="#cfe2f3",
        edgecolor="black",
        linewidth=0.7,
        label="Before"
    )
    bar_after = ax.barh(
        y_pos + 0.2,
        after_vals,
        height=0.4,
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.7,
        label="After (ADASYN)"
    )

    for rect in bar_before:
        w = rect.get_width()
        y_mid = rect.get_y() + rect.get_height()/2
        ax.text(w, y_mid, f"{int(w)}", va="center", ha="left", fontsize=10)

    for rect in bar_after:
        w = rect.get_width()
        y_mid = rect.get_y() + rect.get_height()/2
        ax.text(w, y_mid, f"{int(w)}", va="center", ha="left", fontsize=10, weight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([str(c) for c in classes_sorted])
    ax.set_xlabel("Number of samples")
    ax.set_ylabel("Diabetes Class")
    ax.set_title("Class Distribution Before vs After ADASYN", fontsize=24, weight="bold")
    ax.legend(frameon=True, loc="upper left")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    sns.despine()
    plt.tight_layout()
    save_fig(fig, "after_adasyn_balanced_classes.png")

    # ---------------------------------------------------------------------------------
    # 6) Define baseline models I want to compare
    # ---------------------------------------------------------------------------------
    models = {
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "RandomForest": RandomForestClassifier(
            max_depth=7, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
        "AdaBoost": AdaBoostClassifier(
            random_state=RANDOM_STATE
        ),
        "XGBClassifier": XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="mlogloss",
            use_label_encoder=False
        ),
        "XGBRFClassifier": XGBRFClassifier(
            random_state=RANDOM_STATE
        ),
        "LGBMClassifier": LGBMClassifier(
            n_estimators=150, random_state=RANDOM_STATE
        )
    }

    metrics_summary = {}
    trained_models = {}
    best_name, best_f1 = None, -1.0

    # ---------------------------------------------------------------------------------
    # 7) Train each model on the balanced training set, then evaluate on real test set
    # ---------------------------------------------------------------------------------
    for name, model in models.items():
        model.fit(X_tr_bal, y_tr_bal)
        trained_models[name] = model

        m = evaluate_and_report(name, model, X_te_s, y_te)
        metrics_summary[name] = m

        if m["f1_macro"] > best_f1:
            best_f1 = m["f1_macro"]
            best_name = name

    print("\n================ OVERALL COMPARISON (sorted by Macro F1) ================")
    for name, m in sorted(
        metrics_summary.items(),
        key=lambda kv: kv[1]["f1_macro"],
        reverse=True
    ):
        print(
            f"{name:18s} | "
            f"Acc {m['acc']:.4f} | "
            f"BAcc {m['bacc']:.4f} | "
            f"Prec {m['prec_macro']:.4f} | "
            f"Rec {m['rec_macro']:.4f} | "
            f"F1 {m['f1_macro']:.4f}"
        )

    # Create a table figure summarising the model metrics.
    results_df = pd.DataFrame([
        [
            name,
            metrics_summary[name]["acc"],
            metrics_summary[name]["bacc"],
            metrics_summary[name]["prec_macro"],
            metrics_summary[name]["rec_macro"],
            metrics_summary[name]["f1_macro"]
        ]
        for name in metrics_summary
    ], columns=[
        "Model", "Accuracy", "Balanced Accuracy",
        "Macro Precision", "Macro Recall", "Macro F1"
    ])

    results_df = results_df.round(4).sort_values("Macro F1", ascending=False)

    fig_table = plt.figure(figsize=(10, 2.5))
    ax_table = plt.gca()
    ax_table.axis("off")
    tbl = ax_table.table(
        cellText=results_df.values,
        colLabels=results_df.columns,
        loc="center",
        cellLoc="center"
    )
    for c in tbl.get_celld().values():
        c.set_edgecolor("black")
        c.set_linewidth(0.8)
    save_fig(fig_table, "final_model_table.png")

    # ---------------------------------------------------------------------------------
    # 8) Diagnostic balanced confusion matrices
    # ---------------------------------------------------------------------------------
    print("\n[diagnostic] Generating balanced confusion matrices per model...")
    for name, base_model in models.items():
        diagnostic_balanced_cm_per_model(name, base_model, X_trim, y_trim)

    # ---------------------------------------------------------------------------------
    # 9) Comparison bar chart of metrics for each model
    # ---------------------------------------------------------------------------------
    metrics_to_plot = [
        ("acc", "Accuracy"),
        ("bacc", "Balanced Accuracy"),
        ("prec_macro", "Precision (Macro)"),
        ("rec_macro", "Recall (Macro)"),
        ("f1_macro", "F1-Score (Macro)"),
    ]

    fig_metrics = plt.figure(figsize=(20, 10))
    ax_metrics = plt.gca()
    model_names_list = list(metrics_summary.keys())
    width = 0.12
    x = np.arange(len(model_names_list))

    for i, (k, label) in enumerate(metrics_to_plot):
        vals = [metrics_summary[n][k] for n in model_names_list]
        bars = ax_metrics.bar(x + i * width, vals, width, label=label)
        for b in bars:
            h = b.get_height()
            ax_metrics.annotate(
                f"{h:.3f}",
                (b.get_x() + b.get_width()/2., h),
                ha="center", va="bottom", fontsize=10,
                xytext=(0, 4), textcoords="offset points"
            )

    ax_metrics.set_xticks(x + width * 2)
    ax_metrics.set_xticklabels(model_names_list, rotation=30, ha="right")
    ax_metrics.set_title("Model Comparison Across Metrics")
    ax_metrics.set_ylabel("Score")
    ax_metrics.legend(loc="upper right", ncol=1, frameon=True)

    ax_metrics.set_ylim(
        0,
        max(
            max(v for v in [metrics_summary[n][k] for n in model_names_list])
            for k, _ in metrics_to_plot
        ) + 0.05
    )
    plt.tight_layout()
    save_fig(fig_metrics, "metrics_comparison.png")

    # ---------------------------------------------------------------------------------
    # 9a) Radar / spider chart of model performance
    # ---------------------------------------------------------------------------------
    radar_metrics = ["acc", "prec_macro", "rec_macro", "f1_macro"]
    pretty_labels = ["Accuracy", "Precision", "Recall", "F1"]

    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    fig_radar = plt.figure(figsize=(9, 8))
    axr = fig_radar.add_subplot(111, polar=True)
    axr.set_theta_offset(pi / 2)
    axr.set_theta_direction(-1)

    for model_name in metrics_summary.keys():
        vals = [metrics_summary[model_name][m] for m in radar_metrics]
        vals = np.concatenate([vals, [vals[0]]])
        axr.plot(angles, vals, linewidth=2, label=model_name)
        axr.fill(angles, vals, alpha=0.15)

    axr.set_xticks(angles[:-1])
    axr.set_xticklabels(pretty_labels, fontsize=12, weight="bold")
    axr.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axr.set_yticklabels(["0.5", "0.6", "0.7", "0.8", "0.9", "1.0"], fontsize=10)
    axr.set_title(
        "Model Performance Profile (Radar View)",
        fontsize=16,
        weight="bold",
        pad=20
    )
    axr.legend(
        loc="upper right",
        bbox_to_anchor=(1.3, 1.1),
        frameon=True,
        fontsize=9
    )
    plt.tight_layout()
    save_fig(fig_radar, "model_radar.png")

    # ---------------------------------------------------------------------------------
    # 10) Precision-Recall and ROC curves
    # ---------------------------------------------------------------------------------
    fig_pr = plt.figure(figsize=(12, 9))
    axpr = plt.gca()
    for name, model in trained_models.items():
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(X_te_s)
            y_true = pd.get_dummies(y_te).values
            precision, recall, _ = precision_recall_curve(
                y_true.ravel(), probas.ravel()
            )
            ap = average_precision_score(y_true, probas, average="micro")
            axpr.plot(
                recall, precision,
                linewidth=2,
                label=f"{name} (AP = {ap:.2f})"
            )
    axpr.grid(True, linestyle="--", alpha=0.4)
    axpr.set_title("Precision-Recall Curves")
    axpr.set_xlabel("Recall")
    axpr.set_ylabel("Precision")
    axpr.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    save_fig(fig_pr, "pr_curves.png")

    fig_roc = plt.figure(figsize=(12, 9))
    axroc = plt.gca()
    axroc.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="k")
    for name, model in trained_models.items():
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(X_te_s)
            y_true = pd.get_dummies(y_te).values
            fpr, tpr, _ = roc_curve(y_true.ravel(), probas.ravel())
            auc = roc_auc_score(
                y_true,
                probas,
                average="micro",
                multi_class="ovr"
            )
            axroc.plot(
                fpr, tpr,
                linewidth=2,
                label=f"{name} (AUC = {auc:.2f})"
            )
    axroc.grid(True, linestyle="--", alpha=0.4)
    axroc.set_title("ROC Curves")
    axroc.set_xlabel("False Positive Rate")
    axroc.set_ylabel("True Positive Rate")
    axroc.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    save_fig(fig_roc, "roc_curves.png")

    # 10a) Side-by-side ROC + PR panels (micro-average)
    fig_panel, (axroc2, axpr2) = plt.subplots(1, 2, figsize=(16, 6))

    y_true_bin = pd.get_dummies(y_te).values  # micro-average uses one-hot

    # ROC panel
    axroc2.plot(
        [0, 1], [0, 1],
        linestyle="--",
        linewidth=1,
        color="gray",
        label="Chance"
    )
    for name, model in trained_models.items():
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(X_te_s)
            fpr, tpr, _ = roc_curve(y_true_bin.ravel(), probas.ravel())
            auc_val = roc_auc_score(
                y_true_bin,
                probas,
                average="micro",
                multi_class="ovr"
            )
            axroc2.plot(
                fpr, tpr,
                linewidth=2,
                label=f"{name} (AUC={auc_val:.2f})"
            )
    axroc2.set_title("ROC (Micro-Average)", fontsize=16, weight="bold")
    axroc2.set_xlabel("False Positive Rate")
    axroc2.set_ylabel("True Positive Rate")
    axroc2.legend(loc="lower right", frameon=True, fontsize=9)
    axroc2.grid(True, linestyle="--", alpha=0.4)

    # PR panel
    for name, model in trained_models.items():
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(X_te_s)
            precision, recall, _ = precision_recall_curve(
                y_true_bin.ravel(), probas.ravel()
            )
            ap_val = average_precision_score(
                y_true_bin,
                probas,
                average="micro"
            )
            axpr2.plot(
                recall, precision,
                linewidth=2,
                label=f"{name} (AP={ap_val:.2f})"
            )
    axpr2.set_title(
        "Precision-Recall (Micro-Average)",
        fontsize=16,
        weight="bold"
    )
    axpr2.set_xlabel("Recall")
    axpr2.set_ylabel("Precision")
    axpr2.set_ylim(0.3, 1.0)
    axpr2.legend(loc="lower left", frameon=True, fontsize=9)
    axpr2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    save_fig(fig_panel, "roc_pr_panels.png")

    # ---------------------------------------------------------------------------------
    # 11) Feature importance (ExtraTrees)
    # ---------------------------------------------------------------------------------
    if "ExtraTrees" in trained_models:
        et = trained_models["ExtraTrees"]
        importances = et.feature_importances_

        if importances.shape[0] == X_trim.shape[1]:
            fi = (
                pd.DataFrame({
                    "Feature": X_trim.columns,
                    "Importance": importances
                })
                .sort_values("Importance", ascending=False)
            )

            print("\n[feature importance] Top 15 (ExtraTrees):\n")
            print(fi.head(15).to_string(index=False))

            top = fi.head(15)

            # classic horizontal bar chart (this one usually goes in the report body)
            fig_imp = plt.figure(figsize=(9, 7))
            ax_imp = plt.gca()
            ax_imp.barh(
                top["Feature"][::-1],
                top["Importance"][::-1]
            )
            ax_imp.set_title("Top 15 Feature Importances – ExtraTrees")
            ax_imp.set_xlabel("Importance")
            for i, v in enumerate(top["Importance"][::-1].values):
                ax_imp.text(v, i, f" {v:.3f}", va="center")
            plt.tight_layout()
            save_fig(fig_imp, "feature_importance_extratrees_top15.png")

            # more 'visual/demo' style radial plot (nice for presentation slides)
            polar_top_n = 12
            fi_top = fi.head(polar_top_n)
            angles = np.linspace(0, 2 * np.pi, len(fi_top), endpoint=False)
            radii = fi_top["Importance"].values
            labels = fi_top["Feature"].values

            fig_polar = plt.figure(figsize=(8, 8))
            axp = fig_polar.add_subplot(111, polar=True)
            bars = axp.bar(
                angles,
                radii,
                width=2 * np.pi / len(fi_top) * 0.8,
                bottom=0.0,
                edgecolor="black",
                linewidth=0.7,
                alpha=0.7,
                color="#708fbf"
            )
            axp.set_xticks(angles)
            axp.set_xticklabels(labels, fontsize=9)
            axp.set_title(
                "Top Feature Importances (Polar View)",
                fontsize=16,
                weight="bold"
            )
            axp.set_yticklabels([])

            for bar, r in zip(bars, radii):
                axp.text(
                    bar.get_x() + bar.get_width()/2,
                    r + max(radii)*0.05,
                    f"{r:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    weight="bold"
                )

            plt.tight_layout()
            save_fig(fig_polar, "feature_importance_polar.png")

    # ---------------------------------------------------------------------------------
    # 12) Save the best model + scaler for later inference
    # ---------------------------------------------------------------------------------
    best_model = trained_models[best_name]
    joblib.dump(best_model, f"diabetes_best_{best_name}.pkl")
    joblib.dump(scaler, "diabetes_scaler.pkl")
    print(f"\n[model export] saved best model as diabetes_best_{best_name}.pkl")
    print("[model export] saved scaler as diabetes_scaler.pkl")

    # ---------------------------------------------------------------------------------
    # 13) Chi-square test between actual y_test and predictions
    # ---------------------------------------------------------------------------------
    y_hat = best_model.predict(X_te_s)
    contingency = pd.crosstab(y_te, y_hat)
    chi2, p, dof, ex = chi2_contingency(contingency)
    print("\n[chi-square check] (y_test vs best_model predictions)")
    print(f"  Chi2  : {chi2:.4f}")
    print(f"  p-val : {p:.6f}")
    print(f"  dof   : {dof}")
    print("  Decision:",
          "Reject H0 (dependence)" if p < 0.05 else "Fail to Reject H0 (independence)")

    # ---------------------------------------------------------------------------------
    # 14) Hyperparameter tuning for LightGBM
    # I run both GridSearchCV (basic) and RandomizedSearchCV (wider).
    # ---------------------------------------------------------------------------------
    lgb = LGBMClassifier(random_state=RANDOM_STATE, class_weight=None)

    param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [5, 10],
        'learning_rate': [0.05, 0.1],
        'num_leaves': [31, 50]
    }

    gs = GridSearchCV(
        lgb,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    gs.fit(X_tr, y_tr)
    print("\n[LGBM GridSearch] Best Params:", gs.best_params_)
    print("[LGBM GridSearch] Best CV Acc:", gs.best_score_)
    joblib.dump(gs.best_estimator_, 'diabetes_012_LGB_Tuned.pkl')

    # RandomizedSearch with class_weight='balanced'
    lgb_bal = LGBMClassifier(
        class_weight='balanced',
        random_state=RANDOM_STATE
    )

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

    print("\n[LGBM RandomizedSearch] Best Params:", rs.best_params_)
    print("[LGBM RandomizedSearch] Best F1-Macro (CV):", rs.best_score_)

    # Confusion matrix for the tuned LGBM model
    y_pred_final = rs.best_estimator_.predict(X_te)
    cm_final = confusion_matrix(y_te, y_pred_final)

    fig_cm_final = _plot_confusion_matrix_pretty(
        cm_final,
        "Confusion Matrix for Final Tuned LGBM"
    )
    save_fig(fig_cm_final, "cm_lgbm_final_tuned.png")

    _plot_confusion_matrix_normalized(
        y_true=y_te,
        y_pred=y_pred_final,
        filename="cm_lgbm_final_tuned_normalized.png"
    )

    # ---------------------------------------------------------------------------------
    # 15) Tiny manual "demo prediction" on 3 random samples
    # ---------------------------------------------------------------------------------
    sample = X_te.sample(3, random_state=RANDOM_STATE)
    print("\n=== Manual sanity-check prediction on 3 held-out rows ===")
    print("Input rows:")
    print(sample)

    et_best = trained_models["ExtraTrees"]
    predicted = et_best.predict(scaler.transform(sample))
    probs = et_best.predict_proba(scaler.transform(sample))
    print("\nModel predicted classes:", predicted)
    print("\nClass probabilities matrix:\n", probs)


if __name__ == "__main__":
    main()
