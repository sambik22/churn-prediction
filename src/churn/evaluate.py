"""Metrics, business-value threshold selection and quality plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from churn import config


def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "threshold": threshold,
    }


def expected_profit(
    y_true,
    proba,
    threshold: float,
    customer_value: float = config.CUSTOMER_VALUE,
    offer_cost: float = config.OFFER_COST,
    success_rate: float = config.OFFER_SUCCESS_RATE,
) -> float:
    """Expected profit of a retention campaign at the given threshold.

    Everyone with probability ≥ threshold gets the offer.
    * TP (really leaving): + success_rate * customer_value - offer_cost
    * FP (not leaving): - offer_cost (a discount for someone who would have stayed anyway)
    * FN / TN: 0 (we did nothing)
    """
    pred = proba >= threshold
    y = np.asarray(y_true).astype(bool)
    tp = np.sum(pred & y)
    fp = np.sum(pred & ~y)
    return float(tp * (success_rate * customer_value - offer_cost) - fp * offer_cost)


def best_profit_threshold(y_true, proba) -> tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 91)
    profits = [expected_profit(y_true, proba, t) for t in thresholds]
    i = int(np.argmax(profits))
    return float(thresholds[i]), float(profits[i])


def plot_model_report(y_true, proba, threshold: float, name: str, out_dir: Path) -> None:
    """ROC, Precision-Recall, calibration, confusion matrix and profit curve."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    fpr, tpr, _ = roc_curve(y_true, proba)
    ax = axes[0, 0]
    ax.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y_true, proba):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set(title="ROC curve", xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax.legend(loc="lower right")

    prec, rec, _ = precision_recall_curve(y_true, proba)
    ax = axes[0, 1]
    ax.plot(rec, prec, label=f"PR-AUC = {average_precision_score(y_true, proba):.3f}")
    ax.axhline(np.mean(y_true), ls="--", color="grey", lw=1, label="Random model")
    ax.set(title="Precision-Recall", xlabel="Recall", ylabel="Precision")
    ax.legend(loc="upper right")

    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
    ax = axes[1, 0]
    ax.plot(mean_pred, frac_pos, "o-", label=name)
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Perfect calibration")
    ax.set(title="Calibration", xlabel="Predicted probability", ylabel="Observed fraction")
    ax.legend(loc="upper left")

    cm = confusion_matrix(y_true, (proba >= threshold).astype(int))
    ConfusionMatrixDisplay(cm, display_labels=["Stayed", "Churned"]).plot(
        ax=axes[1, 1], colorbar=False, cmap="Blues"
    )
    axes[1, 1].set_title(f"Confusion matrix (threshold = {threshold:.2f})")

    fig.suptitle(f"Model '{name}' on the test set", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "model_report.png", dpi=120)
    plt.close(fig)

    thresholds = np.linspace(0.05, 0.95, 91)
    profits = [expected_profit(y_true, proba, t) for t in thresholds]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(thresholds, profits)
    ax.axvline(threshold, ls="--", color="C3", label=f"Chosen threshold {threshold:.2f}")
    ax.axvline(0.5, ls=":", color="grey", label="Default threshold 0.5")
    ax.axhline(0, color="black", lw=0.8)
    ax.set(
        title="Expected retention campaign profit vs threshold (test)",
        xlabel="Probability threshold",
        ylabel="Profit, $",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "profit_curve.png", dpi=120)
    plt.close(fig)


def plot_model_comparison(cv_results: dict[str, dict[str, float]], out_dir: Path) -> None:
    names = list(cv_results)
    roc = [cv_results[n]["roc_auc_mean"] for n in names]
    roc_err = [cv_results[n]["roc_auc_std"] for n in names]
    pr = [cv_results[n]["pr_auc_mean"] for n in names]
    pr_err = [cv_results[n]["pr_auc_std"] for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 0.2, roc, 0.4, yerr=roc_err, label="ROC-AUC", capsize=3)
    ax.bar(x + 0.2, pr, 0.4, yerr=pr_err, label="PR-AUC", capsize=3)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1)
    ax.set_title(f"Model comparison: {config.CV_FOLDS}-fold cross-validation on train")
    ax.legend()
    for i, v in enumerate(roc):
        ax.text(i - 0.2, v + 0.03, f"{v:.3f}", ha="center", fontsize=8)
    for i, v in enumerate(pr):
        ax.text(i + 0.2, v + 0.03, f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "model_comparison.png", dpi=120)
    plt.close(fig)
