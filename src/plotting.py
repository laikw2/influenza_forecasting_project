from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay


def save_observed_predicted_plot(df: pd.DataFrame, path: Path, title: str) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_df = df.sort_values("ISO_WEEKSTARTDATE")
    ax.plot(plot_df["ISO_WEEKSTARTDATE"], plot_df["y_true"], label="Observed", linewidth=2)
    ax.plot(plot_df["ISO_WEEKSTARTDATE"], plot_df["y_pred"], label="Predicted", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Week")
    ax.set_ylabel("Value")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_residual_plot(df: pd.DataFrame, path: Path, title: str) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_df = df.sort_values("ISO_WEEKSTARTDATE").copy()
    plot_df["residual"] = plot_df["y_true"] - plot_df["y_pred"]
    ax.axhline(0, color="black", linewidth=1)
    ax.plot(plot_df["ISO_WEEKSTARTDATE"], plot_df["residual"], marker="o", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Week")
    ax.set_ylabel("Observed - predicted")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_calibration_plot(df: pd.DataFrame, path: Path, title: str) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(df["y_pred"], df["y_true"], alpha=0.7)
    low = min(df["y_pred"].min(), df["y_true"].min())
    high = max(df["y_pred"].max(), df["y_true"].max())
    ax.plot([low, high], [low, high], "--", color="gray")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_confusion_plot(cm_df: pd.DataFrame, path: Path, title: str) -> None:
    if cm_df.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(cm_df.values, display_labels=[c.replace("pred_", "") for c in cm_df.columns])
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_curve_plot(curve_csv: Path, path: Path, title: str) -> None:
    if not curve_csv.exists():
        return
    df = pd.read_csv(curve_csv)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    if {"fpr", "tpr"}.issubset(df.columns):
        ax.plot(df["fpr"], df["tpr"], linewidth=2)
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
    elif {"recall", "precision"}.issubset(df.columns):
        ax.plot(df["recall"], df["precision"], linewidth=2)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_loss_curve(history: pd.DataFrame, path: Path, title: str) -> None:
    if history.empty or not {"iteration", "train", "valid"}.issubset(history.columns):
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["iteration"], history["train"], label="Training")
    ax.plot(history["iteration"], history["valid"], label="Validation")
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss / metric")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
