import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(2 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]))


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return {"n": int(mask.sum()), "mae": np.nan, "rmse": np.nan, "smape": np.nan, "r2": np.nan, "spearman": np.nan}
    corr = spearmanr(y_true[mask], y_pred[mask], nan_policy="omit")
    return {
        "n": int(mask.sum()),
        "mae": float(mean_absolute_error(y_true[mask], y_pred[mask])),
        "rmse": float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))),
        "mape": mape(y_true[mask], y_pred[mask]),
        "smape": smape(y_true[mask], y_pred[mask]),
        "r2": float(r2_score(y_true[mask], y_pred[mask])) if mask.sum() > 1 else np.nan,
        "spearman": float(corr.statistic) if np.isfinite(corr.statistic) else np.nan,
    }


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_true) > 1e-12)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def add_interval_metrics(metrics: dict, y_true, y_pred, residual_std: float) -> dict:
    if not np.isfinite(residual_std) or residual_std <= 0:
        metrics["prediction_interval_95_coverage"] = np.nan
        metrics["prediction_interval_80_coverage"] = np.nan
        return metrics
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        metrics["prediction_interval_95_coverage"] = np.nan
        metrics["prediction_interval_80_coverage"] = np.nan
        return metrics
    lower95 = y_pred[mask] - 1.96 * residual_std
    upper95 = y_pred[mask] + 1.96 * residual_std
    lower80 = y_pred[mask] - 1.28 * residual_std
    upper80 = y_pred[mask] + 1.28 * residual_std
    metrics["prediction_interval_95_coverage"] = float(np.mean((y_true[mask] >= lower95) & (y_true[mask] <= upper95)))
    metrics["prediction_interval_80_coverage"] = float(np.mean((y_true[mask] >= lower80) & (y_true[mask] <= upper80)))
    return metrics


def classification_metrics(y_true, y_pred, y_proba=None, classes=None) -> dict:
    y_true = pd.Series(y_true).astype(str)
    y_pred = pd.Series(y_pred).astype(str)
    labels = classes or sorted(set(y_true.dropna()) | set(y_pred.dropna()))
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
    }
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    for label, p, r, f, s in zip(labels, prec, rec, f1, support):
        out[f"precision_{label}"] = float(p)
        out[f"recall_{label}"] = float(r)
        out[f"f1_{label}"] = float(f)
        out[f"support_{label}"] = int(s)
    if y_proba is not None and classes is not None:
        out.update(probability_metrics(y_true, y_proba, classes))
    return out


def probability_metrics(y_true, y_proba, classes) -> dict:
    out = {}
    y_true = pd.Series(y_true).astype(str)
    y_proba = np.asarray(y_proba)
    if len(classes) == 2 and y_proba.ndim == 2:
        pos_index = list(classes).index("increase") if "increase" in classes else 1
        binary_true = (y_true == classes[pos_index]).astype(int)
        if binary_true.nunique() == 2:
            out["roc_auc"] = float(roc_auc_score(binary_true, y_proba[:, pos_index]))
            out["pr_auc"] = float(average_precision_score(binary_true, y_proba[:, pos_index]))
    elif y_proba.ndim == 2 and len(classes) > 2:
        present = [c for c in classes if c in set(y_true)]
        if len(present) > 1:
            y_bin = label_binarize(y_true, classes=classes)
            try:
                out["roc_auc_ovr_macro"] = float(roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr"))
            except ValueError:
                out["roc_auc_ovr_macro"] = np.nan
            pr_scores = []
            for i, c in enumerate(classes):
                if y_bin[:, i].sum() > 0:
                    pr_scores.append(average_precision_score(y_bin[:, i], y_proba[:, i]))
            out["pr_auc_ovr_macro"] = float(np.mean(pr_scores)) if pr_scores else np.nan
    return out


def save_confusion_matrix(y_true, y_pred, labels, path: Path) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{x}" for x in labels], columns=[f"pred_{x}" for x in labels])
    cm_df.to_csv(path)
    return cm_df


def save_curve_data(y_true, y_proba, classes, out_prefix: Path) -> None:
    y_true = pd.Series(y_true).astype(str)
    y_proba = np.asarray(y_proba)
    rows = []
    if len(classes) == 2 and y_proba.ndim == 2:
        pos_index = list(classes).index("increase") if "increase" in classes else 1
        binary_true = (y_true == classes[pos_index]).astype(int)
        if binary_true.nunique() == 2:
            fpr, tpr, _ = roc_curve(binary_true, y_proba[:, pos_index])
            pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(out_prefix.with_name(out_prefix.name + "_roc_curve.csv"), index=False)
            precision, recall, _ = precision_recall_curve(binary_true, y_proba[:, pos_index])
            pd.DataFrame({"precision": precision, "recall": recall}).to_csv(out_prefix.with_name(out_prefix.name + "_pr_curve.csv"), index=False)
    elif len(classes) > 2 and y_proba.ndim == 2:
        y_bin = label_binarize(y_true, classes=classes)
        for i, c in enumerate(classes):
            if y_bin[:, i].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            for x, y in zip(fpr, tpr):
                rows.append({"class": c, "curve": "roc", "x": x, "y": y})
            precision, recall, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
            for x, y in zip(recall, precision):
                rows.append({"class": c, "curve": "pr", "x": x, "y": y})
        if rows:
            pd.DataFrame(rows).to_csv(out_prefix.with_name(out_prefix.name + "_ovr_curves.csv"), index=False)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
