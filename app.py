from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
REGRESSION_PLOTS_DIR = REPORTS_DIR / "plots" / "regression"

APP_TITLE = "Regional Transfer Learning Influenza Forecasting Dashboard"
APP_SUBTITLE = (
    "Numerical forecasting and cautious increase-alert prediction for UAE and Malaysia"
)

COUNTRY_CONFIG = {
    "UAE": {
        "code": "ARE",
        "scenario": "transfer_uae",
        "confidence": "Moderate",
        "classification_model": "random_forest",
        "classification_label": "Random Forest",
        "regression_models": {
            "target_next_INF_ALL": "random_forest",
            "target_next_positivity_rate": "random_forest",
            "target_next_INF_A": "random_forest",
            "target_next_INF_B": "random_forest",
        },
    },
    "Malaysia": {
        "code": "MYS",
        "scenario": "transfer_malaysia",
        "confidence": "Low-to-moderate",
        "classification_model": "lightgbm",
        "classification_label": "LightGBM",
        "regression_models": {
            "target_next_INF_ALL": "elastic_net",
            "target_next_positivity_rate": "lightgbm",
            "target_next_INF_A": "xgboost",
            "target_next_INF_B": "elastic_net",
        },
    },
}

TARGET_LABELS = {
    "target_next_INF_ALL": "INF_ALL",
    "target_next_positivity_rate": "Positivity rate",
    "target_next_INF_A": "INF_A",
    "target_next_INF_B": "INF_B",
    "target_increase_binary": "Increase alert",
}

TARGET_EXPLANATIONS = {
    "INF_ALL": "Total laboratory-confirmed influenza positives, A plus B.",
    "INF_A": "Laboratory-confirmed influenza A positives.",
    "INF_B": "Laboratory-confirmed influenza B positives.",
    "Positivity rate": "Influenza positives divided by specimens processed when specimen count is positive.",
}


st.set_page_config(
    page_title="Regional Influenza Forecasting",
    page_icon="",
    layout="wide",
)


st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    .metric-card {
        border: 1px solid #d9e2ec;
        border-radius: 8px;
        padding: 1rem;
        background: #ffffff;
        min-height: 132px;
    }
    .metric-card .label {
        color: #52606d;
        font-size: 0.85rem;
        margin-bottom: 0.4rem;
    }
    .metric-card .value {
        color: #102a43;
        font-size: 1.55rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .metric-card .note {
        color: #627d98;
        font-size: 0.82rem;
        margin-top: 0.45rem;
    }
    .status-box {
        border-left: 5px solid #2f80ed;
        background: #f5f9ff;
        padding: 0.85rem 1rem;
        border-radius: 6px;
    }
    .warning-box {
        border-left: 5px solid #c05621;
        background: #fffaf0;
        padding: 0.85rem 1rem;
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_csv(relative_path: str) -> pd.DataFrame:
    path = PROJECT_ROOT / relative_path
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def existing_file(relative_path: str) -> bool:
    return (PROJECT_ROOT / relative_path).exists()


@st.cache_resource(show_spinner=False)
def load_model_if_available(relative_path: str):
    path = PROJECT_ROOT / relative_path
    if not path.exists():
        return None
    try:
        import joblib

        return joblib.load(path)
    except Exception:
        return None


def metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="label">{label}</div>
          <div class="value">{value}</div>
          <div class="note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_number(value, target: str) -> str:
    if pd.isna(value):
        return "Unavailable"
    if target == "target_next_positivity_rate":
        return f"{float(value):.1%}"
    return f"{max(float(value), 0):,.0f}"


def format_metric(value, digits: int = 3) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def selected_forecast_rows(country: str) -> pd.DataFrame:
    cfg = COUNTRY_CONFIG[country]
    future = load_csv("reports/next_week_forecasts.csv")
    if future.empty:
        return pd.DataFrame()

    rows = []
    for target, model in cfg["regression_models"].items():
        match = future[
            (future["scenario"] == cfg["scenario"])
            & (future["COUNTRY_CODE"] == cfg["code"])
            & (future["task"] == "regression")
            & (future["target"] == target)
            & (future["model"] == model)
        ]
        if not match.empty:
            rows.append(match.iloc[-1])

    alert = future[
        (future["scenario"] == cfg["scenario"])
        & (future["COUNTRY_CODE"] == cfg["code"])
        & (future["task"] == "classification")
        & (future["target"] == "target_increase_binary")
        & (future["model"] == cfg["classification_model"])
    ]
    if not alert.empty:
        rows.append(alert.iloc[-1])
    return pd.DataFrame(rows)


def selected_metrics(country: str) -> pd.DataFrame:
    cfg = COUNTRY_CONFIG[country]
    metrics = load_csv("reports/model_comparison_metrics.csv")
    if metrics.empty:
        return pd.DataFrame()

    filters = []
    for target, model in cfg["regression_models"].items():
        filters.append(
            (metrics["scenario"] == cfg["scenario"])
            & (metrics["task"] == "regression")
            & (metrics["target"] == target)
            & (metrics["model"] == model)
        )
    filters.append(
        (metrics["scenario"] == cfg["scenario"])
        & (metrics["task"] == "classification")
        & (metrics["target"] == "target_increase_binary")
        & (metrics["model"] == cfg["classification_model"])
    )

    mask = filters[0]
    for item in filters[1:]:
        mask = mask | item
    out = metrics[mask].copy()
    out["display_target"] = out["target"].map(TARGET_LABELS).fillna(out["target"])
    return out


def model_file_status(country: str) -> pd.DataFrame:
    cfg = COUNTRY_CONFIG[country]
    rows = []
    for target, model in cfg["regression_models"].items():
        rel = f"models/{cfg['scenario']}_regression_{target}_{model}.joblib"
        rows.append(
            {
                "task": "Regression",
                "target": TARGET_LABELS[target],
                "model": model,
                "file": rel,
                "available": existing_file(rel),
            }
        )
    rel = (
        f"models/{cfg['scenario']}_classification_target_increase_binary_"
        f"{cfg['classification_model']}.joblib"
    )
    rows.append(
        {
            "task": "Classification",
            "target": "Increase Alert",
            "model": cfg["classification_model"],
            "file": rel,
            "available": existing_file(rel),
        }
    )
    return pd.DataFrame(rows)


def plot_confusion_matrix(csv_path: Path, title: str) -> None:
    if not csv_path.exists():
        st.warning(f"Missing confusion matrix: {csv_path.relative_to(PROJECT_ROOT)}")
        return
    matrix = pd.read_csv(csv_path, index_col=0)
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    image = ax.imshow(matrix.values, cmap="Blues")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix.iloc[i, j]), ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig)


def plot_curve(csv_path: Path, x_col: str, y_col: str, title: str, x_label: str, y_label: str) -> None:
    if not csv_path.exists():
        st.warning(f"Missing curve data: {csv_path.relative_to(PROJECT_ROOT)}")
        return
    curve = pd.read_csv(csv_path)
    if x_col not in curve.columns or y_col not in curve.columns:
        st.warning(f"Curve file has unexpected columns: {csv_path.name}")
        return
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.plot(curve[x_col], curve[y_col], linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    st.pyplot(fig)


def show_regression_plots(country: str) -> None:
    cfg = COUNTRY_CONFIG[country]
    for target, model in cfg["regression_models"].items():
        plot_path = (
            REGRESSION_PLOTS_DIR
            / f"{cfg['scenario']}_{target}_{model}_observed_predicted.png"
        )
        label = TARGET_LABELS[target]
        st.subheader(label)
        if plot_path.exists():
            st.image(str(plot_path), width="stretch")
        else:
            st.warning(f"Missing observed-vs-predicted plot: {plot_path.name}")


def show_alert_diagnostics(country: str) -> None:
    cfg = COUNTRY_CONFIG[country]
    prefix = f"{cfg['scenario']}_target_increase_binary_{cfg['classification_model']}"
    metric_rows = selected_metrics(country)
    metric_rows = metric_rows[metric_rows["task"] == "classification"]
    if not metric_rows.empty:
        cols = st.columns(5)
        row = metric_rows.iloc[0]
        cols[0].metric("Accuracy", format_metric(row.get("accuracy")))
        cols[1].metric("Balanced accuracy", format_metric(row.get("balanced_accuracy")))
        cols[2].metric("Macro F1", format_metric(row.get("macro_f1")))
        cols[3].metric("ROC-AUC", format_metric(row.get("roc_auc")))
        cols[4].metric("PR-AUC", format_metric(row.get("pr_auc")))
        st.caption(f"Holdout weeks: {int(row['n']) if not pd.isna(row.get('n')) else 'unavailable'}")

    left, middle, right = st.columns(3)
    with left:
        plot_confusion_matrix(
            REPORTS_DIR / f"{prefix}_confusion_matrix.csv",
            f"{country} Increase Alert Confusion Matrix",
        )
    with middle:
        plot_curve(
            REPORTS_DIR / f"{prefix}_roc_curve.csv",
            "fpr",
            "tpr",
            f"{country} ROC Curve",
            "False positive rate",
            "True positive rate",
        )
    with right:
        plot_curve(
            REPORTS_DIR / f"{prefix}_pr_curve.csv",
            "recall",
            "precision",
            f"{country} PR Curve",
            "Recall",
            "Precision",
        )


st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

st.markdown(
    """
    <div class="status-box">
    Regional transfer models provide useful numerical influenza forecasts for UAE and Malaysia.
    Increase-alert classification is available, but should be interpreted as low-to-moderate
    confidence only.
    </div>
    """,
    unsafe_allow_html=True,
)

tab_overview, tab_forecast, tab_performance, tab_plots, tab_classification, tab_limits, tab_upload = st.tabs(
    [
        "Overview",
        "Forecast Dashboard",
        "Historical Performance",
        "Observed vs Predicted",
        "Classification Diagnostics",
        "Interpretation / Limitations",
        "Upload Future Data",
    ]
)

with tab_overview:
    st.header("Overview")
    st.write(
        "This dashboard uses WHO FluNet laboratory surveillance outputs from the trained "
        "influenza forecasting project. The final demo focuses on country-specific regional "
        "transfer models for UAE and Malaysia."
    )
    st.write(
        "Regression forecasts estimate next-week numerical influenza activity. The alert "
        "classifier predicts only whether influenza activity is likely to increase versus "
        "not increase."
    )
    st.markdown(
        """
        - `INF_ALL`: total influenza-positive laboratory detections, A plus B.
        - `INF_A`: influenza A positives.
        - `INF_B`: influenza B positives.
        - `positivity_rate`: total influenza positives divided by specimens processed.
        """
    )
    st.warning(
        "Research prototype only. FluNet is laboratory surveillance data, not complete "
        "national incidence. This dashboard is not a public health decision system."
    )

    st.subheader("Recommended Models")
    overview_rows = []
    for country, cfg in COUNTRY_CONFIG.items():
        for target, model in cfg["regression_models"].items():
            overview_rows.append(
                {
                    "country": country,
                    "scenario": cfg["scenario"],
                    "task": "Numerical forecast",
                    "target": TARGET_LABELS[target],
                    "model": model,
                    "confidence": "Stronger than alert classification",
                }
            )
        overview_rows.append(
            {
                "country": country,
                "scenario": cfg["scenario"],
                "task": "Increase Alert",
                "target": "increase vs not increase",
                "model": cfg["classification_model"],
                "confidence": cfg["confidence"],
            }
        )
    st.dataframe(pd.DataFrame(overview_rows), width="stretch", hide_index=True)

with tab_forecast:
    st.header("Forecast Dashboard")
    country = st.selectbox("Select country", list(COUNTRY_CONFIG.keys()), key="forecast_country")
    cfg = COUNTRY_CONFIG[country]
    forecast_rows = selected_forecast_rows(country)

    if forecast_rows.empty:
        st.warning("Missing forecast rows in reports/next_week_forecasts.csv.")
    else:
        latest_week = forecast_rows["latest_observed_week"].dropna().iloc[0]
        forecast_week = forecast_rows["forecast_week"].dropna().iloc[0]
        st.write(f"Latest available week: `{latest_week}`")
        st.write(f"Forecast week: `{forecast_week}`")

        cols = st.columns(4)
        for idx, target in enumerate(cfg["regression_models"]):
            row = forecast_rows[forecast_rows["target"] == target]
            if row.empty:
                cols[idx].warning(f"Missing {TARGET_LABELS[target]}")
                continue
            row = row.iloc[0]
            cols[idx].markdown(
                "",
            )
            with cols[idx]:
                label = TARGET_LABELS[target]
                note = f"Model: {row['model']}"
                if label in TARGET_EXPLANATIONS:
                    note = f"{note}. {TARGET_EXPLANATIONS[label]}"
                metric_card(label, format_number(row["prediction"], target), note)

        st.subheader("Increase Alert")
        alert = forecast_rows[forecast_rows["target"] == "target_increase_binary"]
        if alert.empty:
            st.warning("Missing increase-alert forecast row.")
        else:
            row = alert.iloc[0]
            prob_inc = row.get("prob_increase")
            prob_not = row.get("prob_not_increase")
            prediction = str(row.get("prediction", "unavailable")).replace("_", " ").title()
            cols = st.columns(4)
            with cols[0]:
                metric_card("Alert", prediction, "Increase vs not increase")
            with cols[1]:
                metric_card(
                    "Probability of increase",
                    "Unavailable" if pd.isna(prob_inc) else f"{float(prob_inc):.1%}",
                    f"Model: {cfg['classification_label']}",
                )
            with cols[2]:
                metric_card(
                    "Probability of not increase",
                    "Unavailable" if pd.isna(prob_not) else f"{float(prob_not):.1%}",
                    "Classification probability, not a case-count interval",
                )
            with cols[3]:
                metric_card("Confidence label", cfg["confidence"], "Use cautiously")

    st.subheader("Saved Model Availability")
    status = model_file_status(country)
    st.dataframe(status, width="stretch", hide_index=True)
    missing = status[~status["available"]]
    if not missing.empty:
        st.warning("Some saved model files are missing. The app can still show saved forecasts and metrics.")

with tab_performance:
    st.header("Historical Performance")
    country = st.selectbox("Select country", list(COUNTRY_CONFIG.keys()), key="performance_country")
    metrics = selected_metrics(country)

    if metrics.empty:
        st.warning("Missing selected model metrics.")
    else:
        regression_cols = ["display_target", "model", "n", "mae", "rmse", "r2"]
        classification_cols = [
            "display_target",
            "model",
            "n",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "roc_auc",
            "pr_auc",
        ]
        reg = metrics[metrics["task"] == "regression"][regression_cols].copy()
        cls = metrics[metrics["task"] == "classification"][classification_cols].copy()

        st.subheader("Regression Metrics")
        st.dataframe(reg, width="stretch", hide_index=True)
        chart_reg = reg.set_index("display_target")[["rmse", "mae"]]
        st.bar_chart(chart_reg)
        st.caption("Lower RMSE and MAE are better. Higher R2 is better.")

        st.subheader("Increase-Alert Classification Metrics")
        st.dataframe(cls, width="stretch", hide_index=True)
        if not cls.empty:
            chart_cls = cls.set_index("display_target")[
                ["accuracy", "balanced_accuracy", "macro_f1", "roc_auc", "pr_auc"]
            ]
            st.bar_chart(chart_cls)

    with st.expander("Supplementary exploratory pooled baseline: Joint UAE + Malaysia"):
        all_metrics = load_csv("reports/model_comparison_metrics.csv")
        if all_metrics.empty:
            st.info("No model comparison metrics found.")
        else:
            joint = all_metrics[
                (all_metrics["scenario"] == "joint_uae_malaysia")
                & (all_metrics["target"].isin(list(TARGET_LABELS.keys())))
            ].copy()
            if joint.empty:
                st.info("No joint UAE + Malaysia rows found.")
            else:
                st.write(
                    "These pooled results are exploratory comparison only. They are not "
                    "the recommended final app models."
                )
                st.dataframe(joint, width="stretch", hide_index=True)

with tab_plots:
    st.header("Observed vs Predicted")
    country = st.selectbox("Select country", list(COUNTRY_CONFIG.keys()), key="plot_country")
    st.write("These plots come from the saved holdout validation figures.")
    show_regression_plots(country)

with tab_classification:
    st.header("Classification Diagnostics")
    country = st.selectbox("Select country", list(COUNTRY_CONFIG.keys()), key="classification_country")
    st.write(
        "The increase-alert classifier is useful for a simple warning layer, but numerical "
        "regression is the stronger part of the project."
    )
    show_alert_diagnostics(country)

with tab_limits:
    st.header("Interpretation / Limitations")
    st.markdown(
        """
        - The preferred models are transfer models because UAE and Malaysia each use a
          regional neighboring-country training pool before country-specific evaluation.
        - UAE and Malaysia are geographically and epidemiologically different, so direct
          joint pooling is less scientifically justified as the main model.
        - Numerical regression forecasting is stronger than classification in this project.
        - Increase-alert classification should be interpreted cautiously. For Malaysia,
          the selected LightGBM alert model has low-to-moderate validation strength:
          accuracy 0.702, balanced accuracy 0.542, macro F1 0.502, ROC-AUC 0.641,
          PR-AUC 0.480, based on 235 holdout weeks.
        - FluNet laboratory reports may reflect changes in specimen volume, laboratory
          participation, reporting source, and surveillance practice, not only true
          disease activity.
        - External prospective validation with future WHO FluNet updates is required
          before any operational use.
        """
    )

with tab_upload:
    st.header("Upload Future WHO FluNet Data")
    uploaded = st.file_uploader("Upload a newly downloaded WHO FluNet CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload is optional. The dashboard currently uses saved predictions from the completed experiment.")
    else:
        preview = pd.read_csv(uploaded, nrows=20)
        st.write("Uploaded file preview")
        st.dataframe(preview, width="stretch")
        st.warning("Live prediction requires the original feature-engineering pipeline.")
        st.write(
            "This app does not attempt live prediction from uploaded raw data unless the "
            "same preprocessing and feature-engineering pipeline is safely connected."
        )
