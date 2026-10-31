from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

FEATURES = [
    "Food_type",
    "Storage_condition",
    "Packaging",
    "Temperature",
    "Humidity",
]
REQUIRED_COLUMNS = [*FEATURES, "Expiry_days"]
TARGET = "Expiry_category"

st.set_page_config(
    page_title="Food Expiry Predictor",
    page_icon="🥬",
    layout="wide",
)

st.title("🥬 Food Expiry Predictor")
st.markdown(
    "Estimate whether food has a **short**, **medium**, or **long** shelf life "
    "using a lightweight Bayesian-style predictor trained from your food-storage data."
)
st.info(
    "Predictions are estimates from the uploaded dataset, not food-safety advice. "
    "Always follow product labels and local food-safety guidance."
)


def make_demo_data(rows: int = 240, seed: int = 42) -> pd.DataFrame:
    """Create clearly labeled synthetic data so the app can be previewed."""
    rng = np.random.default_rng(seed)
    food_types = np.array(["Dairy", "Produce", "Meat", "Bakery", "Canned"])
    storage_options = np.array(["Refrigerated", "Frozen", "Pantry"])
    packaging_options = np.array(["Sealed", "Opened", "Vacuum-packed"])
    base_days = {"Dairy": 8, "Produce": 7, "Meat": 5, "Bakery": 6, "Canned": 40}
    storage_bonus = {"Refrigerated": 5, "Frozen": 20, "Pantry": 0}
    packaging_bonus = {"Sealed": 4, "Opened": -2, "Vacuum-packed": 7}

    records = []
    for _ in range(rows):
        food = str(rng.choice(food_types))
        storage = str(rng.choice(storage_options))
        packaging = str(rng.choice(packaging_options))
        temperature = float(rng.normal(4 if storage == "Refrigerated" else 0 if storage == "Frozen" else 22, 3))
        humidity = float(rng.uniform(25, 90))
        days = (
            base_days[food]
            + storage_bonus[storage]
            + packaging_bonus[packaging]
            - max(0, temperature - 5) * 0.45
            - max(0, humidity - 70) * 0.08
            + float(rng.normal(0, 3))
        )
        records.append(
            {
                "Food_type": food,
                "Storage_condition": storage,
                "Packaging": packaging,
                "Temperature": round(temperature, 1),
                "Humidity": round(humidity, 1),
                "Expiry_days": max(1, round(days)),
            }
        )
    return pd.DataFrame(records)


def discretize_numeric_column(series: pd.Series) -> tuple[pd.Series, list[float]]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return pd.Series([], dtype="object"), []
    q1 = float(values.quantile(0.33))
    q2 = float(values.quantile(0.66))
    thresholds = sorted({q1, q2})
    bins = pd.cut(
        series,
        bins=[-float("inf"), *thresholds, float("inf")],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )
    return bins.astype(str), thresholds


def prepare_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[float]]]:
    missing = [column for column in REQUIRED_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    clean = raw[REQUIRED_COLUMNS].copy()
    for column in ("Temperature", "Humidity", "Expiry_days"):
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    clean = clean.dropna(subset=REQUIRED_COLUMNS)
    if clean.empty:
        raise ValueError("No complete rows remain after removing missing or invalid values.")

    clean[TARGET] = pd.cut(
        clean["Expiry_days"],
        bins=[-float("inf"), 7, 15, float("inf")],
        labels=["Short", "Medium", "Long"],
        right=False,
    ).astype(str)

    cutpoints: dict[str, list[float]] = {}
    for column in ("Temperature", "Humidity"):
        clean[column], cutpoints[column] = discretize_numeric_column(clean[column])

    for column in ("Food_type", "Storage_condition", "Packaging"):
        clean[column] = clean[column].astype(str).str.strip()
    clean = clean[(clean["Food_type"] != "") & (clean["Storage_condition"] != "") & (clean["Packaging"] != "")]
    if clean[TARGET].nunique() < 2:
        raise ValueError("The dataset must contain expiry days from at least two categories.")
    return clean.reset_index(drop=True), cutpoints


def build_model(training_data: pd.DataFrame) -> dict[str, Any]:
    categories = list(training_data[TARGET].dropna().unique())
    prior = training_data[TARGET].value_counts(normalize=True).to_dict()
    condensed = {cat: float(prior.get(cat, 1 / len(categories))) for cat in categories}
    conditional = {}
    for feature in FEATURES:
        conditional[feature] = {}
        for value in sorted(training_data[feature].dropna().unique()):
            conditional[feature][value] = {}
            total = (training_data[feature] == value).sum()
            for category in categories:
                count = ((training_data[feature] == value) & (training_data[TARGET] == category)).sum()
                conditional[feature][value][category] = (count + 1) / (total + len(categories))
    return {"categories": categories, "prior": condensed, "conditional": conditional}


def predict_category(model: dict[str, Any], values: dict[str, str]) -> tuple[str, dict[str, float]]:
    categories = model["categories"]
    scores: dict[str, float] = {}
    for category in categories:
        score = math.log(model["prior"].get(category, 1 / len(categories)))
        for feature in FEATURES:
            value = str(values[feature])
            feature_prob = model["conditional"][feature].get(value, {}).get(category, 1 / len(categories))
            score += math.log(max(feature_prob, 1e-9))
        scores[category] = score

    max_score = max(scores.values())
    exp_scores = {k: math.exp(v - max_score) for k, v in scores.items()}
    total = sum(exp_scores.values())
    probabilities = {k: exp_scores[k] / total for k in categories}
    prediction = max(probabilities, key=probabilities.get)
    return prediction, probabilities


with st.sidebar:
    st.header("1. Choose your data")
    data_source = st.radio("Data source", ["Synthetic demo data", "Upload a CSV"])
    uploaded_file = st.file_uploader("Upload food_expiry_dataset.csv", type=["csv"]) if data_source == "Upload a CSV" else None

if data_source == "Upload a CSV":
    if uploaded_file is None:
        st.warning("Upload a CSV file to train the model. The required columns are listed below.")
        st.code(", ".join(REQUIRED_COLUMNS), language="text")
        st.stop()
    try:
        raw_data = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read the CSV: {exc}")
        st.stop()
else:
    raw_data = make_demo_data()
    st.caption("Demo mode uses generated synthetic data; it is not real food-expiry research data.")

try:
    data, cutpoints = prepare_data(raw_data)
    model = build_model(data)
except (ValueError, KeyError) as exc:
    st.error(str(exc))
    st.stop()

counts = data[TARGET].value_counts()
if len(data) < 10:
    st.warning("The dataset has fewer than 10 usable rows; predictions may be unreliable.")

st.header("2. Explore the training data")
metric_cols = st.columns(4)
metric_cols[0].metric("Usable rows", f"{len(data):,}")
metric_cols[1].metric("Food types", f"{data['Food_type'].nunique():,}")
metric_cols[2].metric("Storage conditions", f"{data['Storage_condition'].nunique():,}")
metric_cols[3].metric("Expiry categories", f"{data[TARGET].nunique():,}")

left, right = st.columns(2)
with left:
    st.subheader("Expiry category distribution")
    st.bar_chart(counts.rename_axis("Category"))
with right:
    st.subheader("Preview")
    st.dataframe(data[REQUIRED_COLUMNS[:5] + [TARGET]].head(10), use_container_width=True, hide_index=True)

st.header("3. Estimate shelf-life category")
st.write("Choose the food and storage conditions. Temperature and humidity categories use cutoffs learned from your dataset.")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        food_type = st.selectbox("Food type", sorted(data["Food_type"].unique()))
        storage = st.selectbox("Storage condition", sorted(data["Storage_condition"].unique()))
    with col2:
        packaging = st.selectbox("Packaging", sorted(data["Packaging"].unique()))
        temperature_value = st.number_input(
            "Temperature",
            min_value=float(raw_data["Temperature"].min()),
            max_value=float(raw_data["Temperature"].max()),
            value=float(raw_data["Temperature"].median()),
        )
    with col3:
        humidity_value = st.number_input(
            "Humidity",
            min_value=float(raw_data["Humidity"].min()),
            max_value=float(raw_data["Humidity"].max()),
            value=float(raw_data["Humidity"].median()),
        )
        st.caption("Use the same units as the uploaded dataset.")
    submitted = st.form_submit_button("Predict expiry category", type="primary")


def assign_level(value: float, thresholds: list[float]) -> str:
    if not thresholds:
        return "Medium"
    if value <= thresholds[0]:
        return "Low"
    if value <= thresholds[-1]:
        return "Medium"
    return "High"


if submitted:
    evidence = {
        "Food_type": food_type,
        "Storage_condition": storage,
        "Packaging": packaging,
        "Temperature": assign_level(temperature_value, cutpoints["Temperature"]),
        "Humidity": assign_level(humidity_value, cutpoints["Humidity"]),
    }
    try:
        prediction, probabilities = predict_category(model, evidence)
        st.subheader(f"Estimated shelf life: {prediction}")
        st.caption("Category thresholds: Short ≤ 7 days · Medium ≤ 15 days · Long > 15 days")
        st.write("Model probabilities")
        st.bar_chart(pd.Series(probabilities, name="Probability"))
    except Exception as exc:
        st.error(f"Could not calculate a prediction for these inputs: {exc}")

with st.expander("CSV format and notes"):
    st.write("Provide one row per food observation. Temperature, humidity, and expiry days must be numeric.")
    st.code(", ".join(REQUIRED_COLUMNS), language="text")
    st.write("Rows with missing or invalid required values are excluded. The Bayesian-style model is retrained each time the app starts or a new file is uploaded.")
