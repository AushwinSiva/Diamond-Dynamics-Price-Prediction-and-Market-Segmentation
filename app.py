import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================

# >>>>>>> CHANGE THIS TO YOUR CSV FILE NAME IF NEEDED <<<<<<<
DATA_FILE = "diamonds.csv"   # e.g. "diamonds.csv" or "df_cleaned.csv"

# Manual encodings (must match dataset categories)
CUT_MAP = {
    "Fair": 0,
    "Good": 1,
    "Very Good": 2,
    "Premium": 3,
    "Ideal": 4,
}

COLOR_MAP = {
    "J": 0,
    "I": 1,
    "H": 2,
    "G": 3,
    "F": 4,
    "E": 5,
    "D": 6,
}

CLARITY_MAP = {
    "I1": 0,
    "SI2": 1,
    "SI1": 2,
    "VS2": 3,
    "VS1": 4,
    "VVS2": 5,
    "VVS1": 6,
    "IF": 7,
}

CLUSTER_DESC = {
    "Premium Heavy Diamonds": "High carat, high price – large, expensive, premium-grade stones.",
    "Affordable Small Diamonds": "Low carat, low price – small, budget-friendly stones.",
    "Mid-range Balanced Diamonds": "Medium carat, medium price – balanced in size and cost.",
}


# =========================================================
# TRAINING + MODEL BUILDING (RUNS ONCE, CACHED)
# =========================================================

@st.cache_resource
def load_data_and_train_models(data_path: str):
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file '{data_path}' not found. Put your CSV in the same folder as app.py "
            f"or change DATA_FILE at the top of the script."
        )

    # 1) LOAD DATA
    df = pd.read_csv(path)
    # Basic sanity cleaning: remove invalid dims
    df = df[(df["x"] > 0) & (df["y"] > 0) & (df["z"] > 0)].copy()
    df = df.dropna()

    # 2) FEATURE ENGINEERING
    df["volume"] = df["x"] * df["y"] * df["z"]
    df["dimension_ratio"] = (df["x"] + df["y"]) / (2 * df["z"])
    df["carat_log"] = np.log1p(df["carat"])
    df["price_log"] = np.log1p(df["price"])

    # 3) ENCODE CATEGORICALS USING FIXED MAPS
    df["cut"] = df["cut"].map(CUT_MAP)
    df["color"] = df["color"].map(COLOR_MAP)
    df["clarity"] = df["clarity"].map(CLARITY_MAP)

    # Drop rows where mapping failed
    df = df.dropna(subset=["cut", "color", "clarity"])

    # 4) REGRESSION DATA
    reg_feature_cols = [
        "carat",
        "x",
        "y",
        "volume",
        "carat_log",
        "dimension_ratio",
        "cut",
        "color",
        "clarity",
    ]
    X = df[reg_feature_cols]
    y = df["price_log"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf_model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)

    # regression metrics (optional, not shown in UI but we compute once)
    y_pred_test = rf_model.predict(X_test)
    r2 = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    reg_metrics = {"R2": r2, "MAE": mae, "MSE": mse, "RMSE": rmse}

    # 5) CLUSTERING DATA
    cluster_feature_cols = [
        "carat",
        "x",
        "y",
        "z",
        "volume",
        "dimension_ratio",
        "cut",
        "color",
        "clarity",
        "price_log",
    ]
    df_cluster = df[cluster_feature_cols].copy()

    scaler_clust = StandardScaler()
    X_cluster_scaled = scaler_clust.fit_transform(df_cluster)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_cluster_scaled)
    labels = kmeans.labels_

    df_cluster["cluster"] = labels

    # 6) DETERMINE CLUSTER NAMES DYNAMICALLY (BASED ON CARAT + PRICE)
    stats = (
        df_cluster.groupby("cluster")[["carat", "price_log"]]
        .mean()
        .sort_values("carat")
    )
    # Smallest carat -> Affordable, middle -> Mid-range, largest -> Premium
    cluster_ids_sorted = list(stats.index)
    cluster_name_map = {}
    if len(cluster_ids_sorted) == 3:
        cluster_name_map[cluster_ids_sorted[0]] = "Affordable Small Diamonds"
        cluster_name_map[cluster_ids_sorted[1]] = "Mid-range Balanced Diamonds"
        cluster_name_map[cluster_ids_sorted[2]] = "Premium Heavy Diamonds"
    else:
        # fallback (shouldn't happen)
        cluster_name_map = {i: f"Cluster {i}" for i in range(len(cluster_ids_sorted))}

    # 7) SAVE MODELS + METADATA AS PKL (for your project requirement)
    joblib.dump(rf_model, "best_regression_model.pkl")
    joblib.dump(reg_feature_cols, "reg_feature_columns.pkl")
    joblib.dump(kmeans, "best_clustering_model.pkl")
    joblib.dump(scaler_clust, "cluster_scaler.pkl")
    joblib.dump(cluster_feature_cols, "cluster_feature_columns.pkl")

    return (
        rf_model,
        reg_feature_cols,
        kmeans,
        scaler_clust,
        cluster_feature_cols,
        cluster_name_map,
        reg_metrics,
    )


# =========================================================
# APP LAYOUT
# =========================================================

st.set_page_config(page_title="Diamond Dynamics", layout="centered")

st.title("💎 Diamond Dynamics — Price Prediction & Market Segmentation")

# Load models (trains on first run, cached afterwards)
try:
    (
        reg_model,
        reg_feature_cols,
        kmeans_model,
        cluster_scaler,
        cluster_feature_cols,
        cluster_name_map,
        reg_metrics,
    ) = load_data_and_train_models(DATA_FILE)
except Exception as e:
    st.error(str(e))
    st.stop()

mode = st.sidebar.radio(
    "Select Mode",
    ["Price Prediction (Regression)", "Market Segmentation (Clustering)"],
)
st.sidebar.markdown("---")
st.sidebar.write("Models are trained automatically from the local CSV.")

# =========================================================
# COMMON INPUT FORM
# =========================================================

st.subheader("Enter Diamond Features")

col1, col2 = st.columns(2)

with col1:
    carat = st.number_input(
        "Carat", min_value=0.1, max_value=5.0, value=0.5, step=0.01
    )
    x = st.number_input(
        "X (length in mm)", min_value=0.0, max_value=15.0, value=5.0, step=0.01
    )
    y = st.number_input(
        "Y (width in mm)", min_value=0.0, max_value=15.0, value=5.0, step=0.01
    )
    z = st.number_input(
        "Z (depth in mm)", min_value=0.0, max_value=15.0, value=3.0, step=0.01
    )

with col2:
    cut = st.selectbox("Cut", list(CUT_MAP.keys()))
    color = st.selectbox("Color", list(COLOR_MAP.keys()))
    clarity = st.selectbox("Clarity", list(CLARITY_MAP.keys()))

# Engineered features for input
def compute_engineered(carat, x, y, z):
    volume = x * y * z
    dim_ratio = 0.0 if z == 0 else (x + y) / (2 * z)
    carat_log = np.log1p(carat)
    return volume, dim_ratio, carat_log


volume, dimension_ratio, carat_log = compute_engineered(carat, x, y, z)

cut_enc = CUT_MAP[cut]
color_enc = COLOR_MAP[color]
clarity_enc = CLARITY_MAP[clarity]

# =========================================================
# 1️⃣ PRICE PREDICTION MODE
# =========================================================

if mode == "Price Prediction (Regression)":
    st.subheader("🪙 Price Prediction")

    st.markdown(
        "This uses the **best regression model (Random Forest)** trained on engineered features "
        "like volume, dimension_ratio and carat_log."
    )

    if st.button("Predict Price"):
        feature_dict = {
            "carat": carat,
            "x": x,
            "y": y,
            "volume": volume,
            "carat_log": carat_log,
            "dimension_ratio": dimension_ratio,
            "cut": cut_enc,
            "color": color_enc,
            "clarity": clarity_enc,
        }

        X_input = np.array([[feature_dict[col] for col in reg_feature_cols]])
        y_pred_log = reg_model.predict(X_input)[0]
        y_pred_price = np.expm1(y_pred_log)

        st.success(f"Estimated Diamond Price: **₹{y_pred_price:,.2f}** (approx.)")

        with st.expander("Model performance on test data (log-price):"):
            st.write(
                {
                    "R²": round(reg_metrics["R2"], 4),
                    "MAE": round(reg_metrics["MAE"], 4),
                    "RMSE": round(reg_metrics["RMSE"], 4),
                }
            )

        st.caption(
            "Note: Model was trained on log(price) to handle skewness, "
            "then predictions are converted back to the original price scale."
        )

# =========================================================
# 2️⃣ CLUSTERING MODE
# =========================================================

else:
    st.subheader("📊 Market Segmentation (Clustering)")

    st.markdown(
        "This predicts which **market segment** the diamond belongs to "
        "using a K-Means clustering model trained on size, quality and price."
    )

    price_input = st.number_input(
        "Actual Price (INR, for segmentation)",
        min_value=100.0,
        max_value=2_000_000.0,
        value=20_000.0,
        step=500.0,
    )

    if st.button("Predict Cluster"):
        price_log = np.log1p(price_input)

        feature_dict_cluster = {
            "carat": carat,
            "x": x,
            "y": y,
            "z": z,
            "volume": volume,
            "dimension_ratio": dimension_ratio,
            "cut": cut_enc,
            "color": color_enc,
            "clarity": clarity_enc,
            "price_log": price_log,
        }

        X_cluster_input = np.array(
            [[feature_dict_cluster[col] for col in cluster_feature_cols]]
        )
        X_cluster_scaled_single = cluster_scaler.transform(X_cluster_input)
        cluster_label = int(kmeans_model.predict(X_cluster_scaled_single)[0])

        cluster_name = cluster_name_map.get(cluster_label, f"Cluster {cluster_label}")
        desc = CLUSTER_DESC.get(cluster_name, "")

        st.success(f"Assigned Segment: **Cluster {cluster_label} — {cluster_name}**")
        if desc:
            st.write(desc)

        st.caption(
            "Segmentation uses standardized features and log-transformed price, "
            "with K-Means (K=3) to form Premium / Mid-range / Affordable segments."
        )
