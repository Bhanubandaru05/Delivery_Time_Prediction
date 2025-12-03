import streamlit as st
# use joblib instead of pickle because the model was saved with joblib
from joblib import load
import pandas as pd
import numpy as np
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# (Optional) hide annoying sklearn version warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# ---------------- Streamlit Page Config ----------------
st.set_page_config(page_title="Delivery Time Prediction", layout="centered")

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    # if you changed the filename to .joblib, update it here
    model = load("best_gradient_boosting_model.pkl")
    return model

model = load_model()

st.title("🚚 Delivery Time Prediction App")
st.write("Predict **Estimated Delivery Time (hours)** using your trained Gradient Boosting Regressor.")

# ---------------- Feature Names ----------------
numeric_features = [
    "Shipping_Distance_km",
    "Order_Weight_kg"
]

categorical_features = [
    "Delivery_Method",
    "Traffic_Conditions",
    "Order_Priority",
    "Weather_Conditions",
    "Payment_Method",
    "Region"
]

all_features = numeric_features + categorical_features

# ---------------- Input Mode ----------------
mode = st.radio(
    "Choose input method:",
    ["Single Prediction", "Batch Prediction (CSV)"]
)

# ---------------- Helper: Preprocess Data ----------------
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical features to match model training."""
    df_processed = pd.get_dummies(df)

    # Align columns with what the model expects
    try:
        model_features = model.feature_names_in_
    except AttributeError:
        # fallback if feature_names_in_ is not present
        model_features = df_processed.columns

    df_processed = df_processed.reindex(columns=model_features, fill_value=0)

    return df_processed

# ---------------- Single Prediction ----------------
if mode == "Single Prediction":

    st.subheader("Enter Order Details")

    # Numeric Inputs
    shipping_distance = st.number_input("Shipping Distance (km)", 0.0, 2000.0, 100.0)
    order_weight = st.number_input("Order Weight (kg)", 0.0, 100.0, 5.0)

    # Categorical Inputs
    delivery_method = st.selectbox("Delivery Method", ["Standard", "Express", "Overnight"])
    traffic = st.selectbox("Traffic Conditions", ["Light", "Moderate", "Heavy"])
    priority = st.selectbox("Order Priority", ["Low", "Medium", "High"])
    weather = st.selectbox("Weather Conditions", ["Clear", "Rainy", "Snowy"])
    payment = st.selectbox("Payment Method", ["Credit Card", "PayPal", "Bank Transfer"])
    region = st.selectbox("Region", ["North", "South", "East", "West"])

    # Prepare input row
    row = pd.DataFrame([{
        "Shipping_Distance_km": shipping_distance,
        "Order_Weight_kg": order_weight,
        "Delivery_Method": delivery_method,
        "Traffic_Conditions": traffic,
        "Order_Priority": priority,
        "Weather_Conditions": weather,
        "Payment_Method": payment,
        "Region": region
    }])

    if st.button("Predict Delivery Time"):
        try:
            input_processed = preprocess(row)
            prediction = model.predict(input_processed)[0]
            st.success(f"⏱ Estimated Delivery Time: **{prediction:.2f} hours**")
        except Exception as e:
            st.error("❌ Prediction failed. Check model, feature names, and input format.")
            st.exception(e)

# ---------------- Batch Prediction ----------------
else:
    st.subheader("Upload CSV for Batch Prediction")

    st.markdown(
        "📌 **Note:** The CSV should contain these columns:\n"
        f"- {', '.join(all_features)}"
    )

    uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())

        if st.button("Predict for all rows"):
            try:
                df_processed = preprocess(df)
                preds = model.predict(df_processed)
                df["Predicted_Delivery_Time"] = preds

                st.subheader("Predictions (first 10 rows)")
                st.dataframe(df.head(10))

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Predictions as CSV", csv, "predictions.csv")
            except Exception as e:
                st.error("❌ Prediction failed. Ensure CSV columns match training features.")
                st.exception(e)
