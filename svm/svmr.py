import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Page Config
st.set_page_config(
    page_title="Real Estate Price Prediction",
    layout="centered"
)

st.title("🏠 Real Estate Price Prediction using SVM")

# Upload CSV
uploaded_file = st.file_uploader(
    "Upload Real Estate CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    # Read Dataset
    df = pd.read_csv(uploaded_file)

    st.subheader("📁 Dataset Preview")
    st.dataframe(df.head())

    # Remove unnecessary column if exists
    if 'No' in df.columns:
        df.drop(columns=['No'], inplace=True)

    # Missing Values
    st.subheader("🧹 Missing Values")

    st.write(df.isnull().sum())

    # Fill Missing Values
    df = df.fillna(df.mean(numeric_only=True))

    # Features and Target
    X = df.drop(columns=['Y house price of unit area'])

    y = df['Y house price of unit area']

    # Train Test Split
    xtr, xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Feature Scaling
    scaler = StandardScaler()

    xtr = scaler.fit_transform(xtr)
    xte = scaler.transform(xte)

    # SVR Model
    model = SVR(
        kernel='rbf'
    )

    # Train Model
    model.fit(xtr, ytr)

    # Predictions
    y_pred = model.predict(xte)

    # Metrics
    mae = mean_absolute_error(yte, y_pred)

    mse = mean_squared_error(yte, y_pred)

    r2 = r2_score(yte, y_pred)

    # Display Metrics
    st.subheader("📊 Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric("MAE", f"{mae:.2f}")

    col2.metric("MSE", f"{mse:.2f}")

    col3.metric("R² Score", f"{r2:.2f}")

    # Actual vs Predicted
    st.subheader("📈 Actual vs Predicted")

    comparison = pd.DataFrame({
        "Actual": yte.values,
        "Predicted": y_pred
    })

    st.dataframe(comparison.head(15))

    # Scatter Plot
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        yte,
        y_pred
    )

    ax.set_xlabel("Actual Prices")

    ax.set_ylabel("Predicted Prices")

    ax.set_title("Actual vs Predicted Prices")

    st.pyplot(fig)

    # Residual Plot
    st.subheader("📉 Residual Plot")

    residuals = yte - y_pred

    fig2, ax2 = plt.subplots(figsize=(8, 5))

    ax2.scatter(
        y_pred,
        residuals
    )

    ax2.axhline(
        y=0,
        linestyle='--'
    )

    ax2.set_xlabel("Predicted Prices")

    ax2.set_ylabel("Residuals")

    ax2.set_title("Residual Plot")

    st.pyplot(fig2)

    st.success("✅ SVM Regression Model Trained Successfully")