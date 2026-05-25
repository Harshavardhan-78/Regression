import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="House Price Prediction", layout="centered")

st.title("🏠 House Price Prediction using Linear Regression")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Features and Target
    X = df.drop(columns=['Y house price of unit area'])
    y = df['Y house price of unit area']

    # Train Test Split
    xtr, xts, ytr, yts = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Model
    model = LinearRegression()

    # Train
    model.fit(xtr, ytr)

    # Prediction
    y_pred = model.predict(xts)

    # Metrics
    mae = mean_absolute_error(yts, y_pred)
    r2 = r2_score(yts, y_pred)

    st.subheader("Model Performance")

    st.write(f"### Mean Absolute Error : {mae:.2f}")
    st.write(f"### R² Score : {r2:.2f}")

    st.subheader("Predictions")

    results = pd.DataFrame({
        "Actual": yts,
        "Predicted": y_pred
    })

    st.dataframe(results.head(10))

    st.success("Model Trained Successfully ✅")