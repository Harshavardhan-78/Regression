import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Page Config
st.set_page_config(
    page_title="Logistic Regression",
    layout="centered"
)

st.title("📊 Logistic Regression Classifier")

# Upload File
uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    # Read Dataset
    df = pd.read_csv(uploaded_file)

    st.subheader("📁 Dataset Preview")
    st.dataframe(df.head())

    # Select Target Column
    target_column = st.selectbox(
        "Select Target Column",
        df.columns
    )

    # Features and Target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Handle Missing Values
    X = X.ffill()

    # Convert categorical columns
    X = pd.get_dummies(X, drop_first=True)

    # Train Test Split
    xtr, xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Model
    model = LogisticRegression(max_iter=1000)

    # Train
    model.fit(xtr, ytr)

    # Predict
    y_pred = model.predict(xte)

    # Accuracy
    acc = accuracy_score(yte, y_pred)

    st.subheader("🎯 Accuracy Score")
    st.success(f"Accuracy : {acc:.2f}")

    # Confusion Matrix
    st.subheader("🧩 Confusion Matrix")

    cm = confusion_matrix(yte, y_pred)

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=ax
    )

    st.pyplot(fig)

    # Classification Report
    st.subheader("📄 Classification Report")

    st.text(classification_report(yte, y_pred))

    # Actual vs Predicted
    st.subheader("📈 Actual vs Predicted")

    comparison = pd.DataFrame({
        "Actual": yte.values,
        "Predicted": y_pred
    })

    st.dataframe(comparison.head(10))

    st.success("✅ Model Trained Successfully")
