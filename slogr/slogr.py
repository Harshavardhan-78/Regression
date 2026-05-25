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
    page_title="Titanic Survival Prediction",
    layout="centered"
)

st.title("🚢 Titanic Survival Prediction")

# Upload CSV
uploaded_file = st.file_uploader(
    "Upload Titanic CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    # Read Dataset
    df = pd.read_csv(uploaded_file)

    st.subheader("📁 Dataset Preview")
    st.dataframe(df.head())

    # Missing Values Before Cleaning
    st.subheader("🧹 Missing Values Before Cleaning")
    st.write(df.isnull().sum())

    # Handle Missing Values
    if 'age' in df.columns:
        df['age'].fillna(df['age'].median(), inplace=True)

    if 'embarked' in df.columns:
        df['embarked'].fillna(
            df['embarked'].mode()[0],
            inplace=True
        )

    if 'cabin' in df.columns:
        df['cabin'].fillna('Unknown', inplace=True)

    # Missing Values After Cleaning
    st.subheader("✅ Missing Values After Cleaning")
    st.write(df.isnull().sum())

    # Features and Target
    X = df.drop(columns='survived')
    y = df['survived']

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

    # Train Model
    model.fit(xtr, ytr)

    # Predictions
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

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    st.pyplot(fig)

    # Classification Report
    st.subheader("📄 Classification Report")

    report = classification_report(yte, y_pred)

    st.text(report)

    # Actual vs Predicted
    st.subheader("📈 Actual vs Predicted")

    comparison = pd.DataFrame({
        "Actual": yte.values,
        "Predicted": y_pred
    })

    st.dataframe(comparison.head(15))

    # Plot
    fig2, ax2 = plt.subplots(figsize=(8, 4))

    ax2.plot(
        comparison["Actual"].values,
        label="Actual"
    )

    ax2.plot(
        comparison["Predicted"].values,
        label="Predicted"
    )

    ax2.legend()

    ax2.set_title("Actual vs Predicted")

    st.pyplot(fig2)

    st.success("✅ Model Trained Successfully")