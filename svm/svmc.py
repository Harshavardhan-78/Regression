import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Page Config
st.set_page_config(
    page_title="Breast Cancer Prediction",
    layout="centered"
)

st.title("🩺 Breast Cancer Prediction using SVM")

# Upload CSV
uploaded_file = st.file_uploader(
    "Upload Breast Cancer CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    # Read Dataset
    df = pd.read_csv(uploaded_file)

    st.subheader("📁 Dataset Preview")
    st.dataframe(df.head())

    # Remove unnecessary columns
    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)

    if 'Unnamed: 32' in df.columns:
        df.drop(columns=['Unnamed: 32'], inplace=True)

    # Missing values
    st.subheader("🧹 Missing Values")
    st.write(df.isnull().sum())

    # Fill missing values
    df = df.fillna(df.mean(numeric_only=True))

    # Features and Target
    X = df.drop(columns=['diagnosis'])
    y = df['diagnosis']

    # Convert target labels
    y = y.map({
        'M': 1,
        'B': 0
    })

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

    # SVM Model
    model = SVC(kernel='linear')

    # Train
    model.fit(xtr, ytr)

    # Prediction
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

    st.pyplot(fig)

    # Classification Report
    st.subheader("📄 Classification Report")

    report = classification_report(yte, y_pred)

    st.text(report)

    st.success("✅ SVM Model Trained Successfully")