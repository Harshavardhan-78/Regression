import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Logistic Regression App", layout="centered")

st.title("📊 Logistic Regression – Classification App")

# Load dataset
@st.cache_data
def load_data():
    df = sns.load_dataset("iris")
    df["target"] = (df["species"] == "setosa").astype(int)
    return df

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Features & target
X = df.drop(["species", "target"], axis=1)
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
st.subheader("📈 Model Performance")
st.write("Accuracy:", accuracy_score(y_test, y_pred))

st.text("Classification Report")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

# User input
st.subheader("🔍 Make a Prediction")

sl = st.number_input("Sepal Length", min_value=0.0)
sw = st.number_input("Sepal Width", min_value=0.0)
pl = st.number_input("Petal Length", min_value=0.0)
pw = st.number_input("Petal Width", min_value=0.0)

if st.button("Predict"):
    input_data = np.array([[sl, sw, pl, pw]])
    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)[0]

    if pred == 1:
        st.success("Prediction: Setosa 🌸")
    else:
        st.warning("Prediction: Not Setosa 🌼")
