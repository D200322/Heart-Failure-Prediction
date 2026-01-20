import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

# Load dataset and model
df = pd.read_csv('heart.csv')  # adjust to your file
model = joblib.load('heart_rf_model.pkl')  # adjust to your model

st.set_page_config(page_title="Heart Disease Prediction Dashboard", layout="wide")

st.title("Heart Disease Prediction Dashboard")

# Sidebar navigation
section = st.sidebar.radio("Choose a section", ["Key Metrics", "Visualizations", "Prediction"])

# Section: Key Metrics
if section == "Key Metrics":
    st.subheader("Dataset Overview")
    st.write("Shape of dataset:", df.shape)
    st.write("Columns:", df.columns.tolist())
    st.write("Missing values:")
    st.dataframe(df.isnull().sum())

    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

# Section: Visualizations
elif section == "Visualizations":
    st.subheader("Distribution of Age")
    fig1, ax1 = plt.subplots()
    sns.histplot(df['Age'], kde=True, ax=ax1)
    st.pyplot(fig1)

    st.subheader("Boxplot of Cholesterol")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x=df['Cholesterol'], ax=ax2)
    st.pyplot(fig2)

    st.subheader("Count of Chest Pain Type")
    fig3, ax3 = plt.subplots()
    sns.countplot(data=df, x='ChestPainType', ax=ax3)
    st.pyplot(fig3)

# Section: Prediction
elif section == "Prediction":
    st.subheader("Predict Heart Disease")

    # Collect user input
    age = st.slider("Age", 20, 100, 50)
    sex = st.selectbox("Sex", ["M", "F"])
    resting_bp = st.number_input("RestingBP", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol", 100, 600, 200)
    max_hr = st.number_input("MaxHR", 60, 220, 150)
    oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
    exercise_angina = st.selectbox("Exercise Angina", ["Y", "N"])

    # Simplified encoding (adjust based on your model)
    sex_encoded = 1 if sex == 'M' else 0
    angina_encoded = 1 if exercise_angina == 'Y' else 0

    # Create input array (update columns order if needed)
    input_data = np.array([[age, resting_bp, cholesterol, max_hr, oldpeak, sex_encoded, angina_encoded]])
    
    if st.button("Predict"):
        prediction = model.predict(input_data)
        result = "High Risk of Heart Disease" if prediction[0] == 1 else "Low Risk"
        st.success(f"Prediction: {result}")
