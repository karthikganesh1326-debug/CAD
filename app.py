import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Title
st.title("Coronary Artery Disease Prediction System")

# Sample dataset (same structure as your project)
data = pd.DataFrame([
    [63,1,3,145,233,1,0,150,0,2.3,0,0,1,1],
    [37,1,2,130,250,0,1,187,0,3.5,0,0,2,1],
    [41,0,1,130,204,0,0,172,0,1.4,2,0,2,1],
    [56,1,1,120,236,0,1,178,0,0.8,2,0,2,1],
    [57,0,0,120,354,0,1,163,1,0.6,2,0,2,1],
],
columns=[
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal","target"
])

X = data.drop("target", axis=1)
y = data["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# User Inputs
st.subheader("Enter Patient Details")

age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex (1 = Male, 0 = Female)", [0, 1])
cp = st.selectbox("Chest Pain Type (0–3)", [0,1,2,3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol Level", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 (1 = Yes, 0 = No)", [0,1])
restecg = st.selectbox("Resting ECG (0–2)", [0,1,2])
thalach = st.number_input("Max Heart Rate", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0,1])
oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope (0–2)", [0,1,2])
ca = st.selectbox("Number of Major Vessels (0–3)", [0,1,2,3])
thal = st.selectbox("Thalassemia (1–3)", [1,2,3])

if st.button("Predict CAD Risk"):
    user_data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,
                            thalach,exang,oldpeak,slope,ca,thal]])
    user_scaled = scaler.transform(user_data)
    result = model.predict(user_scaled)

    if result[0] == 1:
        st.error("⚠️ High Risk of Coronary Artery Disease")
    else:
        st.success("✅ Low Risk of Coronary Artery Disease")
