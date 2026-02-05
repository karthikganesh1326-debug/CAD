import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="CAD Risk Predictor",
    page_icon="❤️",
    layout="centered"
)

# ---------------- HEADER ----------------
st.markdown(
    """
    <h1 style='text-align: center; color: #b30000;'>
    ❤️ Coronary Artery Disease Risk Prediction
    </h1>
    <p style='text-align: center; color: gray;'>
    Clinical Decision Support System (ML-Based)
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ---------------- TRAIN MODEL (DEMO DATA) ----------------
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

model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_scaled, y)

# ---------------- INPUT FORM ----------------
st.subheader("🩺 Patient Clinical Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 20, 100, 50)
    sex = st.selectbox("Sex", ["Female", "Male"])
    cp = st.selectbox("Chest Pain Type", [0,1,2,3])
    trestbps = st.number_input("Resting Blood Pressure (mmHg)", 80, 200, 120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 400, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])

with col2:
    restecg = st.selectbox("Resting ECG Result", [0,1,2])
    thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", [0,1,2])
    ca = st.selectbox("Number of Major Vessels", [0,1,2,3])
    thal = st.selectbox("Thalassemia Type", [1,2,3])

# Convert categorical text to numbers
sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

# ---------------- PREDICTION ----------------
st.markdown("<hr>", unsafe_allow_html=True)

if st.button("🔍 Predict CAD Risk", use_container_width=True):
    input_data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,
                             thalach,exang,oldpeak,slope,ca,thal]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1] * 100

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ **High Risk of Coronary Artery Disease**  
        \nEstimated Risk Probability: **{probability:.2f}%**")
    else:
        st.success(f"✅ **Low Risk of Coronary Artery Disease**  
        \nEstimated Risk Probability: **{probability:.2f}%**")

# ---------------- FOOTER ----------------
st.markdown(
    """
    <hr>
    <p style='text-align: center; color: gray; font-size: 13px;'>
    ML-powered clinical risk assessment tool • For educational use only
    </p>
    """,
    unsafe_allow_html=True
)

