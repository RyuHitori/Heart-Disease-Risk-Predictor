"""
BDA700 Milestone 2: Heart Disease Risk Predictor (Streamlit)
Dataset : UCI Heart Disease (Cleveland Cohort)
Model   : classification_model.joblib (scikit-learn Pipeline trained in the notebook)
Run     : streamlit run classification_app.py
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="❤️", layout="wide")

MODEL_PATH = Path(__file__).parent / "classification_model.joblib"

# Column order must match the order used when the model was trained
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

# ------------------------------------------------------------------ header
st.title("❤️ Heart Disease Risk Predictor")
st.write(
    "Enter a patient's clinical measurements below and click **Predict** to see whether the "
    "trained machine learning model classifies the patient as **Normal** or at **Risk** of "
    "ischemic heart disease."
)

with st.sidebar:
    st.header("How to use")
    st.markdown(
        "1. Enter the patient's information in the form.\n"
        "2. Click **Predict**.\n"
        "3. Read the predicted class and the risk probability."
    )
    st.header("About the model")
    st.markdown(
        "- **Dataset:** UCI Heart Disease (Cleveland), 303 patients\n"
        "- **Target:** 0 = Normal, 1 = Risk (original levels 1 to 4)\n"
        "- **Model:** Random Forest in a scikit-learn pipeline"
    )
    st.caption("This app is for a course project and is not medical advice.")

# ------------------------------------------------------------------ inputs
st.subheader("Patient information")

SEX = {"Female": 0, "Male": 1}
CP = {"Typical angina": 1, "Atypical angina": 2, "Non-anginal pain": 3, "Asymptomatic": 4}
YES_NO = {"No": 0, "Yes": 1}
RESTECG = {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}
SLOPE = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}
THAL = {"Normal": 3, "Fixed defect": 6, "Reversible defect": 7}

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age (years)", 20, 80, 54)
    sex = st.selectbox("Sex", list(SEX))
    cp = st.selectbox("Chest pain type", list(CP))
    trestbps = st.slider("Resting blood pressure (mm Hg)", 90, 200, 130)
    chol = st.slider("Serum cholesterol (mg/dl)", 120, 570, 240)

with col2:
    fbs = st.selectbox("Fasting blood sugar above 120 mg/dl?", list(YES_NO))
    restecg = st.selectbox("Resting ECG result", list(RESTECG))
    thalach = st.slider("Maximum heart rate achieved (bpm)", 70, 210, 150)
    exang = st.selectbox("Exercise-induced angina?", list(YES_NO))

with col3:
    oldpeak = st.slider("ST depression induced by exercise (oldpeak)", 0.0, 6.5, 1.0, 0.1)
    slope = st.selectbox("Slope of the peak exercise ST segment", list(SLOPE))
    ca = st.selectbox("Major vessels colored by fluoroscopy (0 to 3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia (thal) test result", list(THAL))

# ------------------------------------------------------------------ prediction
if st.button("Predict", type="primary"):
    row = pd.DataFrame([{
        "age": age, "sex": SEX[sex], "cp": CP[cp], "trestbps": trestbps, "chol": chol,
        "fbs": YES_NO[fbs], "restecg": RESTECG[restecg], "thalach": thalach,
        "exang": YES_NO[exang], "oldpeak": oldpeak, "slope": SLOPE[slope],
        "ca": ca, "thal": THAL[thal],
    }])[FEATURES].astype(float)

    pred = int(model.predict(row)[0])
    proba = model.predict_proba(row)[0]
    risk_prob = float(proba[1])

    st.subheader("Prediction result")
    if pred == 1:
        st.error("Predicted class: **Risk** (heart disease likely)")
    else:
        st.success("Predicted class: **Normal** (heart disease not likely)")

    m1, m2 = st.columns(2)
    m1.metric("Probability of Risk", f"{risk_prob:.1%}")
    m2.metric("Probability of Normal", f"{1 - risk_prob:.1%}")
    st.progress(risk_prob)
    st.bar_chart(pd.Series({"Normal": 1 - risk_prob, "Risk": risk_prob}))
    st.caption("This prediction comes from a student machine learning model and is not a medical diagnosis.")
else:
    st.info("Fill in the form above and click **Predict**.")
