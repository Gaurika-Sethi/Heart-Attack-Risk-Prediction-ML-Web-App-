import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(page_title="Heart Attack Risk Predictor", layout="centered")

@st.cache_resource
def load_assets():
    model = joblib.load("heart_attack_model.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("model_columns.pkl")
    return model, scaler, columns

model, scaler, model_columns = load_assets()

st.title("Heart Attack Risk Prediction App")
st.markdown("Provide your health details below and see your predicted heart disease risk.")

st.subheader("🩺 Enter Patient Details")

age = st.number_input("Age", 18, 100, 50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["True", "False"])
restecg = st.selectbox("Resting ECG Results", ["normal", "st-t abnormality", "left ventricular hypertrophy"])
thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", ["True", "False"])
oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, step=0.1)
slope = st.selectbox("Slope of Peak Exercise ST Segment", ["upsloping", "flat", "downsloping"])
ca = st.number_input("Major Vessels Colored by Fluoroscopy (0–3)", 0, 3, 0)
thal = st.selectbox("Thalassemia Type", ["normal", "fixed defect", "reversable defect"])

input_dict = {
    'age': [age],
    'sex': [sex],
    'cp': [cp],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs],
    'restecg': [restecg],
    'thalach': [thalach],
    'exang': [exang],
    'oldpeak': [oldpeak],
    'slope': [slope],
    'ca': [ca],
    'thal': [thal]
}

input_df = pd.DataFrame(input_dict)

# One-hot encode categorical features
input_processed = pd.get_dummies(
    input_df,
    columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal'],
    dummy_na=False
)

# Align columns with model
input_processed = input_processed.reindex(columns=model_columns, fill_value=0)

# Scale input
scaled = scaler.transform(input_processed)

if st.button("Predict"):
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]

    if pred == 1:
        st.error(f"**High Risk of Heart Disease** ({prob*100:.2f}% probability)")
    else:
        st.success(f"**Low Risk of Heart Disease** ({(1 - prob)*100:.2f}% probability)")

    st.markdown("### Heart Risk Gauge")

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob * 100,
        number={'suffix': "%"},
        delta={'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "crimson"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 60], 'color': "orange"},
                {'range': [60, 100], 'color': "red"}
            ],
            'threshold': {'line': {'color': "black", 'width': 4}, 'value': prob * 100}
        },
        title={'text': "Heart Attack Probability", 'font': {'size': 20}}
    ))

    fig.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Feature Importance")

try:
    rf_model = model.named_estimators_['rf']
    importances = pd.Series(rf_model.feature_importances_, index=model_columns)
    top_features = importances.sort_values(ascending=False).head(10)

    fig2, ax = plt.subplots()
    top_features.plot(kind='barh', color='tomato', ax=ax)
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature")
    ax.set_title("Top 10 Important Features (from Random Forest)")
    st.pyplot(fig2)

except Exception as e:
    st.info(f"Feature importance not available: {e}")

st.subheader("Model Explainability (SHAP)")

if st.button("Explain Prediction (SHAP)"):
    try:
        if hasattr(model, "named_estimators_") and 'rf' in model.named_estimators_:
            rf_model = model.named_estimators_['rf']
        else:
            rf_model = model

        st.write("Computing SHAP values...")

        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(input_processed)

        if isinstance(shap_values, list):
            shap_for_pos = shap_values[1][0]
        else:
            shap_for_pos = shap_values.values[0]

        shap_series = pd.Series(shap_for_pos, index=model_columns)
        shap_series_abs = shap_series.abs().sort_values().tail(10)

        fig, ax = plt.subplots(figsize=(6, 4))
        shap_series_abs.plot(kind='barh', color='salmon', ax=ax)
        ax.set_title("Top 10 SHAP Features Affecting This Prediction")
        st.pyplot(fig)

        signed = shap_series.sort_values(key=abs, ascending=False).head(10)
        df_explain = pd.DataFrame({
            "Feature": signed.index,
            "SHAP Value": signed.values,
            "Effect": ["Increases risk" if v > 0 else "Decreases risk" for v in signed.values]
        })
        st.table(df_explain.reset_index(drop=True))

    except Exception as e:
        st.error(f"SHAP explanation failed: {e}")






