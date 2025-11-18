import joblib
import pandas as pd

# Load model assets once at startup
model = joblib.load("../heart_attack_model.pkl")
scaler = joblib.load("../scaler.pkl")
model_columns = joblib.load("../model_columns.pkl")


def preprocess_input(data: dict):
    """Convert incoming JSON into model-ready dataframe"""

    df = pd.DataFrame([data])

    # One-hot encode same as Streamlit
    df = pd.get_dummies(
        df,
        columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal'],
        dummy_na=False
    )

    # Align missing columns
    df = df.reindex(columns=model_columns, fill_value=0)

    # Scale numeric values
    scaled = scaler.transform(df)

    return scaled