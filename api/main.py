from fastapi import FastAPI
from pydantic import BaseModel
from model_loading import preprocess_input, model

app = FastAPI(
    title="Heart Attack Risk Prediction API",
    description="Send patient data and receive predicted risk score.",
    version="1.0"
)

# Define input schema
class PatientData(BaseModel):
    age: int
    sex: str
    cp: str
    trestbps: int
    chol: int
    fbs: str
    restecg: str
    thalach: int
    exang: str
    oldpeak: float
    slope: str
    ca: int
    thal: str


@app.get("/")
def home():
    return {"message": "Heart Attack Prediction API is running!"}


@app.post("/predict")
def predict_heart_attack(data: PatientData):
    processed = preprocess_input(data.dict())
    probability = model.predict_proba(processed)[0][1]
    prediction = int(probability >= 0.50)

    return {
        "prediction": prediction,
        "probability": float(probability)
    }