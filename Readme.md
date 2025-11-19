# Heart Attack Risk Prediction — ML Pipeline with Explainability & Edge Simulation

![MIT License](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Docker](https://img.shields.io/badge/Deploy-Docker-blue)
![Scikit-Learn](https://img.shields.io/badge/Model-Scikit--Learn-yellow)

## Problem Statement
Heart disease prediction remains clinically important because early risk estimation reduces severe cardiac events. This project builds a fully reproducible ML pipeline that predicts heart attack risk using structured clinical features and provides interpretability, calibration, and deployable inference.

## Dataset & Preprocessing 

### Source
The dataset used in this project comes from the UCI Heart Disease Repository, one of the most widely referenced clinical datasets for cardiovascular risk modeling.
A compiled and cleaned version of this dataset is available on Kaggle:
https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data

### Samples
The dataset contains:

- 303 patient records
- Each record represents a real clinical case from the Cleveland Clinic study
- Binary classification target (num) indicating presence or absence of heart disease

## Features
The dataset includes 14 clinically relevant attributes, capturing demographic, symptomatic, ECG-based, and exercise-induced measurements. Key features include:

- Age
- Sex
- Chest pain type (cp)
- Resting blood pressure (trestbps)
- Serum cholesterol (chol)
- Fasting blood sugar > 120 mg/dl (fbs)
- Resting ECG results (restecg)
- Maximum heart rate achieved (thalach)
- Exercise-induced angina (exang)
- ST depression (oldpeak)
- Slope of peak exercise ST segment (slope)
- Number of major vessels colored by fluoroscopy (ca)
- Thalassemia type (thal)

These features form the basis of most medical heart-risk diagnostic models, enabling both classical ML and clinical interpretability.

Train/test split: 80/20 with stratification.

### Missing-Value Handling
The dataset contains a small number of missing entries, primarily in the ca and thal columns.

To maintain dataset integrity:

- Rows with missing target (num) values are removed
- Remaining missing values in inputs are imputed using simple statistical strategies:
1. Numerical features → filled with mean
2. Categorical features → filled with mode


This ensures the model receives a complete, stable training matrix.  

### One-Hot Encoding
Several features are categorical (e.g., sex, cp, restecg, slope, thal).
To make them suitable for machine learning models:

- All categorical columns were converted into binary indicator variables
- get_dummies(..., drop_first=True) was applied to avoid multicollinearity
- Only the resulting transformed feature names were saved and reused during prediction to ensure input consistency 

### Standard Scaling
To ensure numerical stability across models:

- All continuous features (age, trestbps, chol, thalach, oldpeak) were scaled using StandardScaler
- This transformation brings every feature to zero mean and unit variance
- Scaling prevents features with larger numeric ranges from dominating model training

The scaler was fitted on the training set only and saved so the exact transformation is applied during inference.

### Machine Learning Algorithms
- Logistic Regression - Served as the baseline linear classifier 
- K-Nearest Neighbors (KNN) - Instance-based classifier 
- Support Vector Machine (SVM, RBF kernel) - Non-linear classifier 
- Random Forest- Ensemble of decision trees 
- Gradient Boosting Classifier - Sequential boosting algorithm .

These models were compared using the same training and testing pipeline to ensure consistency.

### Metrics reported
- ROC AUC
- PR AUC
- Accuracy
- F1 score
- Brier calibration score  

| Model                | ROC AUC | F1    | Brier  | Notes               |
|----------------------|---------|-------|---------|----------------------|
| Logistic Regression  | 0.900   | 0.845 | 0.122   | baseline             |
| KNN                  | 0.912   | 0.867 | 0.115   | strong local models  |
| SVM (RBF)            | 0.913   | 0.863 | 0.108   | highest calibration  |
| Random Forest        | 0.916   | 0.870 | 0.114   | best overall stability |
| Naive Bayes          | 0.896   | 0.870 | 0.136   | simple & fast        |
| Gradient Boosting    | 0.909   | 0.867 | 0.114   | balanced metrics     |

---

## Explainability (SHAP Analysis)
To ensure transparency and interpretability of the prediction model, this project incorporates SHAP (SHapley Additive exPlanations) — a state-of-the-art method for explaining machine-learning decisions.

### Global insights
Top predictive features:

- Chest Pain Type, Maximum Heart Rate (thalach), ST Depression (oldpeak), and Exercise-Induced Angina emerged as the strongest predictors in the model.
- Higher values of ST depression and exercise-induced angina often push predictions toward higher risk, while healthier ECG profile values push the prediction lower.
- Variables such as age, cholesterol, and resting blood pressure still contribute meaningfully but with lower overall SHAP magnitude.

## Deployment Architecture
```bash
User Input → Streamlit UI → Preprocessing (scaler + OHE alignment) → ML Model → SHAP Explanation → Output
  ```
### Components:
- Streamlit frontend (UI)
- FastAPI backend 
- Docker container support

## Results
This heart-disease prediction model was evaluated using multiple performance metrics to ensure reliability in a medical-risk context.

**Overall Model Performance (Best Models):**
- **ROC AUC:** 0.91 (Random Forest & SVM performed best)
- **F1 Score:** ~0.87 (consistent across RF, GB, KNN, NB)
- **PR-AUC:** ~0.90 (strong performance on positive-class detection)
- **Calibration (Brier Score):** ~0.11–0.13 (moderately calibrated; acceptable for medical tabular models)
- **Prediction Latency (Local/Docker):** ~3–10 ms per request (lightweight model, near-instant inference)

## Limitations
- Small dataset (303 samples) 
- Not clinically validated 
- Feature inputs are static
- Model calibration is moderate 
- SHAP explanations run on tree-based models only 
- No full backend–frontend integration deployed

## Future Work
- Integrate real patient/sensor datasets (e.g., PhysioNet) to improve robustness.
- Add probability calibration using Platt scaling or isotonic regression.
- Optimize for real-time inference and expose the VotingClassifier through the API backend.
- Add ONNX export for lightweight deployment on edge devices (e.g., Jetson Nano, Raspberry Pi).
- Incorporate live HR/ECG monitoring with periodic inference.

## Installation

Follow these steps to run the project locally:

### Clone the repository  
```bash
   git clone https://github.com/your-username/heart-attack-predictor.git
   cd heart-attack-predictor
  ```
### Create a virtual environment 
```bash
python -m venv venv
```
### Activate the environment 

Windows:
```bash
venv\Scripts\activate
```
macOS/Linux:
```bash
source venv/bin/activate
```

### Install dependencies 
```bash
pip install -r requirements.txt
```
### Run the Streamlit app 
```bash
streamlit run app.py
```
## Usage

Follow these steps to run the project locally:

### Open the app at 
```bash
 http://localhost:8501
```
### Enter patient attributes such as age, cholesterol, chest pain type, ECG results, and more.

### View:
- Risk prediction score
- Probability gauge meter
- Feature importance chart
- SHAP-based explanation of model reasoning

## Docker Deployment

### Build Image
```bash
docker build -t heart-attack-app .
```
### Run Container
```bash
docker run -p 8501:8501 heart-attack-app
```
### Visit in browser: 
```bash
http://localhost:8501
```
## API Backend (Optional)
A FastAPI backend is included for programmatic predictions.

### Run locally:
```bash
cd api
uvicorn main:app --reload
```
### Documentation will be available at:
```bash
http://localhost:8000/docs
```
## Requirements
- Python 3.10
- pip
- Docker (optional)
- SHAP support enabled by using compatible NumPy and scikit-learn versions
## Contact
- LinkedIn: https://www.linkedin.com/in/gaurika-sethi-53043b321
- Medium: https://medium.com/@pixelsnsyntax
- Twitter: https://twitter.com/pixelsnsyntax

Project Link: [Repository link](https://github.com/Gaurika-Sethi/Heart-Attack-Risk-Prediction-ML-Web-App-.git)

## License

This project is licensed under the **MIT License**  see the [LICENSE](LICENSE) file for full details. 
