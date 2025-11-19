# Heart Attack Prediction System (ML Web App)

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg) 
![Built with ML](https://img.shields.io/badge/Model-Scikit--Learn-darkblue)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)

The Heart Attack Prediction System is a machine learning–powered web application that analyzes key medical indicators and predicts the likelihood of heart disease.  
It provides real-time risk scoring, interactive visualizations, and model explainability for informed decision-making.  
The app is designed for learning, experimentation, and demonstrating end-to-end ML deployment skills.

## Features

- Risk prediction using a trained ensemble model (Random Forest + Gradient Boosting + SVM).
- Clean Streamlit interface for clinical parameter input.
- Probability-based risk score with interactive gauge visualization.
- Feature importance insights from the trained model.
- SHAP-based explainability to understand individual predictions.
- Docker-containerized deployment for reproducibility.
- Optional FastAPI backend for API-based inference (not enabled in deployment v1).

## Tech Stack

### Machine Learning
- Python  
- Scikit-Learn  
- Joblib  

### Web Application
- Streamlit  

### Data Processing and Visualization
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Plotly  

### Deployment
- Docker  

---

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
