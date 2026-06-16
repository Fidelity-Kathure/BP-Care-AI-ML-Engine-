---
title: BP Care AI Engine
emoji: 🩺
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.36.1
python_version: "3.10"
app_file: src/Engine.py
pinned: false
license: mit
---

# 🩺 BP Care AI Engine

## 🌐 Live Application

**Try the deployed system here:**

👉 [BP Care AI Engine on Hugging Face Spaces](https://huggingface.co/spaces/Bloomy-3Springs/BP-care-Ai-engine)

---

BPCare AI is an advanced predictive clinical decision support system designed to assess, monitor, and explain cardiovascular health risks through machine learning and explainable artificial intelligence (XAI).

The platform integrates physiological measurements, lifestyle behaviors, and engineered health indicators into a multimodal prediction pipeline capable of generating personalized cardiovascular risk assessments. Unlike traditional scorecard-based systems, BPCare AI utilizes ensemble machine learning models to capture complex non-linear interactions between health variables and lifestyle triggers.

To ensure transparency and trustworthiness, the platform incorporates SHAP and LIME explainability frameworks, enabling clinicians and patients to understand the factors influencing each prediction.

---

## 🎯 Key Features

- 🧠 Machine Learning–Driven Cardiovascular Risk Prediction
- 📊 Personalized Risk Scoring and Health Trajectories
- 🔍 Explainable AI using SHAP and LIME
- 🌐 Interactive Gateway API Dashboard
- ⚡ Real-Time Inference Pipeline
- 📈 Trend and Volatility Analysis
- 🩺 Lifestyle-Aware Risk Assessment
- 🔄 Dynamic Model Routing Architecture

---

## 🏗 System Architecture

```text
Patient Health Data
         │
         ▼
 Data Processing Pipeline
         │
         ▼
 Feature Engineering Layer
         │
         ▼
 Trajectory Analysis Engine
         │
         ▼
 Dynamic Routing Logic
         │
    ┌────┴────┐
    ▼         ▼
Classification Regression
(Random Forest) (Random Forest)
    │
    ▼
 Explainability Layer
 (SHAP + LIME)
    │
    ▼
 Gateway API Response
    │
    ▼
 Web Interface
```

---

## 📊 Prediction Capabilities

The system analyzes multiple health indicators, including:

- Age
- Gender
- Body Mass Index (BMI)
- Systolic Blood Pressure
- Diastolic Blood Pressure
- Sleep Duration
- Stress Levels
- Salt Intake
- Lifestyle Factors
- Additional Engineered Features

### Outputs

- Cardiovascular Risk Classification
- Risk Confidence Score
- Patient Status Assessment
- Trend Analysis
- Volatility Analysis
- Personalized Care Insights
- Feature-Level Explainability Reports

---

## 🔍 Explainable AI

BPCare AI integrates:

### SHAP (SHapley Additive exPlanations)

Provides global and local feature attribution analysis to identify how individual health factors influence prediction outcomes.

### LIME (Local Interpretable Model-Agnostic Explanations)

Generates interpretable explanations for individual predictions, improving transparency and clinician trust.

---

## 🛠 Technology Stack

### Machine Learning

- Python
- Scikit-learn
- Pandas
- NumPy
- Joblib

### Explainability

- SHAP
- LIME

### Interface & Deployment

- Gradio
- Hugging Face Spaces

### Development

- Jupyter Notebooks
- Git & GitHub

---

## 📂 Project Structure

```text
BP Care AI ML Engine/
│
├── data/
├── models/
├── notebooks/
├── src/
│   └── Engine.py
├── requirements.txt
├── README.md
└── venv/
```

---

## ▶️ Running Locally

### Activate Virtual Environment

```bash
source venv/Scripts/activate
```

### Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Launch Application

```bash
python src/Engine.py
```

---

## 📈 Example Prediction Output

The platform generates structured prediction responses containing:

- Risk Classification
- Confidence Score
- Trend Analysis
- Patient State Assessment
- Explainability Metrics
- Personalized Recommendations

Example:

```json
{
  "route": "classification",
  "assigned_class": 0,
  "status": "Low Risk",
  "confidence": 88.13
}
```

---

## 🚀 Deployment

The system is deployed through Hugging Face Spaces and exposed through an interactive Gateway API interface for real-time cardiovascular risk assessment.

---

## 👨‍💻 Author

**Fidelity Wilson**

Bachelor of Science in Computer Technology

Interests:

- Artificial Intelligence
- Clinical Decision Support Systems
- Cybersecurity
- Software Engineering
- Explainable AI
