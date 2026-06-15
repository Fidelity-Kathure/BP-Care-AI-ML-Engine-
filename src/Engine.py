import os
import joblib
import numpy as np
import pandas as pd
import shap

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'hypertension_rf_model.pkl')
PREDICTOR_PATH = os.path.join(BASE_DIR, 'models', 'predictor.pkl')

classifier = joblib.load(MODEL_PATH)
predictor = joblib.load(PREDICTOR_PATH)

# Initialize SHAP Explaners once to keep the backend fast
classifier_explainer = shap.TreeExplainer(classifier)
predictor_explainer = shap.TreeExplainer(predictor)

def analyze_patient_vitals(current_vitals, history_logs):
    # 1. Feature Engineering
    historical_baseline = np.mean(history_logs)
    x = np.arange(len(history_logs))
    trend_slope = np.polyfit(x, history_logs, 1)[0] * 7  
    
    sbp = current_vitals.get('Systolic_BP', 120.0)
    stress = current_vitals.get('Stress_Score', 4)
    bp_deviation = sbp - historical_baseline
    personalized_threshold = 15.0 - (trend_slope * 2.0)
    
    is_volatile = (bp_deviation > personalized_threshold) or (stress >= 8) or (trend_slope > 5.0)
    
    # 2. Vector Alignment
    expected_features = classifier.feature_names_in_
    mapping_dict = {
        'Age': current_vitals.get('Age', 45),
        'Systolic_BP': sbp,
        'Diastolic_BP': current_vitals.get('Diastolic_BP', 80.0),
        'BMI': current_vitals.get('BMI', 25.0),
        'Stress_Score': stress,
        'Sleep_Duration': current_vitals.get('Sleep_Duration', 7.0),
        'Gender_Encoded': current_vitals.get('Gender', 1),
        'Medication_Encoded': current_vitals.get('Medication', 0),
        'Salt_Intake': current_vitals.get('Salt_Intake', 1)
    }
    
    ordered_row = {feat: mapping_dict.get(feat, 0) for feat in expected_features}
    patient_df = pd.DataFrame([ordered_row], columns=expected_features)
    
    output = {
        "is_volatile": bool(is_volatile),
        "metrics": {
            "deviation": round(bp_deviation, 2),
            "trend_slope": round(trend_slope, 2),
            "baseline": round(historical_baseline, 2)
        }
    }
    
    # 3. Execution & Precision XAI Generation
    if is_volatile:
        predicted_score = predictor.predict(patient_df)[0]
        output["route"] = "forecasting"
        output["risk_score"] = round(float(predicted_score), 4)
        output["message"] = "Acute vital instability detected."
        
        # Compute Regressor SHAP values
        shap_values = predictor_explainer.shap_values(patient_df)
        # Handle shape differences across shap versions
        if isinstance(shap_values, list):
            shap_contributions = shap_values[0]
        elif len(shap_values.shape) == 3:
            shap_contributions = shap_values[0, :, 0]
        else:
            shap_contributions = shap_values[0]
            
    else:
        risk_labels = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"}
        pred_class = int(classifier.predict(patient_df)[0])
        pred_proba = classifier.predict_proba(patient_df)[0][pred_class]
        
        output["route"] = "classification"
        output["assigned_class"] = pred_class
        output["status"] = risk_labels[pred_class]
        output["confidence"] = round(float(pred_proba * 100), 2)
        output["message"] = "Patient state is bounded within stable limits."
        
        # Compute Classifier SHAP values for the predicted class tier
        shap_values = classifier_explainer.shap_values(patient_df)
        if isinstance(shap_values, list):
            shap_contributions = shap_values[pred_class][0]
        else:
            shap_contributions = shap_values[0, :, pred_class]

    # Map features to their exact directional impact
    xai_impacts = {}
    for feat, val, shap_val in zip(expected_features, patient_df.iloc[0], shap_contributions):
        # Clean up names for clinical reading
        clean_name = feat.replace('_Encoded', '').replace('_', ' ')
        xai_impacts[clean_name] = {
            "value": float(val),
            "impact": round(float(shap_val), 4),
            "direction": "Increases Risk" if shap_val > 0 else "Decreases Risk"
        }
        
    output["xai_analysis"] = xai_impacts
    return output