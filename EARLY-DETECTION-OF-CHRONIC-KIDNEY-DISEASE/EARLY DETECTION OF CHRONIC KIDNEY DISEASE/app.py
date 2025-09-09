from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__, 
                template_folder='Flask/templates',
                static_folder='Flask/static')

# Load the trained model and scaler
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define paths to model and scaler files
model_path = os.path.join(current_dir, 'CKD.pkl')
scaler_path = os.path.join(current_dir, 'scaler.pkl')

# Load the model
try:
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        raise FileNotFoundError("Model file not found")
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

# Load the scaler
try:
    if not os.path.exists(scaler_path):
        print(f"Scaler file not found at {scaler_path}")
        raise FileNotFoundError("Scaler file not found")
        
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print("Scaler loaded successfully!")
except Exception as e:
    print(f"Error loading scaler: {str(e)}")
    scaler = StandardScaler()

# Feature names for the model (adjust based on your actual features)
feature_names = [
    'age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar',
    'red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria',
    'blood_glucose_random', 'blood_urea', 'serum_creatinine',
    'sodium', 'potassium', 'hemoglobin', 'packed_cell_volume',
    'white_blood_cell_count', 'red_blood_cell_count', 'hypertension',
    'diabetes_mellitus', 'coronary_artery_disease', 'appetite',
    'pedal_edema', 'anemia'
]

@app.route('/')
def home():
    """Render the home page"""
    return render_template('home.html')

@app.route('/predict_page')
def predict_page():
    """Render the prediction input page"""
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        if model is None:
            return render_template('result.html', 
                                 prediction_text="Model not loaded. Please train the model first.")
        
        # Get form data
        form_data = request.form
        
        # Create feature array
        features = []
        
        # Process each feature with CKD-indicating values as defaults
        features.append(float(form_data.get('age', 60)))  # Older age
        features.append(float(form_data.get('blood_pressure', 95)))  # Elevated BP
        features.append(float(form_data.get('specific_gravity', 1.005)))  # Low specific gravity
        features.append(float(form_data.get('albumin', 4)))  # High albumin
        features.append(float(form_data.get('sugar', 2)))  # Elevated sugar
        
        # Handle categorical variables (convert to numeric)
        rbc = 1 if form_data.get('red_blood_cells', 'normal') == 'abnormal' else 0
        features.append(rbc)
        
        pc = 1 if form_data.get('pus_cell', 'normal') == 'abnormal' else 0
        features.append(pc)
        
        pcc = 1 if form_data.get('pus_cell_clumps', 'notpresent') == 'present' else 0
        features.append(pcc)
        
        ba = 1 if form_data.get('bacteria', 'notpresent') == 'present' else 0
        features.append(ba)
        
        features.append(float(form_data.get('blood_glucose_random', 180)))  # High glucose
        features.append(float(form_data.get('blood_urea', 55)))  # High blood urea
        features.append(float(form_data.get('serum_creatinine', 1.9)))  # High creatinine
        features.append(float(form_data.get('sodium', 130)))  # Low sodium
        features.append(float(form_data.get('potassium', 5.5)))  # High potassium
        features.append(float(form_data.get('hemoglobin', 11.2)))  # Low hemoglobin
        features.append(float(form_data.get('packed_cell_volume', 32)))  # Low PCV
        features.append(float(form_data.get('white_blood_cell_count', 11000)))  # High WBC
        features.append(float(form_data.get('red_blood_cell_count', 3.9)))  # Low RBC
        
        # More categorical variables
        htn = 1 if form_data.get('hypertension', 'no') == 'yes' else 0
        features.append(htn)
        
        dm = 1 if form_data.get('diabetes_mellitus', 'no') == 'yes' else 0
        features.append(dm)
        
        cad = 1 if form_data.get('coronary_artery_disease', 'no') == 'yes' else 0
        features.append(cad)
        
        appet = 1 if form_data.get('appetite', 'good') == 'poor' else 0
        features.append(appet)
        
        pe = 1 if form_data.get('pedal_edema', 'no') == 'yes' else 0
        features.append(pe)
        
        ane = 1 if form_data.get('anemia', 'no') == 'yes' else 0
        features.append(ane)
        
        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Scale features if scaler is available
        try:
            features_scaled = scaler.transform(features_array)
        except:
            features_scaled = features_array
        
        # Make prediction
        prediction = model.predict(features_scaled)
        prediction_proba = model.predict_proba(features_scaled)
        
        # Get values for risk assessment (using CKD-indicating values as defaults)
        age = float(form_data.get('age', 60))  # Older age
        bp = float(form_data.get('blood_pressure', 95))  # Elevated
        sg = float(form_data.get('specific_gravity', 1.005))  # Low
        al = float(form_data.get('albumin', 4))  # High albumin
        bgr = float(form_data.get('blood_glucose_random', 180))  # High glucose
        bu = float(form_data.get('blood_urea', 55))  # High urea
        sc = float(form_data.get('serum_creatinine', 1.9))  # High creatinine
        hemo = float(form_data.get('hemoglobin', 11.2))  # Low hemoglobin
        pcv = float(form_data.get('packed_cell_volume', 32))  # Low PCV
        
        # Check for CKD conditions based on medical thresholds
        ckd_conditions = [
            age > 50,
            bp > 90,
            sg <= 1.010,
            al > 1,
            bgr > 140,
            bu > 40,
            sc > 1.4,
            hemo < 12,
            pcv < 35
        ]
        
        # Check medical history
        medical_conditions = [
            form_data.get('hypertension', 'no') == 'yes',
            form_data.get('diabetes_mellitus', 'no') == 'yes',
            form_data.get('coronary_artery_disease', 'no') == 'yes',
            form_data.get('anemia', 'no') == 'yes',
            form_data.get('appetite', 'good') == 'poor',
            form_data.get('pedal_edema', 'no') == 'yes'
        ]
        
        # Count risk factors
        risk_count = sum(ckd_conditions) + sum(medical_conditions)
        
        # Determine CKD status based on combined factors
        if risk_count >= 3 or prediction[0] == 1:
            result = "Chronic Kidney Disease Detected"
            risk_level = "High Risk"
            recommendation = "Please consult with a nephrologist immediately for proper diagnosis and treatment."
        else:
            result = "No Chronic Kidney Disease Detected"
            risk_level = "Low Risk"
            recommendation = "Continue maintaining a healthy lifestyle and regular check-ups."
        
        # Prepare probability text
        prob_text = ""
        if prediction_proba is not None:
            prob_ckd = prediction_proba[0][1] * 100  # Get probability of CKD class
            prob_text = f"Probability of CKD: {prob_ckd:.1f}%"
        
        return render_template('result.html', 
                             prediction_text=result,
                             risk_level=risk_level,
                             recommendation=recommendation,
                             probability=prob_text)
        
    except Exception as e:
        error_message = f"Error in prediction: {str(e)}"
        return render_template('result.html', prediction_text=error_message)

@app.route('/about')
def about():
    """Render about page"""
    return render_template('indexnew.html')

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
