# Chronic Kidney Disease (CKD) Prediction System

## Overview
This project implements a machine learning-based system for early prediction of Chronic Kidney Disease (CKD). The system uses advanced algorithms to analyze medical test results and predict the likelihood of CKD, enabling timely intervention and better patient outcomes.

## Features
- **AI-Powered Prediction**: Uses multiple machine learning algorithms (Random Forest, Decision Tree, Logistic Regression, ANN)
- **Comprehensive Analysis**: Analyzes 24+ medical parameters including blood tests, urine tests, and medical history
- **Real-time Results**: Instant prediction with risk assessment and recommendations
- **User-Friendly Interface**: Intuitive web interface built with Flask
- **High Accuracy**: Multiple algorithms tested and validated for optimal performance

## Project Structure
```
CKD_Prediction_Project/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── CKD.pkl                        # Trained model (generated after training)
├── scaler.pkl                     # Feature scaler (generated after training)
├── templates/                     # HTML templates
│   ├── home.html                  # Landing page
│   ├── index1.html                # Prediction input form
│   ├── result.html                # Results display
│   └── indexnew.html              # About page
└── training/                      # Training scripts
    └── ckd_model_training.py       # Complete ML pipeline
```

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Step 1: Clone/Download the Project
Download or clone this project to your local machine.

### Step 2: Install Dependencies
```bash
cd CKD_Prediction_Project
pip install -r requirements.txt
```

### Step 3: Train the Model
Run the training script to create the machine learning model:
```bash
cd training
python ckd_model_training.py
```

This will:
- Create sample data (or use your own dataset)
- Perform data preprocessing and exploratory data analysis
- Train multiple ML models
- Save the best performing model as `CKD.pkl`
- Save the feature scaler as `scaler.pkl`

### Step 4: Run the Web Application
```bash
cd ..
python app.py
```

### Step 5: Access the Application
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

## Using Your Own Dataset

If you have your own CKD dataset, modify the `run_complete_pipeline()` call in `ckd_model_training.py`:

```python
ckd_trainer.run_complete_pipeline("path/to/your/dataset.csv")
```

### Dataset Requirements
Your dataset should include the following columns:
- age, blood_pressure, specific_gravity, albumin, sugar
- red_blood_cells, pus_cell, pus_cell_clumps, bacteria
- blood_glucose_random, blood_urea, serum_creatinine
- sodium, potassium, hemoglobin, packed_cell_volume
- white_blood_cell_count, red_blood_cell_count
- hypertension, diabetes_mellitus, coronary_artery_disease
- appetite, pedal_edema, anemia
- classification (target variable: 'ckd' or 'notckd')

## Machine Learning Pipeline

The project follows a comprehensive ML pipeline:

1. **Data Collection**: Using Kaggle CKD dataset or custom data
2. **Data Preprocessing**: 
   - Handling missing values
   - Label encoding for categorical variables
   - Feature scaling
3. **Exploratory Data Analysis**:
   - Statistical analysis
   - Correlation analysis
   - Data visualization
4. **Model Training**:
   - Random Forest Classifier
   - Decision Tree Classifier
   - Logistic Regression
   - Artificial Neural Network (ANN)
5. **Model Evaluation**:
   - Accuracy, Precision, Recall, F1-score
   - Cross-validation
6. **Model Deployment**:
   - Flask web application
   - Real-time predictions

## Web Application Features

### Home Page
- Project overview and introduction
- Key features and scenarios
- Navigation to prediction and about pages

### Prediction Page
- Comprehensive form with all medical parameters
- Input validation and user-friendly interface
- Organized sections: Basic Info, Urine Tests, Blood Tests, Medical History

### Results Page
- Clear prediction results with risk assessment
- Medical recommendations based on prediction
- Probability scores (when available)
- Professional disclaimer

### About Page
- Detailed project information
- Technology stack
- Social impact and benefits
- Usage instructions

## Model Performance

The system trains multiple models and selects the best performer:
- **Random Forest**: Robust ensemble method
- **Decision Tree**: Interpretable tree-based model
- **Logistic Regression**: Linear classification with probability output
- **ANN**: Deep learning for complex pattern recognition

Performance metrics include:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Medical Parameters Analyzed

### Blood Tests
- Blood Glucose Random
- Blood Urea
- Serum Creatinine
- Sodium, Potassium
- Hemoglobin
- Packed Cell Volume
- White Blood Cell Count
- Red Blood Cell Count

### Urine Tests
- Specific Gravity
- Albumin levels
- Sugar levels
- Red Blood Cells
- Pus Cells and Clumps
- Bacteria presence

### Medical History
- Age and Blood Pressure
- Hypertension
- Diabetes Mellitus
- Coronary Artery Disease
- Appetite changes
- Pedal Edema
- Anemia

## Important Disclaimers

⚠️ **Medical Disclaimer**: This system is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for any medical concerns.

⚠️ **Accuracy Note**: While the system uses advanced ML algorithms, medical diagnosis requires comprehensive evaluation by healthcare professionals.

## Contributing

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is for educational purposes. Please ensure proper attribution when using or modifying the code.

## Support

For issues or questions:
1. Check the documentation
2. Review the code comments
3. Ensure all dependencies are installed correctly
4. Verify the model training completed successfully

## Future Enhancements

Potential improvements:
- Integration with real medical databases
- Advanced deep learning models
- Mobile application development
- Multi-language support
- Electronic Health Record (EHR) integration
- Real-time monitoring capabilities

---

**Built with ❤️ for better healthcare through AI**
