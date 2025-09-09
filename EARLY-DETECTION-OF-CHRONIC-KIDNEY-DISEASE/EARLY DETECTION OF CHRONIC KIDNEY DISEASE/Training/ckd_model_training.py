# Chronic Kidney Disease Prediction Model Training
# Following the complete pipeline from the documentation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

# Set style for visualizations
plt.style.use('fivethirtyeight')
sns.set_palette("husl")

# Define healthy ranges for features
HEALTHY_RANGES = {
    'age': (25, 35),
    'blood_pressure': (70, 85),
    'specific_gravity': (1.020, 1.025),
    'albumin': (0, 0),
    'sugar': (0, 0),
    'blood_glucose_random': (80, 90),
    'blood_urea': (13, 18),
    'serum_creatinine': (0.6, 0.8),
    'sodium': (135, 145),
    'potassium': (3.5, 4.5),
    'hemoglobin': (14, 16),
    'packed_cell_volume': (45, 50),
    'white_blood_cell_count': (7000, 9000),
    'red_blood_cell_count': (4.5, 5.5)
}

class CKDModelTraining:
    def __init__(self):
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.le = LabelEncoder()
        self.models = {}
        self.model_scores = {}
        
    def load_data(self, file_path):
        """Load the CKD dataset"""
        try:
            self.df = pd.read_csv(file_path)
            print("Dataset loaded successfully!")
            print(f"Dataset shape: {self.df.shape}")
            return True
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Please download the dataset from: https://www.kaggle.com/datasets/mansoordaku/ckdisease")
            return False
    
    def data_exploration(self):
        """Explore the dataset"""
        print("\n=== DATA EXPLORATION ===")
        print(f"Dataset Info:")
        print(self.df.info())
        print(f"\nDataset Description:")
        print(self.df.describe())
        print(f"\nMissing Values:")
        print(self.df.isnull().sum())
        print(f"\nTarget Distribution:")
        print(self.df['classification'].value_counts())
        
    def rename_columns(self):
        """Rename columns for better readability"""
        column_mapping = {
            'bp': 'blood_pressure',
            'sg': 'specific_gravity',
            'al': 'albumin',
            'su': 'sugar',
            'rbc': 'red_blood_cells',
            'pc': 'pus_cell',
            'pcc': 'pus_cell_clumps',
            'ba': 'bacteria',
            'bgr': 'blood_glucose_random',
            'bu': 'blood_urea',
            'sc': 'serum_creatinine',
            'sod': 'sodium',
            'pot': 'potassium',
            'hemo': 'hemoglobin',
            'pcv': 'packed_cell_volume',
            'wc': 'white_blood_cell_count',
            'rc': 'red_blood_cell_count',
            'htn': 'hypertension',
            'dm': 'diabetes_mellitus',
            'cad': 'coronary_artery_disease',
            'appet': 'appetite',
            'pe': 'pedal_edema',
            'ane': 'anemia'
        }
        
        # Only rename columns that exist in the dataset
        existing_columns = {k: v for k, v in column_mapping.items() if k in self.df.columns}
        self.df.rename(columns=existing_columns, inplace=True)
        print("Columns renamed successfully!")
        
    def handle_missing_values(self):
        """Handle missing values in the dataset"""
        print("\n=== HANDLING MISSING VALUES ===")
        
        # Fill missing values based on data type
        for column in self.df.columns:
            if self.df[column].dtype == 'object':
                # For categorical columns, fill with mode
                mode_value = self.df[column].mode()
                if len(mode_value) > 0:
                    self.df[column].fillna(mode_value[0], inplace=True)
            else:
                # For numerical columns, fill with median
                self.df[column].fillna(self.df[column].median(), inplace=True)
        
        print("Missing values handled!")
        print("Missing values after handling:")
        print(self.df.isnull().sum().sum())
        
    def handle_categorical_data(self):
        """Handle categorical data using Label Encoding"""
        print("\n=== HANDLING CATEGORICAL DATA ===")
        
        # Get categorical columns
        categorical_cols = set(self.df.select_dtypes(include=['object']).columns)
        
        # Remove columns that have too many unique values (likely numerical)
        for col in list(categorical_cols):
            if self.df[col].nunique() > 10:
                categorical_cols.remove(col)
                # Convert to numeric if possible
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        print(f"Categorical columns to encode: {categorical_cols}")
        
        # Apply label encoding to categorical columns
        for col in categorical_cols:
            if col != 'classification':  # Don't encode target variable yet
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
        
        # Handle target variable separately
        if 'classification' in self.df.columns:
            self.df['classification'] = LabelEncoder().fit_transform(self.df['classification'])
        
        print("Categorical data encoded successfully!")
        
    def handle_numerical_data(self):
        """Handle numerical data - clean and convert"""
        print("\n=== HANDLING NUMERICAL DATA ===")
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            # Remove any non-numeric characters and convert to float
            if self.df[col].dtype == 'object':
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
            # Fill any NaN values created during conversion
            self.df[col].fillna(self.df[col].median(), inplace=True)
        
        print("Numerical data processed successfully!")
        
    def exploratory_data_analysis(self):
        """Perform EDA with visualizations"""
        print("\n=== EXPLORATORY DATA ANALYSIS ===")
        
        # Create output directory for plots
        import os
        if not os.path.exists('plots'):
            os.makedirs('plots')
        
        # Descriptive statistics
        print("Descriptive Statistics:")
        print(self.df.describe())
        
        # Target distribution
        plt.figure(figsize=(8, 6))
        target_counts = self.df['classification'].value_counts()
        plt.pie(target_counts.values, labels=['No CKD', 'CKD'], autopct='%1.1f%%')
        plt.title('Distribution of CKD Classification')
        plt.savefig('plots/target_distribution.png')
        plt.close()
        
        # Correlation matrix
        plt.figure(figsize=(12, 10))
        correlation_matrix = self.df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig('plots/correlation_matrix.png')
        plt.close()
        
        print("EDA completed! Plots saved in 'plots' directory.")
        
    def prepare_data_for_modeling(self):
        """Prepare data for machine learning"""
        print("\n=== PREPARING DATA FOR MODELING ===")
        
        # Separate features and target
        if 'classification' in self.df.columns:
            X = self.df.drop('classification', axis=1)
            y = self.df['classification']
        else:
            print("Target column 'classification' not found!")
            return False
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set size: {self.X_train.shape}")
        print(f"Test set size: {self.X_test.shape}")
        print("Data prepared for modeling!")
        return True
        
    def build_ann_model(self):
        """Build and train Artificial Neural Network"""
        print("\n=== BUILDING ANN MODEL ===")
        
        model = Sequential()
        model.add(Dense(128, input_dim=self.X_train_scaled.shape[1], activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='binary_crossentropy', 
                     metrics=['accuracy'])
        
        # Train the model
        history = model.fit(self.X_train_scaled, self.y_train,
                           batch_size=32,
                           epochs=100,
                           validation_split=0.2,
                           verbose=0)
        
        # Evaluate
        y_pred_ann = (model.predict(self.X_test_scaled) > 0.5).astype(int)
        accuracy = accuracy_score(self.y_test, y_pred_ann)
        
        self.models['ANN'] = model
        self.model_scores['ANN'] = accuracy
        
        print(f"ANN Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred_ann))
        
        # Save the model
        model.save('ann_model.h5')
        
    def build_random_forest(self):
        """Build Random Forest model with optimized parameters"""
        print("\n=== BUILDING RANDOM FOREST MODEL ===")
        
        rf_model = RandomForestClassifier(
            n_estimators=500,
            max_depth=15,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight='balanced_subsample',
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(self.X_train_scaled, self.y_train)
        
        y_pred_rf = rf_model.predict(self.X_test_scaled)
        accuracy = accuracy_score(self.y_test, y_pred_rf)
        
        self.models['Random Forest'] = rf_model
        self.model_scores['Random Forest'] = accuracy
        
        print(f"Random Forest Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred_rf))
        
    def build_decision_tree(self):
        """Build Decision Tree model"""
        print("\n=== BUILDING DECISION TREE MODEL ===")
        
        dt_model = DecisionTreeClassifier(random_state=42)
        dt_model.fit(self.X_train, self.y_train)
        
        y_pred_dt = dt_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred_dt)
        
        self.models['Decision Tree'] = dt_model
        self.model_scores['Decision Tree'] = accuracy
        
        print(f"Decision Tree Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred_dt))
        
    def build_logistic_regression(self):
        """Build Logistic Regression model"""
        print("\n=== BUILDING LOGISTIC REGRESSION MODEL ===")
        
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(self.X_train_scaled, self.y_train)
        
        y_pred_lr = lr_model.predict(self.X_test_scaled)
        accuracy = accuracy_score(self.y_test, y_pred_lr)
        
        self.models['Logistic Regression'] = lr_model
        self.model_scores['Logistic Regression'] = accuracy
        
        print(f"Logistic Regression Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred_lr))
        
    def evaluate_models(self):
        """Evaluate and compare all models"""
        print("\n=== MODEL EVALUATION ===")
        
        print("Model Performance Summary:")
        for model_name, score in self.model_scores.items():
            print(f"{model_name}: {score:.4f}")
        
        # Find best model
        best_model_name = max(self.model_scores, key=self.model_scores.get)
        best_score = self.model_scores[best_model_name]
        
        print(f"\nBest Model: {best_model_name} with accuracy: {best_score:.4f}")
        
        return best_model_name
        
    def save_best_model(self, best_model_name):
        """Save the best model for deployment"""
        print(f"\n=== SAVING BEST MODEL: {best_model_name} ===")
        
        best_model = self.models[best_model_name]
        
        # Save model and scaler
        with open('CKD.pkl', 'wb') as f:
            pickle.dump(best_model, f)
            
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
            
        print("Model and scaler saved successfully!")
        
    def run_complete_pipeline(self, data_file=None):
        """Run the complete ML pipeline"""
        print("=== STARTING CKD PREDICTION MODEL TRAINING ===")
        
        # Create sample data if no file provided
        if data_file is None:
            self.create_sample_data()
        else:
            if not self.load_data(data_file):
                self.create_sample_data()
        
        # Run all steps
        self.data_exploration()
        self.rename_columns()
        self.handle_missing_values()
        self.handle_categorical_data()
        self.handle_numerical_data()
        self.exploratory_data_analysis()
        
        if self.prepare_data_for_modeling():
            self.build_random_forest()
            self.build_decision_tree()
            self.build_logistic_regression()
            # self.build_ann_model()  # Uncomment if TensorFlow is available
            
            best_model = self.evaluate_models()
            self.save_best_model(best_model)
            
            print("\n=== PIPELINE COMPLETED SUCCESSFULLY! ===")
        
    def create_sample_data(self):
        """Create sample CKD data for demonstration"""
        print("Creating sample data for demonstration...")
        
        np.random.seed(42)
        n_samples = 400
        
        # Create synthetic CKD data
        data = {
            'age': np.random.randint(20, 80, n_samples),
            'blood_pressure': np.random.randint(60, 180, n_samples),
            'specific_gravity': np.random.uniform(1.005, 1.025, n_samples),
            'albumin': np.random.randint(0, 5, n_samples),
            'sugar': np.random.randint(0, 5, n_samples),
            'red_blood_cells': np.random.choice(['normal', 'abnormal'], n_samples),
            'pus_cell': np.random.choice(['normal', 'abnormal'], n_samples),
            'pus_cell_clumps': np.random.choice(['present', 'notpresent'], n_samples),
            'bacteria': np.random.choice(['present', 'notpresent'], n_samples),
            'blood_glucose_random': np.random.randint(70, 300, n_samples),
            'blood_urea': np.random.randint(10, 150, n_samples),
            'serum_creatinine': np.random.uniform(0.5, 15, n_samples),
            'sodium': np.random.randint(120, 160, n_samples),
            'potassium': np.random.uniform(2.5, 7.0, n_samples),
            'hemoglobin': np.random.uniform(6, 18, n_samples),
            'packed_cell_volume': np.random.randint(20, 55, n_samples),
            'white_blood_cell_count': np.random.randint(3000, 15000, n_samples),
            'red_blood_cell_count': np.random.uniform(2.5, 8.0, n_samples),
            'hypertension': np.random.choice(['yes', 'no'], n_samples),
            'diabetes_mellitus': np.random.choice(['yes', 'no'], n_samples),
            'coronary_artery_disease': np.random.choice(['yes', 'no'], n_samples),
            'appetite': np.random.choice(['good', 'poor'], n_samples),
            'pedal_edema': np.random.choice(['yes', 'no'], n_samples),
            'anemia': np.random.choice(['yes', 'no'], n_samples)
        }
        
        # Create target variable with some logic
        classification = []
        for i in range(n_samples):
            # Simple logic to create realistic target
            risk_score = 0
            if data['age'][i] > 60: risk_score += 1
            if data['blood_pressure'][i] > 140: risk_score += 1
            if data['serum_creatinine'][i] > 1.5: risk_score += 2
            if data['hemoglobin'][i] < 10: risk_score += 1
            if data['diabetes_mellitus'][i] == 'yes': risk_score += 1
            if data['hypertension'][i] == 'yes': risk_score += 1
            
            # Add some randomness
            risk_score += np.random.randint(-1, 2)
            
            classification.append('ckd' if risk_score >= 3 else 'notckd')
        
        data['classification'] = classification
        
        self.df = pd.DataFrame(data)
        
        # Add some missing values
        for col in ['specific_gravity', 'albumin', 'sugar', 'serum_creatinine']:
            missing_idx = np.random.choice(self.df.index, size=int(0.1 * len(self.df)), replace=False)
            self.df.loc[missing_idx, col] = np.nan
            
        print(f"Sample dataset created with {n_samples} samples!")

if __name__ == "__main__":
    # Initialize and run the training pipeline
    ckd_trainer = CKDModelTraining()
    
    # You can provide your own dataset file path here
    # ckd_trainer.run_complete_pipeline("path/to/your/ckd_dataset.csv")
    
    # Or run with sample data
    ckd_trainer.run_complete_pipeline()
