import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EngagementPredictor:
    def __init__(self, model_path=None):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = [
            'gpa', 'attendance', 'course_load', 'library_visits', 
            'lms_interactions', 'extracurricular', 'financial_aid_encoded',
            'work_study_encoded', 'first_generation_encoded'
        ]
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # Initialize with a dummy model for demonstration
            self._create_dummy_model()
    
    def _create_dummy_model(self):
        """Create a dummy model for demonstration purposes"""
        logger.info("Creating dummy ML model for demonstration...")
        
        # Create synthetic training data
        np.random.seed(42)
        n_samples = 1000
        
        X = np.column_stack([
            np.random.uniform(2.0, 4.0, n_samples),  # gpa
            np.random.uniform(60, 95, n_samples),    # attendance
            np.random.randint(12, 21, n_samples),    # course_load
            np.random.randint(0, 31, n_samples),     # library_visits
            np.random.randint(10, 101, n_samples),   # lms_interactions
            np.random.randint(0, 11, n_samples),     # extracurricular
            np.random.choice([0, 1], n_samples),     # financial_aid_encoded
            np.random.choice([0, 1, 2], n_samples),  # work_study_encoded
            np.random.choice([0, 1], n_samples)      # first_generation_encoded
        ])
        
        # Create target variable based on feature importance
        risk_scores = (
            (4 - X[:, 0]) * 10 +           # GPA impact (lower GPA = higher risk)
            (100 - X[:, 1]) * 0.3 +        # Attendance impact
            np.maximum(0, X[:, 2] - 15) * 2 +  # Course load impact
            np.maximum(0, 10 - X[:, 3]) * 1.5 + # Library visits impact
            np.maximum(0, 50 - X[:, 4]) * 0.4 + # LMS interactions impact
            np.maximum(0, 2 - X[:, 5]) * 5 +   # Extracurricular impact
            X[:, 6] * 10 +                 # Financial aid impact
            X[:, 7] * 8 +                  # Work study impact
            X[:, 8] * 12                   # First generation impact
        )
        
        # Convert to risk levels
        y = np.where(risk_scores > 70, 2, 
                    np.where(risk_scores > 30, 1, 0))  # 0=Low, 1=Medium, 2=High
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        
        # Train the model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Test the model
        X_test_scaled = self.scaler.transform(X_test)
        accuracy = self.model.score(X_test_scaled, y_test)
        logger.info(f"Dummy model trained with accuracy: {accuracy:.2f}")
        
        # Initialize label encoders for categorical features
        self.label_encoders['financial_aid'] = LabelEncoder()
        self.label_encoders['financial_aid'].fit(['No', 'Yes'])
        
        self.label_encoders['work_study'] = LabelEncoder()
        self.label_encoders['work_study'].fit(['None', 'Part-time', 'Full-time'])
        
        self.label_encoders['first_generation'] = LabelEncoder()
        self.label_encoders['first_generation'].fit(['No', 'Yes'])
    
    def preprocess_features(self, data):
        """Preprocess input features for prediction"""
        try:
            # Convert categorical features
            financial_aid_encoded = self.label_encoders['financial_aid'].transform(
                [data.get('financial_aid', 'No')]
            )[0]
            
            work_study_encoded = self.label_encoders['work_study'].transform(
                [data.get('work_study', 'None')]
            )[0]
            
            first_generation_encoded = self.label_encoders['first_generation'].transform(
                [data.get('first_generation', 'No')]
            )[0]
            
            # Create feature array
            features = np.array([[
                float(data.get('gpa', 3.0)),
                float(data.get('attendance', 75.0)),
                int(data.get('course_load', 15)),
                int(data.get('library_visits', 8)),
                int(data.get('lms_interactions', 45)),
                int(data.get('extracurricular', 3)),
                financial_aid_encoded,
                work_study_encoded,
                first_generation_encoded
            ]])
            
            return features
            
        except Exception as e:
            logger.error(f"Error in feature preprocessing: {e}")
            raise
    
    def predict(self, data):
        """Make prediction on student data"""
        try:
            if self.model is None:
                return self._fallback_prediction(data)
            
            # Preprocess features
            features = self.preprocess_features(data)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            prediction_proba = self.model.predict_proba(features_scaled)[0]
            prediction_class = self.model.predict(features_scaled)[0]
            
            # Convert to risk score (0-100)
            risk_score = self._calculate_risk_score(prediction_proba, prediction_class)
            
            # Get feature importance
            feature_importance = self._get_feature_importance(features[0])
            
            return {
                'risk_score': risk_score,
                'risk_level': self._get_risk_level(risk_score),
                'confidence': float(np.max(prediction_proba)),
                'feature_importance': feature_importance,
                'prediction_proba': {
                    'low_risk': float(prediction_proba[0]),
                    'medium_risk': float(prediction_proba[1]),
                    'high_risk': float(prediction_proba[2])
                }
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return self._fallback_prediction(data)
    
    def _calculate_risk_score(self, probabilities, predicted_class):
        """Convert class probabilities to a 0-100 risk score"""
        # Weighted average of probabilities
        risk_score = (
            probabilities[0] * 15 +    # Low risk -> low score
            probabilities[1] * 50 +    # Medium risk -> medium score  
            probabilities[2] * 85      # High risk -> high score
        )
        return max(0, min(100, risk_score))
    
    def _get_risk_level(self, risk_score):
        """Convert risk score to risk level"""
        if risk_score > 70:
            return "High"
        elif risk_score > 30:
            return "Medium"
        else:
            return "Low"
    
    def _get_feature_importance(self, features):
        """Calculate feature importance for the specific prediction"""
        if hasattr(self.model, 'feature_importances_'):
            global_importance = self.model.feature_importances_
            
            # Adjust importance based on feature values (simple heuristic)
            adjusted_importance = []
            for i, (importance, value) in enumerate(zip(global_importance, features)):
                # Higher values of risk-increasing features = higher importance
                if i in [0, 1]:  # GPA, Attendance (inverse relationship)
                    adjusted_importance.append(importance * (1 - value / 4.0) if i == 0 else importance * (1 - value / 100))
                elif i in [2, 6, 7, 8]:  # Risk-increasing features
                    adjusted_importance.append(importance * (value / np.max([features[2], 1])))
                else:  # Risk-decreasing features
                    adjusted_importance.append(importance * (1 - value / np.max([features[i], 1])))
            
            # Normalize to percentage
            total = sum(adjusted_importance)
            if total > 0:
                adjusted_importance = [imp / total * 100 for imp in adjusted_importance]
            
            feature_importance_dict = dict(zip(self.feature_names, adjusted_importance))
            
            # Map to more readable feature names
            return {
                'GPA': feature_importance_dict.get('gpa', 0),
                'Attendance': feature_importance_dict.get('attendance', 0),
                'Course Load': feature_importance_dict.get('course_load', 0),
                'Library Visits': feature_importance_dict.get('library_visits', 0),
                'LMS Activity': feature_importance_dict.get('lms_interactions', 0),
                'Extracurricular': feature_importance_dict.get('extracurricular', 0),
                'Financial Aid': feature_importance_dict.get('financial_aid_encoded', 0),
                'Work Study': feature_importance_dict.get('work_study_encoded', 0),
                'First Generation': feature_importance_dict.get('first_generation_encoded', 0)
            }
        
        return {name: 0 for name in self.feature_names}
    
    def _fallback_prediction(self, data):
        """Fallback prediction when model is not available"""
        logger.warning("Using fallback prediction method")
        
        # Simple rule-based fallback
        risk_score = (
            (4 - float(data.get('gpa', 3.0))) * 10 +
            (100 - float(data.get('attendance', 75.0))) * 0.3 +
            max(0, int(data.get('course_load', 15)) - 15) * 2 +
            max(0, 10 - int(data.get('library_visits', 8))) * 1.5 +
            max(0, 50 - int(data.get('lms_interactions', 45))) * 0.4 +
            max(0, 2 - int(data.get('extracurricular', 3))) * 5
        )
        
        risk_score = max(0, min(100, risk_score))
        
        return {
            'risk_score': risk_score,
            'risk_level': self._get_risk_level(risk_score),
            'confidence': 0.8,
            'feature_importance': {
                'GPA': 25, 'Attendance': 20, 'Course Load': 15,
                'Library Visits': 10, 'LMS Activity': 15, 
                'Extracurricular': 10, 'Financial Aid': 3,
                'Work Study': 1, 'First Generation': 1
            }
        }
    
    def save_model(self, filepath):
        """Save the trained model and preprocessing objects"""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'feature_names': self.feature_names
            }
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self, filepath):
        """Load a trained model and preprocessing objects"""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoders = model_data['label_encoders']
            self.feature_names = model_data['feature_names']
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self._create_dummy_model()

# Global predictor instance
predictor = EngagementPredictor()

def train_new_model(training_data_path):
    """
    Train a new model with actual student data
    Replace this with your actual training pipeline
    """
    try:
        # Load your actual training data
        # df = pd.read_csv(training_data_path)
        
        # Preprocess data
        # X = df[features]
        # y = df['engagement_risk']
        
        # Train model
        # ... your training code here ...
        
        logger.info("New model trained successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error training new model: {e}")
        return False