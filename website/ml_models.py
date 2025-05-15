import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os

class MLModel:
    def __init__(self):
        self.model = LogisticRegression()
        self.scaler = StandardScaler()
        
    def preprocess_data(self, X, y=None, training=True):
        """Preprocess the input data"""
        # Handle missing values
        X = pd.DataFrame(X).fillna(0)
        
        if training:
            # Fit and transform the scaler on training data
            X_scaled = self.scaler.fit_transform(X)
            return X_scaled, y
        else:
            # Only transform for prediction data
            X_scaled = self.scaler.transform(X)
            return X_scaled

    def train_model(self, X, y):
        """Train the logistic regression model"""
        # Preprocess the data
        X_processed, y = self.preprocess_data(X, y, training=True)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        return metrics
    
    def predict(self, X):
        """Generate predictions for new data"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Preprocess the data
        X_processed = self.preprocess_data(X, training=False)
        
        # Generate predictions
        prediction = self.model.predict(X_processed)
        probability = self.model.predict_proba(X_processed)[:, 1]
        
        return prediction, probability
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save both the model and the scaler
        model_data = {
            'model': self.model,
            'scaler': self.scaler
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler'] 