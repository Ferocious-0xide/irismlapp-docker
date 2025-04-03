"""
Model training and serialization module.
This module contains functionality to train and save the Iris classifier model.
"""

import os
import pickle
from pathlib import Path
import logging

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
    """Load the iris dataset."""
    logger.info("Loading iris dataset")
    iris = datasets.load_iris()
    return iris.data, iris.target, iris.feature_names, iris.target_names

def train_model(X, y, model_path='data/model.pkl'):
    """
    Train the KNN model with grid search for hyperparameter tuning and save to disk.
    
    Args:
        X: Features
        y: Target labels
        model_path: Path to save the model
    
    Returns:
        Trained model
    """
    logger.info("Splitting data into train and test sets")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create a pipeline with preprocessing and model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())
    ])
    
    # Set up grid search
    param_grid = {
        'knn__n_neighbors': range(1, 21),
        'knn__weights': ['uniform', 'distance'],
        'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }
    
    logger.info("Training model with grid search")
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=5, 
        scoring='accuracy', 
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    logger.info(f"Best parameters: {grid_search.best_params_}")
    
    # Evaluate the model
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Model accuracy: {accuracy:.4f}")
    logger.info("\nClassification Report:\n" + classification_report(y_test, y_pred))
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save model
    logger.info(f"Saving model to {model_path}")
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    
    return best_model

def load_model(model_path='data/model.pkl'):
    """
    Load the trained model from disk.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded model
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    logger.info(f"Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model

if __name__ == "__main__":
    # Train and save the model
    X, y, feature_names, target_names = load_data()
    train_model(X, y)