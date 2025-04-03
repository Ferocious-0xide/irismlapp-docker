"""
Tests for the machine learning model functionality.
"""

import os
import pytest
import numpy as np
from sklearn.pipeline import Pipeline

from app.models import load_data, train_model, load_model
from app.utils import get_iris_species

@pytest.fixture
def iris_data():
    """Load iris dataset for testing."""
    X, y, feature_names, target_names = load_data()
    return X, y, feature_names, target_names

def test_load_data(iris_data):
    """Test that the data loading function returns the expected shapes."""
    X, y, feature_names, target_names = iris_data
    assert X.shape[1] == 4  # 4 features
    assert len(y) == X.shape[0]  # Same number of samples and labels
    assert len(feature_names) == 4  # 4 feature names
    assert len(target_names) == 3  # 3 classes (setosa, versicolor, virginica)

def test_train_model(iris_data, tmpdir):
    """Test model training and serialization."""
    X, y, _, _ = iris_data
    
    # Create a temporary model file path
    model_path = os.path.join(tmpdir, 'model.pkl')
    
    # Train and save the model
    model = train_model(X, y, model_path=model_path)
    
    # Check that the model file exists
    assert os.path.exists(model_path)
    
    # Check that the model is a scikit-learn pipeline
    assert isinstance(model, Pipeline)
    
    # Check that the model has the expected steps
    assert 'scaler' in model.named_steps
    assert 'knn' in model.named_steps

def test_load_model(iris_data, tmpdir):
    """Test model loading functionality."""
    X, y, _, _ = iris_data
    
    # Create a temporary model file path
    model_path = os.path.join(tmpdir, 'model.pkl')
    
    # Train and save the model
    original_model = train_model(X, y, model_path=model_path)
    
    # Load the model
    loaded_model = load_model(model_path=model_path)
    
    # Check that the loaded model is a scikit-learn pipeline
    assert isinstance(loaded_model, Pipeline)
    
    # Check that the loaded model has the expected steps
    assert 'scaler' in loaded_model.named_steps
    assert 'knn' in loaded_model.named_steps
    
    # Test prediction consistency
    X_sample = X[:5]
    original_preds = original_model.predict(X_sample)
    loaded_preds = loaded_model.predict(X_sample)
    assert np.array_equal(original_preds, loaded_preds)

def test_model_prediction(iris_data, tmpdir):
    """Test model prediction functionality."""
    X, y, _, _ = iris_data
    
    # Create a temporary model file path
    model_path = os.path.join(tmpdir, 'model.pkl')
    
    # Train and save the model
    model = train_model(X, y, model_path=model_path)
    
    # Test prediction on a sample
    sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Known setosa sample
    prediction = model.predict(sample)[0]
    
    # Check prediction
    assert prediction == 0  # 0 corresponds to setosa
    assert get_iris_species(prediction) == 'setosa'
    
    # Test prediction on another sample
    sample = np.array([[7.0, 3.2, 4.7, 1.4]])  # Known versicolor sample
    prediction = model.predict(sample)[0]
    
    # Check prediction
    assert prediction == 1  # 1 corresponds to versicolor
    assert get_iris_species(prediction) == 'versicolor'