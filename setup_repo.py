#!/usr/bin/env python3
"""
Script to create the initial repository structure for the Iris ML API project.
This script will create all necessary directories and files to get started.

Run this script with:
    python setup_repo.py
"""

import os
import shutil
import json
from pathlib import Path
import textwrap
import stat

# Define the repository structure
REPO_STRUCTURE = {
    "app": {
        "__init__.py": None,
        "api.py": None,
        "models.py": None,
        "utils.py": None
    },
    "data": {
        ".gitkeep": "# This file is a placeholder to ensure the data directory is included in Git\n# The actual model.pkl file will be created when train_model.py is run"
    },
    "notebooks": {},
    "tests": {
        "__init__.py": """\"\"\"Test package initialization.\"\"\"""",
        "test_api.py": None,
        "test_model.py": None
    },
    ".github/workflows": {
        "deploy.yml": None
    },
    ".gitignore": None,
    "Dockerfile": None,
    "docker-compose.yml": None,
    "requirements.txt": None,
    "setup.py": None,
    "Procfile": "web: gunicorn app.api:app",
    "train_model.py": None,
    "run_web.bat": "@echo off\necho Starting Flask development server...\nset FLASK_APP=app.api\nset FLASK_ENV=development\nset PYTHONPATH=.\npython -m flask run --host=0.0.0.0 --port=5000",
    "run_web.sh": "#!/bin/bash\necho \"Starting Flask development server...\"\nexport FLASK_APP=app.api\nexport FLASK_ENV=development\nexport PYTHONPATH=.\npython -m flask run --host=0.0.0.0 --port=5000",
    "README.md": None
}

# Create the notebook JSON structure
NOTEBOOK_STRUCTURE = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Iris Species Classification - Model Development\n",
                "\n",
                "This notebook demonstrates the development of a machine learning model for predicting iris flower species based on their measurements. We'll use the classic Iris dataset from scikit-learn."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Import Libraries"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "from sklearn import datasets\n",
                "from sklearn.model_selection import train_test_split, GridSearchCV\n",
                "from sklearn.preprocessing import StandardScaler\n",
                "from sklearn.neighbors import KNeighborsClassifier\n",
                "from sklearn.pipeline import Pipeline\n",
                "from sklearn.metrics import classification_report, confusion_matrix, accuracy_score\n",
                "\n",
                "import pickle\n",
                "import os\n",
                "\n",
                "# Set style for plots\n",
                "sns.set(style=\"whitegrid\")\n",
                "%matplotlib inline"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Load and Explore the Data"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load the iris dataset\n",
                "iris = datasets.load_iris()\n",
                "X = iris.data\n",
                "y = iris.target\n",
                "feature_names = iris.feature_names\n",
                "target_names = iris.target_names\n",
                "\n",
                "print(f\"Features: {feature_names}\")\n",
                "print(f\"Target classes: {target_names}\")\n",
                "print(f\"Data shape: {X.shape}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Create a DataFrame for easier data exploration\n",
                "iris_df = pd.DataFrame(X, columns=feature_names)\n",
                "iris_df['species'] = pd.Categorical.from_codes(y, target_names)\n",
                "iris_df.head()"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Define file contents for required files - using triple quotes properly
FILES = {
    "app/__init__.py": """
\"\"\"
Iris prediction application initialization.
\"\"\"

import logging
from flask import Flask

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def create_app():
    \"\"\"
    Create and configure the Flask app.
    
    Returns:
        Configured Flask application
    \"\"\"
    app = Flask(__name__)
    
    # Import here to avoid circular imports
    from app.api import app as api_app
    
    # Register blueprints if we had any
    # app.register_blueprint(some_blueprint)
    
    return app
""",
    
    "app/models.py": """
\"\"\"
Model training and serialization module.
This module contains functionality to train and save the Iris classifier model.
\"\"\"

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
    \"\"\"Load the iris dataset.\"\"\"
    logger.info("Loading iris dataset")
    iris = datasets.load_iris()
    return iris.data, iris.target, iris.feature_names, iris.target_names

def train_model(X, y, model_path='data/model.pkl'):
    \"\"\"
    Train the KNN model with grid search for hyperparameter tuning and save to disk.
    
    Args:
        X: Features
        y: Target labels
        model_path: Path to save the model
    
    Returns:
        Trained model
    \"\"\"
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
    logger.info("\\nClassification Report:\\n" + classification_report(y_test, y_pred))
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save model
    logger.info(f"Saving model to {model_path}")
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    
    return best_model

def load_model(model_path='data/model.pkl'):
    \"\"\"
    Load the trained model from disk.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded model
    \"\"\"
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
""",
    
    "app/utils.py": """
\"\"\"
Utility functions for the Iris prediction application.
\"\"\"

import logging
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Iris species mapping
IRIS_SPECIES = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}

def get_iris_species(class_id: int) -> str:
    \"\"\"
    Map the class ID to the Iris species name.
    
    Args:
        class_id: Numeric ID of the Iris class
        
    Returns:
        Species name
    \"\"\"
    return IRIS_SPECIES.get(class_id, "unknown")

def log_prediction(prediction: Dict[str, Any]) -> None:
    \"\"\"
    Log prediction details.
    
    Args:
        prediction: Prediction data dictionary
    \"\"\"
    logger.info(
        f"Prediction: {prediction['species']} (ID: {prediction['prediction']}) "
        f"for features: {prediction['features']}"
    )
""",
    
    "app/api.py": """
\"\"\"
Flask API for Iris species prediction.
This module handles HTTP requests and serves model predictions.
\"\"\"

import os
import logging
from typing import Dict, Union, List

import numpy as np
from flask import Flask, request, jsonify
from pydantic import BaseModel, Field, ValidationError

from app.models import load_model
from app.utils import get_iris_species

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load the model
model = load_model()

# Pydantic models for request validation
class IrisFeatures(BaseModel):
    \"\"\"Validation model for Iris features.\"\"\"
    sepal_length: float = Field(..., gt=0, description="Sepal length in cm")
    sepal_width: float = Field(..., gt=0, description="Sepal width in cm")
    petal_length: float = Field(..., gt=0, description="Petal length in cm")
    petal_width: float = Field(..., gt=0, description="Petal width in cm")

class PredictionResponse(BaseModel):
    \"\"\"Validation model for prediction response.\"\"\"
    prediction: int
    species: str
    features: List[float]
    probability: Dict[str, float] = None

@app.route("/")
def index():
    \"\"\"Home endpoint with API information.\"\"\"
    return jsonify({
        "name": "Iris Species Predictor API",
        "version": "1.0.0",
        "description": "API for predicting Iris flower species",
        "endpoints": {
            "/predict": "GET endpoint for predictions with query parameters (sl, sw, pl, pw)",
            "/api/v1/predict": "POST endpoint for predictions with JSON body"
        },
        "health": "OK"
    })

@app.route("/health")
def health():
    \"\"\"Health check endpoint.\"\"\"
    return jsonify({"status": "healthy"})

@app.route("/predict")
def predict_iris():
    \"\"\"
    GET endpoint for Iris predictions.
    
    Query parameters:
        sl: Sepal length
        sw: Sepal width
        pl: Petal length
        pw: Petal width
    
    Returns:
        JSON with prediction results
    \"\"\"
    try:
        # Extract query parameters
        sepal_length = float(request.args.get("sl", 0))
        sepal_width = float(request.args.get("sw", 0))
        petal_length = float(request.args.get("pl", 0))
        petal_width = float(request.args.get("pw", 0))
        
        # Validate input data
        features = IrisFeatures(
            sepal_length=sepal_length,
            sepal_width=sepal_width,
            petal_length=petal_length,
            petal_width=petal_width
        )
        
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid input. All features must be numeric."}), 400
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400
    
    # Make prediction
    feature_array = np.array([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]])
    
    prediction = int(model.predict(feature_array)[0])
    species = get_iris_species(prediction)
    
    # Get prediction probabilities if available
    probabilities = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(feature_array)[0]
        for i, prob in enumerate(probs):
            probabilities[get_iris_species(i)] = float(prob)
    
    # Prepare response
    response = PredictionResponse(
        prediction=prediction,
        species=species,
        features=feature_array[0].tolist(),
        probability=probabilities if probabilities else None
    )
    
    logger.info(f"Prediction made: {response.species} for features {feature_array[0]}")
    return jsonify(response.dict())

@app.route("/api/v1/predict", methods=["POST"])
def predict_iris_json():
    \"\"\"
    POST endpoint for Iris predictions with JSON body.
    
    Request body:
        {
            "sepal_length": float,
            "sepal_width": float,
            "petal_length": float,
            "petal_width": float
        }
    
    Returns:
        JSON with prediction results
    \"\"\"
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        
        # Validate input data
        features = IrisFeatures(**data)
        
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400
    
    # Make prediction
    feature_array = np.array([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]])
    
    prediction = int(model.predict(feature_array)[0])
    species = get_iris_species(prediction)
    
    # Get prediction probabilities if available
    probabilities = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(feature_array)[0]
        for i, prob in enumerate(probs):
            probabilities[get_iris_species(i)] = float(prob)
    
    # Prepare response
    response = PredictionResponse(
        prediction=prediction,
        species=species,
        features=feature_array[0].tolist(),
        probability=probabilities if probabilities else None
    )
    
    logger.info(f"Prediction made: {response.species} for features {feature_array[0]}")
    return jsonify(response.dict())

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
""",
    
    "tests/test_api.py": """
\"\"\"
Tests for the Flask API endpoints.
\"\"\"

import json
import pytest
from flask import url_for
import numpy as np

from app.api import app
from app.models import load_data

@pytest.fixture
def client():
    \"\"\"Create a test client for the app.\"\"\"
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index(client):
    \"\"\"Test the index endpoint.\"\"\"
    response = client.get('/')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'name' in data
    assert 'version' in data
    assert 'endpoints' in data

def test_health(client):
    \"\"\"Test the health endpoint.\"\"\"
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'healthy'

def test_predict_get_valid(client):
    \"\"\"Test the GET predict endpoint with valid data.\"\"\"
    # Values for a setosa iris
    response = client.get('/predict?sl=5.1&sw=3.5&pl=1.4&pw=0.2')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'prediction' in data
    assert 'species' in data
    assert data['species'] == 'setosa'

def test_predict_get_invalid(client):
    \"\"\"Test the GET predict endpoint with invalid data.\"\"\"
    response = client.get('/predict?sl=invalid&sw=3.5&pl=1.4&pw=0.2')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_predict_get_missing(client):
    \"\"\"Test the GET predict endpoint with missing data.\"\"\"
    response = client.get('/predict?sl=5.1&sw=3.5&pl=1.4')  # Missing pw parameter
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_predict_post_valid(client):
    \"\"\"Test the POST predict endpoint with valid data.\"\"\"
    response = client.post(
        '/api/v1/predict',
        data=json.dumps({
            'sepal_length': 5.1,
            'sepal_width': 3.5,
            'petal_length': 1.4,
            'petal_width': 0.2
        }),
        content_type='application/json'
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'prediction' in data
    assert 'species' in data
    assert data['species'] == 'setosa'

def test_predict_post_invalid(client):
    \"\"\"Test the POST predict endpoint with invalid data.\"\"\"
    response = client.post(
        '/api/v1/predict',
        data=json.dumps({
            'sepal_length': 'invalid',
            'sepal_width': 3.5,
            'petal_length': 1.4,
            'petal_width': 0.2
        }),
        content_type='application/json'
    )
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_predict_post_missing(client):
    \"\"\"Test the POST predict endpoint with missing data.\"\"\"
    response = client.post(
        '/api/v1/predict',
        data=json.dumps({
            'sepal_length': 5.1,
            'sepal_width': 3.5,
            'petal_length': 1.4
            # Missing petal_width
        }),
        content_type='application/json'
    )
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data
""",
    
    "tests/test_model.py": """
\"\"\"
Tests for the machine learning model functionality.
\"\"\"

import os
import pytest
import numpy as np
from sklearn.pipeline import Pipeline

from app.models import load_data, train_model, load_model
from app.utils import get_iris_species

@pytest.fixture
def iris_data():
    \"\"\"Load iris dataset for testing.\"\"\"
    X, y, feature_names, target_names = load_data()
    return X, y, feature_names, target_names

def test_load_data(iris_data):
    \"\"\"Test that the data loading function returns the expected shapes.\"\"\"
    X, y, feature_names, target_names = iris_data
    assert X.shape[1] == 4  # 4 features
    assert len(y) == X.shape[0]  # Same number of samples and labels
    assert len(feature_names) == 4  # 4 feature names
    assert len(target_names) == 3  # 3 classes (setosa, versicolor, virginica)

def test_train_model(iris_data, tmpdir):
    \"\"\"Test model training and serialization.\"\"\"
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
    \"\"\"Test model loading functionality.\"\"\"
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
    \"\"\"Test model prediction functionality.\"\"\"
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
""",
    
    "train_model.py": """
#!/usr/bin/env python3
\"\"\"
Script to train and save the Iris classifier model.
Run this script to train a new model and save it to the data directory.
\"\"\"

import os
import logging

from app.models import load_data, train_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    \"\"\"
    Main function to train and save the model.
    \"\"\"
    logger.info("Starting model training process")
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Load the data
    X, y, feature_names, target_names = load_data()
    logger.info(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Features: {feature_names}")
    logger.info(f"Target classes: {target_names}")
    
    # Train the model
    model = train_model(X, y, model_path='data/model.pkl')
    logger.info("Model training completed and saved to data/model.pkl")
    
    return model

if __name__ == "__main__":
    main()
""",
    
    "Dockerfile": """
# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1 \\
    PIP_NO_CACHE_DIR=off \\
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Install system dependencies
RUN apt-get update \\
    && apt-get install -y --no-install-recommends gcc \\
    && apt-get clean \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 5000

# Set environment variable for Flask
ENV FLASK_APP=app/api.py

# Run the application
CMD gunicorn --bind 0.0.0.0:$PORT app.api:app
""",
    
    "docker-compose.yml": """
version: '3.8'

services:
  api:
    build: .
    image: iris-ml-api
    ports:
      - "5000:5000"
    environment:
      - PORT=5000
      - FLASK_APP=app/api.py
      - FLASK_ENV=development
    volumes:
      - .:/app
    command: gunicorn --bind 0.0.0.0:5000 --workers 1 --threads 8 --timeout 0 app.api:app
""",
    
    "requirements.txt": """
# Web Framework
flask==2.2.5
gunicorn==21.2.0
pydantic==1.10.13

# Machine Learning
scikit-learn==1.3.0
numpy==1.24.3
scipy==1.11.3

# Utilities
python-dotenv==1.0.0
pytest==7.4.0
pytest-cov==4.1.0
""",
    
    "setup.py": """
\"\"\"
Setup script for the iris-ml-api package.
\"\"\"

from setuptools import setup, find_packages

setup(
    name="iris-ml-api",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "flask>=2.2.0",
        "gunicorn>=21.0.0",
        "pydantic>=1.8.0",
        "scikit-learn>=1.0.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "python-dotenv>=0.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
        ],
    },
    python_requires=">=3.8",
    author="Your Name",
    author_email="your.email@example.com",
    description="A containerized machine learning API for Iris species prediction",
    keywords="machine learning, docker, heroku, flask, api",
    url="https://github.com/yourusername/iris-ml-api",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
""",
    
    ".github/workflows/deploy.yml": """
name: Deploy to Heroku

on:
  push:
    branches:
      - main

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest pytest-cov

      - name: Run tests
        run: |
          pytest tests/ --cov=app

      - name: Login to Heroku Container Registry
        env:
          HEROKU_API_KEY: ${{ secrets.HEROKU_API_KEY }}
        run: |
          heroku container:login

      - name: Build and push Docker image
        env:
          HEROKU_API_KEY: ${{ secrets.HEROKU_API_KEY }}
        run: |
          heroku container:push web --app ${{ secrets.HEROKU_APP_NAME }}

      - name: Release to Heroku
        env:
          HEROKU_API_KEY: ${{ secrets.HEROKU_API_KEY }}
        run: |
          heroku container:release web --app ${{ secrets.HEROKU_APP_NAME }}
""",

    "README.md": """
# Iris Machine Learning API

A modern containerized machine learning API that predicts Iris flower species based on flower measurements.

## Features

- 🌸 Predicts Iris flower species using Scikit-learn
- 🐳 Containerized with Docker for consistent deployment
- 🚀 Ready for deployment to Heroku
- 🔄 CI/CD with GitHub Actions
- 📊 Enhanced model with hyperparameter tuning
- ✅ Input validation with Pydantic
- 📝 Comprehensive API documentation

## Project Structure

```
iris-ml-api/
├── .github/workflows/     # CI/CD pipeline definitions
├── app/                   # Application code
├── data/                  # Model storage
├── notebooks/             # Development notebooks
├── tests/                 # Test suite
├── Dockerfile             # Container definition
├── docker-compose.yml     # Local development setup
├── requirements.txt       # Python dependencies
└── Procfile               # Heroku deployment configuration
```

## Quick Start

### Prerequisites

- Python 3.8+
- Docker and Docker Compose
- Git

### Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/iris-ml-api.git
   cd iris-ml-api
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   pip install -r requirements.txt
   ```

3. Train the model:
   ```bash
   python -m app.models
   ```

4. Run the API locally:
   ```bash
   python -m app.api
   ```

5. Test the API:
   ```bash
   curl "http://localhost:5000/predict?sl=5.1&sw=3.5&pl=1.4&pw=0.2"
   ```

### Using Docker

1. Build and run with Docker Compose:
   ```bash
   docker-compose up --build
   ```

2. The API will be available at `http://localhost:5000`

## Deploying to Heroku

### Manual Deployment

1. Install the Heroku CLI and log in:
   ```bash
   heroku login
   heroku container:login
   ```

2. Create a Heroku app:
   ```bash
   heroku create your-app-name
   ```

3. Build and push the container:
   ```bash
   heroku container:push web --app your-app-name
   ```

4. Release the container:
   ```bash
   heroku container:release web --app your-app-name
   ```

### Automatic Deployment with GitHub Actions

1. Fork this repository

2. In your GitHub repository settings, add the following secrets:
   - `HEROKU_API_KEY`: Your Heroku API key
   - `HEROKU_APP_NAME`: Your Heroku app name

3. Push changes to the main branch to trigger automatic deployment

## API Usage

### GET Endpoint

```
GET /predict?sl=5.1&sw=3.5&pl=1.4&pw=0.2
```

Query Parameters:
- `sl`: Sepal length in cm
- `sw`: Sepal width in cm
- `pl`: Petal length in cm
- `pw`: Petal width in cm

### POST Endpoint

```
POST /api/v1/predict
```

Request Body:
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

### Response Format

```json
{
  "prediction": 0,
  "species": "setosa",
  "features": [5.1, 3.5, 1.4, 0.2],
  "probability": {
    "setosa": 1.0,
    "versicolor": 0.0,
    "virginica": 0.0
  }
}
```

## License

MIT

## Acknowledgements

This project is based on the concepts from the article "Build and Deploy a Docker Containerized Python Machine Learning Model On Heroku" by Piyush Singhal, with significant modernizations and enhancements.
"""
}