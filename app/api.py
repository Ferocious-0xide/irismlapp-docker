"""
Flask API for Iris species prediction.
This module handles HTTP requests and serves model predictions.
"""

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
    """Validation model for Iris features."""
    sepal_length: float = Field(..., gt=0, description="Sepal length in cm")
    sepal_width: float = Field(..., gt=0, description="Sepal width in cm")
    petal_length: float = Field(..., gt=0, description="Petal length in cm")
    petal_width: float = Field(..., gt=0, description="Petal width in cm")

class PredictionResponse(BaseModel):
    """Validation model for prediction response."""
    prediction: int
    species: str
    features: List[float]
    probability: Dict[str, float] = None

@app.route("/")
def index():
    """Home endpoint with API information."""
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
    """Health check endpoint."""
    return jsonify({"status": "healthy"})

@app.route("/predict")
def predict_iris():
    """
    GET endpoint for Iris predictions.
    
    Query parameters:
        sl: Sepal length
        sw: Sepal width
        pl: Petal length
        pw: Petal width
    
    Returns:
        JSON with prediction results
    """
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
    """
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
    """
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