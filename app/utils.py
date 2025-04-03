"""
Utility functions for the Iris prediction application.
"""

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
    """
    Map the class ID to the Iris species name.
    
    Args:
        class_id: Numeric ID of the Iris class
        
    Returns:
        Species name
    """
    return IRIS_SPECIES.get(class_id, "unknown")

def log_prediction(prediction: Dict[str, Any]) -> None:
    """
    Log prediction details.
    
    Args:
        prediction: Prediction data dictionary
    """
    logger.info(
        f"Prediction: {prediction['species']} (ID: {prediction['prediction']}) "
        f"for features: {prediction['features']}"
    )