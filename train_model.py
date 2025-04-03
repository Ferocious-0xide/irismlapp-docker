#!/usr/bin/env python3
"""
Script to train and save the Iris classifier model.
Run this script to train a new model and save it to the data directory.
"""

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
    """
    Main function to train and save the model.
    """
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