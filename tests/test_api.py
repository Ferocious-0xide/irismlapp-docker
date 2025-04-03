"""
Tests for the Flask API endpoints.
"""

import json
import pytest
from flask import url_for
import numpy as np

from app.api import app
from app.models import load_data

@pytest.fixture
def client():
    """Create a test client for the app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index(client):
    """Test the index endpoint."""
    response = client.get('/')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'name' in data
    assert 'version' in data
    assert 'endpoints' in data

def test_health(client):
    """Test the health endpoint."""
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'healthy'

def test_predict_get_valid(client):
    """Test the GET predict endpoint with valid data."""
    # Values for a setosa iris
    response = client.get('/predict?sl=5.1&sw=3.5&pl=1.4&pw=0.2')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'prediction' in data
    assert 'species' in data
    assert data['species'] == 'setosa'

def test_predict_get_invalid(client):
    """Test the GET predict endpoint with invalid data."""
    response = client.get('/predict?sl=invalid&sw=3.5&pl=1.4&pw=0.2')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_predict_get_missing(client):
    """Test the GET predict endpoint with missing data."""
    response = client.get('/predict?sl=5.1&sw=3.5&pl=1.4')  # Missing pw parameter
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_predict_post_valid(client):
    """Test the POST predict endpoint with valid data."""
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
    """Test the POST predict endpoint with invalid data."""
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
    """Test the POST predict endpoint with missing data."""
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