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
   source venv/bin/activate  # On Windows: venv\Scripts\activate
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

## Acknowledgements# Trigger Deployment
