#!/bin/bash
# Simple setup script to create directory structure for Iris Docker ML app

# Create the main directories
mkdir -p app data notebooks tests

# Create app modules
touch app/__init__.py
touch app/api.py
touch app/models.py
touch app/utils.py

# Create placeholder for model storage
touch data/.gitkeep

# Create Dockerfile and related files
touch Dockerfile
touch docker-compose.yml
touch requirements.txt
touch Procfile

# Create executable scripts
touch train_model.py
touch run_web.sh
chmod +x train_model.py run_web.sh

# Create empty test files
touch tests/__init__.py
touch tests/test_api.py
touch tests/test_model.py

# Create README
touch README.md

echo "App structure created successfully in current directory."
echo "Directory structure:"
find . -type f | sort