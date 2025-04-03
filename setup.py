"""
Setup script for the iris-ml-api package.
"""

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