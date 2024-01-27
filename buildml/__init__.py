"""
Machine Learning Toolkit for Python
===================================

BuildML is a comprehensive Python toolkit designed to simplify and streamline the machine learning workflow. It provides a set of tools and utilities that cover various aspects of building machine learning models.

Key Features:
- Data Exploration and Analysis: Perform exploratory data analysis to gain insights into your datasets.
- Data Preprocessing and Cleaning: Easily handle data preprocessing and cleaning tasks to ensure high-quality input for your models.
- Model Training and Prediction: Train machine learning models effortlessly and make predictions with ease.
- Regression and Classification: Support for both regression and classification tasks to address diverse machine learning needs.
- Supervised Learning: Built to support various supervised learning scenarios, making it versatile for different use cases.
- Model Evaluation: Evaluate the performance of your models using comprehensive metrics.

BuildML is built on top of popular scientific Python packages such as numpy, scipy, and matplotlib, ensuring seamless integration with the broader Python ecosystem.

Visit our documentation at https://buildml.readthedocs.io/ for detailed information on how to use BuildML and unleash the power of machine learning in your projects.
"""

from ._automate import SupervisedLearning
import date_features
import eda
import output_dataset
import model
import preprocessing

__author__ = "TechLeo"
__email__ = "techleo.ng@outlook.com"
__copyright__ = "Copyright (c) 2023 TechLeo"
__license__ = "MIT"
__version__ = "1.0.6"

__all__ = [
    'SupervisedLearning',
    'date_features',
    'eda',
    'output_dataset',
    'model',
    'preprocessing'
]
