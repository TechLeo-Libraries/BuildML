import pathlib
from setuptools import setup, find_packages
import codecs
import os


with open('requirements.txt', 'r') as f:
    install_requires = f.read().splitlines()


VERSION = '0.0.5'
DESCRIPTION = 'Building machine learning models has never been easier.'
LONG_DESCRIPTION = 'The ability to perform exploratory data analysis, data preprocessing, data cleaning, data transformation, data segregation, model training, model prediction, and model evaluation has been put together in this package to allow simple flow of ML operations.'
HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

# Setting up
setup(
    name="mlwizard",
    license="MIT",
    version=VERSION,
    author="TechLeo (Onyiriuba Leonard Chukwubuikem)",
    author_email="<techleo.ng@outlook.com>",
    description=DESCRIPTION,
    # long_description=README,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    # install_requires=['pandas', 'numpy', 'matplotlib', 'seaborn', 'scikit-learn'],
    install_requires=install_requires,
    keywords = ['machine learning', 'data science', 'python library', 'deep learning', 'data analysis', 'predictive modeling', 'feature engineering', 'model training', 'algorithm implementation', 'data preprocessing', 'open source', 'neural networks', 'supervised learning', 'unsupervised learning', 'model evaluation', 'ensemble learning', 'AI development', 'data exploration', 'ML framework', 'data cleaning', 'big data analytics', 'model deployment', 'natural language processing', 'computer vision', 'regression', 'classification', 'clustering', 'hyperparameter tuning', 'cross-validation', 'machine learning toolkit'],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
    zip_safe=False,
    python_requires='>=3.0',
)