import pathlib
from setuptools import setup, find_packages
import codecs
import os


with open('requirements.txt', 'r') as f:
    install_requires = f.read().splitlines()


VERSION = '1.0.9'
DESCRIPTION = "Let's make building machine learning models the complex way, easy."
LONG_DESCRIPTION = 'The ability to perform exploratory data analysis, data preprocessing, data cleaning, data transformation, data segregation, model training, model prediction, and model evaluation has been put together in this package to allow simple flow of ML operations.'
HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text(encoding='utf-8')

# Setting up
setup(
    name="buildml",
    license="MIT",
    version=VERSION,
    author="TechLeo (Onyiriuba Leonard Chukwubuikem)",
    author_email="<techleo.ng@outlook.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description = (HERE / "README.md").read_text(encoding='utf-8'),
    packages=find_packages(),
    install_requires=install_requires,
    keywords = ['machine learning', 'data science', 'data preprocessing', 'supervised learning', 'data exploration', 'ML framework', 'data cleaning', 'regression', 'classification', 'machine learning toolkit'],
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