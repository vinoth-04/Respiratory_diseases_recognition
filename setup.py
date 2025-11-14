"""
Setup configuration for Respiratory Diseases Recognition project.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="respiratory-diseases-recognition",
    version="1.0.0",
    author="Vinoth",
    description="Machine learning project for predicting lung diseases from respiratory sound recordings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vinoth-04/Respiratory_diseases_recognition",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "librosa>=0.9.2",
        "audiomentations>=0.29.0",
        "tensorflow>=2.10.0",
        "keras>=2.10.0",
        "Flask==2.3.3",
        "Werkzeug==2.3.7",
        "streamlit>=1.28.0",
        "joblib>=1.2.0",
        "matplotlib>=3.5.0",
        "Pillow>=9.0.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "ipython>=8.0.0",
            "pytest>=7.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "lung-predict=src.lung_app:main",
        ],
    },
)
