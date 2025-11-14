"""
Utility functions for model predictions and data handling.
"""

import joblib
import pickle
import numpy as np
from typing import Tuple, List, Optional
from tensorflow.keras.models import load_model


def load_rf_model(model_path: str = 'models/best_random_forest_model.pkl'):
    """Load Random Forest model."""
    return joblib.load(model_path)


def load_svm_model(model_path: str = 'models/svm_model.pkl'):
    """Load SVM model."""
    return joblib.load(model_path)


def load_cnn1d_model(model_path: str = 'models/1d_cnn_model.keras'):
    """Load 1D CNN model."""
    return load_model(model_path)


def load_cnn2d_model(model_path: str = 'models/2d_cnn_model.keras'):
    """Load 2D CNN model."""
    return load_model(model_path)


def load_label_encoder(encoder_path: str):
    """Load label encoder."""
    return joblib.load(encoder_path)


def load_class_names(class_names_path: str = 'models/class_names.pkl') -> List[str]:
    """Load class names."""
    with open(class_names_path, 'rb') as f:
        return pickle.load(f)


def predict_rf(model, features: np.ndarray) -> Tuple[str, np.ndarray]:
    """
    Make prediction using Random Forest model.
    
    Args:
        model: Random Forest model
        features: Feature vector
        
    Returns:
        Tuple of (predicted_class, probabilities)
    """
    pred_class = model.predict(features)[0]
    pred_proba = model.predict_proba(features)[0]
    return pred_class, pred_proba


def predict_svm(model, features: np.ndarray) -> Tuple[str, np.ndarray]:
    """
    Make prediction using SVM model.
    
    Args:
        model: SVM model
        features: Feature vector
        
    Returns:
        Tuple of (predicted_class, probabilities)
    """
    pred_class = model.predict(features)[0]
    pred_proba = model.predict_proba(features)[0]
    return pred_class, pred_proba


def predict_ensemble(rf_proba: np.ndarray, svm_proba: np.ndarray, 
                     cnn1d_proba: np.ndarray, cnn2d_proba: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Make ensemble prediction by averaging probabilities.
    
    Args:
        rf_proba: Random Forest probabilities
        svm_proba: SVM probabilities
        cnn1d_proba: 1D CNN probabilities
        cnn2d_proba: 2D CNN probabilities
        
    Returns:
        Tuple of (predicted_class_index, ensemble_probabilities)
    """
    ensemble_proba = (rf_proba + svm_proba + cnn1d_proba + cnn2d_proba) / 4
    pred_class = np.argmax(ensemble_proba)
    return pred_class, ensemble_proba


def get_patient_id(filename: str) -> int:
    """Extract patient ID from filename."""
    return int(filename.split('_')[0])


def get_confidence_level(probability: float) -> str:
    """
    Convert probability to confidence level description.
    
    Args:
        probability: Probability value (0-1)
        
    Returns:
        Confidence level string
    """
    if probability >= 0.9:
        return "Very High Confidence"
    elif probability >= 0.8:
        return "High Confidence"
    elif probability >= 0.7:
        return "Moderate Confidence"
    elif probability >= 0.6:
        return "Low Confidence"
    else:
        return "Very Low Confidence"


def format_prediction_output(pred_class: str, pred_proba: np.ndarray, 
                            class_names: List[str], confidence_threshold: float = 0.6) -> dict:
    """
    Format prediction output.
    
    Args:
        pred_class: Predicted class
        pred_proba: Prediction probabilities
        class_names: List of class names
        confidence_threshold: Minimum confidence for reliable prediction
        
    Returns:
        Formatted prediction dictionary
    """
    max_prob = np.max(pred_proba)
    confidence_level = get_confidence_level(max_prob)
    is_reliable = max_prob >= confidence_threshold
    
    return {
        'predicted_condition': pred_class,
        'confidence': max_prob,
        'confidence_level': confidence_level,
        'is_reliable': is_reliable,
        'probabilities': {class_names[i]: float(pred_proba[i]) for i in range(len(class_names))}
    }
