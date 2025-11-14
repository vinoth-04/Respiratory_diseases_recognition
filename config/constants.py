"""
Configuration constants for the project.
"""

# Audio Processing
SAMPLE_RATE = 22050
AUDIO_DURATION = 4  # seconds
N_MFCC = 13

# Feature Extraction
MFCC_FEATURES_SIZE = 52  # 13 * 4 (mean, std, min, max)
MFCC_SEQUENCE_LENGTH = 100
MEL_SPECTROGRAM_SIZE = (128, 128, 1)

# Model Paths
MODEL_PATHS = {
    'random_forest': 'models/best_random_forest_model.pkl',
    'svm': 'models/svm_model.pkl',
    '1d_cnn': 'models/1d_cnn_model.keras',
    '2d_cnn': 'models/2d_cnn_model.keras',
    'label_encoder_1d': 'models/label_encoder_1dcnn.pkl',
    'label_encoder_2d': 'models/label_encoder_2dcnn.pkl',
    'class_names': 'models/class_names.pkl'
}

# Classes
DISEASE_CLASSES = [
    'Healthy',
    'COPD',
    'Asthma',
    'Bronchiectasis',
    'URTI',
    'LRTI',
    'Pneumonia'
]

# Thresholds
CONFIDENCE_THRESHOLD = 0.6
MIN_PROBABILITY_FOR_PREDICTION = 0.5

# Data Augmentation
AUGMENTATION_PARAMS = {
    'noise_amplitude': (0.001, 0.015),
    'pitch_shift_semitones': (-1.5, 1.5),
    'time_stretch_rate': (0.9, 1.1),
    'augmentation_factor': 3  # Times to augment rare classes
}

# Training Parameters
TRAIN_TEST_SPLIT = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
