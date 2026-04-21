"""
Configuration file for the Fraud Detection project.
"""

# Data configuration
DATA_CONFIG = {
    'filepath': 'creditcard.csv',
    'sample_size': 50000,
    'random_state': 42,
    'test_size': 0.2,
    'target_column': 'Class'
}

# Model configuration
MODEL_CONFIG = {
    'logistic_regression': {
        'max_iter': 1000,
        'random_state': 42
    },
    'random_forest': {
        'n_estimators': 100,
        'random_state': 42,
        'n_jobs': -1
    }
}

# Visualization configuration
VISUALIZATION_CONFIG = {
    'figsize_default': (10, 6),
    'figsize_large': (12, 8),
    'dpi': 100,
    'style': 'seaborn-v0_8-darkgrid'
}

# Output configuration
OUTPUT_CONFIG = {
    'save_models': False,
    'models_dir': 'models/',
    'results_dir': 'results/'
}
