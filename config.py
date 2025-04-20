"""
Configuration settings for the real estate price prediction model.
"""

import os
from datetime import datetime

# Data settings
DATA_PATH = "property_prediction_data_final.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42


# Features
CATEGORICAL_FEATURES = ['ListingSeason']
TARGET_COLUMN = 'Price_USD'
LOG_TARGET_COLUMN = 'LogPrice'
ID_COLUMN = 'ID'

# Columns to drop
COLS_TO_DROP_MODELING = ['Price_USD', 'LogPrice', 'PricePerSqm_USD', 'LogPricePerSqm', 
                        'ListingDate', 'District', 'Subdistrict', 'Street']

# Preprocessing
OUTLIER_PERCENTILE = 0.99

# Model settings
MODELS_CONFIG = {
    'Linear Regression': {
        'model': 'LinearRegression',
        'params': {}
    },
    'Ridge Regression': {
        'model': 'Ridge',
        'params': {'alpha': 1.0}
    },
    'Lasso Regression': {
        'model': 'Lasso',
        'params': {'alpha': 0.001}
    },
    'Elastic Net': {
        'model': 'ElasticNet',
        'params': {'alpha': 0.1, 'l1_ratio': 0.5}
    },
    'Random Forest': {
        'model': 'RandomForestRegressor',
        'params': {'n_estimators': 100, 'random_state': RANDOM_STATE}
    },
    'Gradient Boosting': {
        'model': 'GradientBoostingRegressor',
        'params': {'n_estimators': 100, 'random_state': RANDOM_STATE}
    },
    'XGBoost': {
        'model': 'XGBRegressor',
        'params': {'n_estimators': 100, 'random_state': RANDOM_STATE}
    },
    'LightGBM': {
        'model': 'LGBMRegressor',
        'params': {'n_estimators': 100, 'random_state': RANDOM_STATE}
    }
}

# Hyperparameter tuning
HYPERPARAMETER_GRIDS = {
    'Linear Regression': {},  # No hyperparameters to tune
    
    'Ridge Regression': {
        'model__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
    },
    
    'Lasso Regression': {
        'model__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0]
    },
    
    'Elastic Net': {
        'model__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
        'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
    },
    
    'Random Forest': {
        'model__n_estimators': [100, 200],
        'model__max_depth': [None, 20],
        'model__min_samples_split': [2, 10],
        'model__min_samples_leaf': [1]
    },
    
    'Gradient Boosting': {
        'model__n_estimators': [100, 200],
        'model__learning_rate': [0.01, 0.1],
        'model__max_depth': [3, 7]
    },
    
    'XGBoost': {
        'model__n_estimators': [100, 200],
        'model__learning_rate': [0.01, 0.1],
        'model__max_depth': [3, 7],
        'model__subsample': [0.8, 1.0],
        'model__colsample_bytree': [0.8, 1.0]
    },
    
    'LightGBM': {
        'model__n_estimators': [100, 200],
        'model__learning_rate': [0.01, 0.1],
        'model__num_leaves': [31, 63],
        'model__max_depth': [3, 7]
    }
}

# Paths and directories
MODEL_DIR = 'model_artifacts'
MODEL_FILENAME = f"real_estate_model_{datetime.now().strftime('%Y%m%d')}.joblib"
MODEL_INFO_FILENAME = 'model_info.json'
PREDICTION_FUNCTION_FILENAME = 'predict_function.py'

# Default values for missing features in prediction
DEFAULT_VALUES = {
    'DistrictAvgPrice': 1700,
    'DistrictMedianPrice': 1550,
    'DistrictPriceStd': 500,
    'SubdistrictAvgPrice': 1700,
    'SubdistrictMedianPrice': 1550,
    'SubdistrictPriceStd': 500,
    'LowPriceProperty': 0
}

# Ensure model directory exists
def ensure_model_dir():
    """Create model directory if it doesn't exist."""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)