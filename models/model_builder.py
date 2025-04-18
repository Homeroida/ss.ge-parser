"""
Model building functions for the real estate price prediction model.
"""

import pandas as pd
import numpy as np
import time
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb

import sys
import os

# Add the project root to the path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS_CONFIG, RANDOM_STATE

def build_models():
    """
    Build multiple regression models for evaluation.
    
    Returns:
    --------
    dict
        Dictionary of model names and initialized model objects.
    """
    # Initialize models
    models = {}
    
    for name, config in MODELS_CONFIG.items():
        models[name] = create_model(config['model'], config['params'])
    
    return models

def create_model(model_type, params=None):
    """
    Create a model instance based on model type and parameters.
    
    Parameters:
    -----------
    model_type : str
        Type of model to create.
    params : dict, optional
        Parameters for the model. Defaults to None.
        
    Returns:
    --------
    object
        Model instance.
    """
    if params is None:
        params = {}
    
    # Dictionary mapping model types to their classes
    model_classes = {
        'LinearRegression': LinearRegression,
        'Ridge': Ridge,
        'Lasso': Lasso,
        'ElasticNet': ElasticNet,
        'RandomForestRegressor': RandomForestRegressor,
        'GradientBoostingRegressor': GradientBoostingRegressor,
        'XGBRegressor': xgb.XGBRegressor,
        'LGBMRegressor': lgb.LGBMRegressor
    }
    
    # Create and return the model instance
    if model_type in model_classes:
        # Add random_state to params for models that support it
        if 'random_state' not in params and hasattr(model_classes[model_type](), 'random_state'):
            params['random_state'] = RANDOM_STATE
        
        return model_classes[model_type](**params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def create_model_pipeline(model, preprocessor):
    """
    Create a pipeline with preprocessing and model.
    
    Parameters:
    -----------
    model : object
        Model instance.
    preprocessor : ColumnTransformer
        Preprocessing pipeline.
        
    Returns:
    --------
    Pipeline
        Scikit-learn pipeline with preprocessor and model.
    """
    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])