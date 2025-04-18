"""
Main training script for the real estate price prediction model.
"""

import sys
import os

from data.loader import load_and_explore_data
from data.preprocessor import preprocess_data, prepare_data_for_modeling
from data.feature_engineer import engineer_features
from models.model_builder import build_models
from models.evaluator import evaluate_models, evaluate_final_model
from models.hypertuner import tune_hyperparameters
from utils.helpers import save_model
from config import DATA_PATH, MODEL_DIR

def train_real_estate_model(data_path=DATA_PATH):
    """
    Run the entire model training pipeline.
    
    Parameters:
    -----------
    data_path : str, optional
        Path to the data file. Defaults to DATA_PATH from config.
        
    Returns:
    --------
    tuple
        Tuple containing (model, predict_function).
    """
    print("=" * 70)
    print("REAL ESTATE PRICE PREDICTION MODEL TRAINING")
    print("=" * 70)
    
    # Step 1: Load and explore data
    df = load_and_explore_data(data_path)
    
    # Step 2: Preprocess data
    df_clean = preprocess_data(df)
    
    # Step 3: Feature engineering
    df_engineered = engineer_features(df_clean)
    
    # Step 4: Prepare data for modeling
    X_train, X_test, y_train, y_test, preprocessor, numerical_features, categorical_features = prepare_data_for_modeling(df_engineered)
    
    # Step 5: Model selection
    models = build_models()
    results, best_model, best_pipeline = evaluate_models(models, X_train, X_test, y_train, y_test, preprocessor)
    
    # Step 6: Hyperparameter tuning
    tuned_model = tune_hyperparameters(best_model['Model'], best_pipeline, X_train, y_train)
    
    # Step 7: Final model evaluation
    metrics = evaluate_final_model(tuned_model, X_test, y_test, output_dir=MODEL_DIR)
    
    # Step 8: Save model and create prediction function
    predict_function = save_model(tuned_model, numerical_features, categorical_features, metrics)
    
    print("\nModel training complete!")
    print("=" * 70)
    
    return tuned_model, predict_function

if __name__ == "__main__":
    # Set the data path (default or from command line)
    data_path = DATA_PATH
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    
    # Run the training pipeline
    model, predict_function = train_real_estate_model(data_path)