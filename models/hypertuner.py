"""
Hyperparameter tuning functions for the real estate price prediction model.
"""

import time
from sklearn.model_selection import GridSearchCV

import sys
import os

# Add the project root to the path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HYPERPARAMETER_GRIDS

def tune_hyperparameters(best_model_name, best_pipeline, X_train, y_train):
    """
    Tune the hyperparameters of the best model.
    
    Parameters:
    -----------
    best_model_name : str
        Name of the best model.
    best_pipeline : Pipeline
        Pipeline of the best model.
    X_train : pandas.DataFrame
        Training features.
    y_train : pandas.Series
        Training target.
        
    Returns:
    --------
    object
        Tuned model pipeline.
    """
    print(f"\nTuning hyperparameters for: {best_model_name}")
    
    # Try to import tqdm for progress bars
    try:
        from tqdm.auto import tqdm
        tqdm_available = True
    except ImportError:
        print("tqdm not found - install with 'pip install tqdm' for progress bars")
        tqdm_available = False
    
    # Select the parameter grid for the best model
    if best_model_name in HYPERPARAMETER_GRIDS:
        param_grid = HYPERPARAMETER_GRIDS[best_model_name]
        
        if param_grid:  # Skip if empty (e.g., for Linear Regression)
            # Calculate total combinations
            total_combinations = 1
            for param_values in param_grid.values():
                total_combinations *= len(param_values)
            
            print(f"Testing {total_combinations} hyperparameter combinations")
            print(f"With 5-fold cross-validation, will train {total_combinations * 5} models")
            
            # Estimate time based on model type
            if best_model_name == 'Random Forest':
                est_time_per_model = 18.5  # seconds
            elif best_model_name == 'XGBoost':
                est_time_per_model = 0.3
            else:
                est_time_per_model = 1.0
                
            est_total_time = (total_combinations * 5 * est_time_per_model) / 60  # in minutes
            print(f"Estimated time: {est_total_time:.1f} minutes")
            
            # Manual progress tracking
            print("\nProgress:")
            start_time = time.time()
            last_update = start_time
            
            # Use pre_dispatch to limit memory usage
            grid = GridSearchCV(
                estimator=best_pipeline,
                param_grid=param_grid,
                cv=5,  # 5-fold cross-validation
                scoring='neg_root_mean_squared_error',  # Optimize for RMSE
                n_jobs=-1,  # Use all available cores
                verbose=1,
                pre_dispatch='2*n_jobs',  # To avoid memory issues
                return_train_score=False  # To save memory
            )
            
            try:
                # Fit the grid search
                grid_result = grid.fit(X_train, y_train)
                
                # Print results
                print(f"\nBest parameters: {grid_result.best_params_}")
                print(f"Best RMSE: {-grid_result.best_score_:.4f}")
                
                return grid_result.best_estimator_
                
            except Exception as e:
                print(f"\nError during hyperparameter tuning: {str(e)}")
                print("Falling back to the original model without tuning.")
                return best_pipeline
                
        else:
            print(f"No parameter grid defined for {best_model_name}, skipping tuning")
            return best_pipeline
    else:
        print(f"No parameter grid defined for {best_model_name}, skipping tuning")
        return best_pipeline