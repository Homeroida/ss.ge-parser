"""
Model evaluation functions for the real estate price prediction model.
"""

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from .model_builder import create_model_pipeline

def evaluate_model(model_name, model, X_train, X_test, y_train, y_test, preprocessor):
    """
    Evaluate a model with various metrics.
    
    Parameters:
    -----------
    model_name : str
        Name of the model.
    model : object
        Model instance.
    X_train : pandas.DataFrame
        Training features.
    X_test : pandas.DataFrame
        Testing features.
    y_train : pandas.Series
        Training target.
    y_test : pandas.Series
        Testing target.
    preprocessor : ColumnTransformer
        Preprocessing pipeline.
        
    Returns:
    --------
    dict
        Dictionary containing evaluation results.
    """
    # Create a pipeline with preprocessing and model
    pipeline = create_model_pipeline(model, preprocessor)
    
    # Record training time
    start_time = time.time()
    
    # Fit the model
    print(f"Training {model_name}...")
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    train_pred = pipeline.predict(X_train)
    test_pred = pipeline.predict(X_test)
    
    # Calculate time taken
    training_time = time.time() - start_time
    
    # Calculate metrics on log scale
    train_rmse_log = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse_log = np.sqrt(mean_squared_error(y_test, test_pred))
    train_r2_log = r2_score(y_train, train_pred)
    test_r2_log = r2_score(y_test, test_pred)
    test_mae_log = mean_absolute_error(y_test, test_pred)
    
    # Convert predictions back to original scale
    train_pred_orig = np.expm1(train_pred)
    test_pred_orig = np.expm1(test_pred)
    y_train_orig = np.expm1(y_train)
    y_test_orig = np.expm1(y_test)
    
    # Calculate metrics on original scale
    train_rmse = np.sqrt(mean_squared_error(y_train_orig, train_pred_orig))
    test_rmse = np.sqrt(mean_squared_error(y_test_orig, test_pred_orig))
    train_r2 = r2_score(y_train_orig, train_pred_orig)
    test_r2 = r2_score(y_test_orig, test_pred_orig)
    test_mae = mean_absolute_error(y_test_orig, test_pred_orig)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    test_mape = np.mean(np.abs((y_test_orig - test_pred_orig) / y_test_orig)) * 100
    
    # Get feature importance for tree-based models
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': [f"Feature_{i}" for i in range(len(model.feature_importances_))],
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
    
    print(f"  Test RMSE: ${test_rmse:.2f}")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  Test MAPE: {test_mape:.2f}%")
    print(f"  Training time: {training_time:.2f} seconds\n")
    
    return {
        'Model': model_name,
        'Training Time (s)': training_time,
        'Train RMSE (log)': train_rmse_log,
        'Test RMSE (log)': test_rmse_log,
        'Train R² (log)': train_r2_log,
        'Test R² (log)': test_r2_log,
        'Train RMSE': train_rmse,
        'Test RMSE': test_rmse,
        'Train R²': train_r2,
        'Test R²': test_r2,
        'Test MAE': test_mae,
        'Test MAPE (%)': test_mape,
        'Feature Importance': feature_importance,
        'Pipeline': pipeline
    }

def evaluate_models(models, X_train, X_test, y_train, y_test, preprocessor):
    """
    Evaluate multiple models and select the best one.
    
    Parameters:
    -----------
    models : dict
        Dictionary of model names and model instances.
    X_train : pandas.DataFrame
        Training features.
    X_test : pandas.DataFrame
        Testing features.
    y_train : pandas.Series
        Training target.
    y_test : pandas.Series
        Testing target.
    preprocessor : ColumnTransformer
        Preprocessing pipeline.
        
    Returns:
    --------
    tuple
        Tuple containing (results, best_model, best_pipeline).
    """
    print("\nEvaluating multiple models...")
    
    # Evaluate all models and collect results
    results = []
    for model_name, model in models.items():
        result = evaluate_model(model_name, model, X_train, X_test, y_train, y_test, preprocessor)
        results.append(result)
    
    # Convert results to DataFrame for easier comparison
    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'Feature Importance' and k != 'Pipeline'} 
                              for r in results])
    
    print("\nModel comparison:")
    print(results_df[['Model', 'Test RMSE', 'Test R²', 'Test MAPE (%)', 'Training Time (s)']])
    
    # Find the best model based on test R²
    best_model_idx = results_df['Test R²'].idxmax()
    best_model = results_df.iloc[best_model_idx]
    print(f"\nBest model: {best_model['Model']} with Test R² of {best_model['Test R²']:.4f}")
    
    return results, best_model, results[best_model_idx]['Pipeline']

def evaluate_final_model(model, X_test, y_test, output_dir=None):
    """
    Evaluate the final tuned model on the test set.
    
    Parameters:
    -----------
    model : Pipeline
        Trained model pipeline.
    X_test : pandas.DataFrame
        Testing features.
    y_test : pandas.Series
        Testing target.
    output_dir : str, optional
        Directory to save visualizations. Defaults to None.
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics.
    """
    print("\nEvaluating final model on test data...")
    # Make predictions with our tuned model on the test set
    test_pred = model.predict(X_test)
    test_pred_orig = np.expm1(test_pred)  # Convert back from log scale
    y_test_orig = np.expm1(y_test)
    
    # Calculate detailed metrics
    test_mse = mean_squared_error(y_test_orig, test_pred_orig)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test_orig, test_pred_orig)
    test_r2 = r2_score(y_test_orig, test_pred_orig)
    test_mape = np.mean(np.abs((y_test_orig - test_pred_orig) / y_test_orig)) * 100
    
    print("Final Model Evaluation on Test Data:")
    print(f"Root Mean Squared Error (RMSE): ${test_rmse:.2f}")
    print(f"Mean Absolute Error (MAE): ${test_mae:.2f}")
    print(f"R-squared (R²): {test_r2:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {test_mape:.2f}%")
    
    # Create visualization for actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_orig, test_pred_orig, alpha=0.5)
    plt.plot([y_test_orig.min(), y_test_orig.max()], [y_test_orig.min(), y_test_orig.max()], 'r--')
    plt.xlabel('Actual Price (USD)')
    plt.ylabel('Predicted Price (USD)')
    plt.title('Actual vs Predicted Property Prices')
    plt.tight_layout()
    
    # Save visualization if output_dir is provided
    if output_dir:
        plt.savefig(f'{output_dir}/actual_vs_predicted.png')
        print(f"Created visualization: {output_dir}/actual_vs_predicted.png")
    
    # Analyze errors by price range
    error_analysis = analyze_errors_by_price_bracket(y_test_orig, test_pred_orig)
    
    return {
        'rmse': test_rmse,
        'mae': test_mae,
        'r2': test_r2,
        'mape': test_mape,
        'error_analysis': error_analysis
    }

def analyze_errors_by_price_bracket(y_true, y_pred):
    """
    Analyze prediction errors by price bracket.
    
    Parameters:
    -----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
        
    Returns:
    --------
    pandas.DataFrame
        Error analysis by price bracket.
    """
    # Create price brackets
    price_brackets = [0, 50000, 100000, 150000, 200000, 300000, 500000, 1000000, float('inf')]
    price_labels = ['<50K', '50K-100K', '100K-150K', '150K-200K', '200K-300K', '300K-500K', '500K-1M', '>1M']
    
    # Create DataFrame with actual, predicted, and error values
    error_df = pd.DataFrame({
        'actual': y_true,
        'predicted': y_pred,
        'abs_error': np.abs(y_true - y_pred),
        'pct_error': np.abs((y_true - y_pred) / y_true) * 100
    })
    
    error_df['price_bracket'] = pd.cut(error_df['actual'], bins=price_brackets, labels=price_labels)
    
    # Calculate error metrics by price bracket
    error_by_bracket = error_df.groupby('price_bracket').agg({
        'actual': 'count',
        'abs_error': 'mean',
        'pct_error': 'mean'
    }).rename(columns={'actual': 'count', 'abs_error': 'MAE', 'pct_error': 'MAPE (%)'})
    
    print("\nError Analysis by Price Bracket:")
    print(error_by_bracket)
    
    return error_by_bracket