"""
Helper functions for the real estate price prediction model.
"""

import os
import json
import joblib
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime
import inspect

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (MODEL_DIR, MODEL_FILENAME, MODEL_INFO_FILENAME, 
                   PREDICTION_FUNCTION_FILENAME, DEFAULT_VALUES, ensure_model_dir)

def save_model(model, numerical_features, categorical_features, metrics):
    """
    Save the model, metadata, and create a prediction function.
    
    Parameters:
    -----------
    model : Pipeline
        Trained model pipeline.
    numerical_features : list
        List of numerical feature names.
    categorical_features : list
        List of categorical feature names.
    metrics : dict
        Dictionary containing evaluation metrics.
        
    Returns:
    --------
    function
        Prediction function.
    """
    print("\nSaving model and related artifacts...")
    # Create model artifacts directory if it doesn't exist
    ensure_model_dir()
    
    # Save the complete model pipeline
    model_filename = os.path.join(MODEL_DIR, MODEL_FILENAME)
    joblib.dump(model, model_filename)
    print(f"Model saved to {model_filename}")
    
    # Save model metadata and performance metrics
    model_info = {
        'model_type': str(model.named_steps['model'].__class__.__name__),
        'trained_date': datetime.now().strftime('%Y-%m-%d'),
        'numerical_features': numerical_features,
        'categorical_features': categorical_features,
        'performance': {
            'rmse': float(metrics['rmse']),
            'mae': float(metrics['mae']),
            'r2': float(metrics['r2']),
            'mape': float(metrics['mape'])
        },
        'description': 'Real estate price prediction model for Georgian properties'
    }
    
    # Save model info to JSON with UTF-8 encoding
    with open(os.path.join(MODEL_DIR, MODEL_INFO_FILENAME), 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=4, ensure_ascii=False)
    print(f"Model info saved to {os.path.join(MODEL_DIR, MODEL_INFO_FILENAME)}")
    
    # Create a prediction function
    predict_function = create_prediction_function(model_filename)
    
    # Save the prediction function to a file with UTF-8 encoding
    with open(os.path.join(MODEL_DIR, PREDICTION_FUNCTION_FILENAME), 'w', encoding='utf-8') as f:
        f.write(inspect.getsource(predict_function))
        f.write("\n\n# Example usage:\n")
        f.write("# example_property = " + str(get_example_property()) + "\n")
        f.write("# predicted_price = predict_property_price(example_property)\n")
        f.write("# print(f'Predicted price: ${predicted_price:.2f}')\n")
    
    print(f"Prediction function saved to {os.path.join(MODEL_DIR, PREDICTION_FUNCTION_FILENAME)}")
    
    # Test the prediction function
    print("\nExample prediction:")
    example_property = get_example_property()
    predicted_price = predict_function(example_property)
    print(f"Property: {example_property}")
    print(f"Predicted price: ${predicted_price:.2f}")
    
    return predict_function

def create_prediction_function(model_path):
    """
    Create a prediction function for a saved model.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model pipeline.
        
    Returns:
    --------
    function
        Prediction function.
    """
    def predict_property_price(property_data, model_path=model_path):
        """
        Predict the price of a new property.
        
        Parameters:
        -----------
        property_data : dict
            A dictionary containing property features. Required fields include:
            Area_SqM, Bedrooms, TotalRooms, Floor, TotalFloors, PropertyType,
            and location information (District, Subdistrict, Street).
            
        model_path : str
            Path to the saved model pipeline.
            
        Returns:
        --------
        float
            Predicted property price in USD.
        """
        # Function to convert text to hash ID
        def text_to_hash_id(text):
            return int(hashlib.md5(str(text).encode()).hexdigest(), 16) % 10000
        
        # Load the model
        model = joblib.load(model_path)
        
        # Create a copy of property data
        data = property_data.copy()
        
        # Process location text
        if 'District' in data and 'DistrictID' not in data:
            data['DistrictID'] = text_to_hash_id(data['District'])
            del data['District']
            
        if 'Subdistrict' in data and 'SubdistrictID' not in data:
            data['SubdistrictID'] = text_to_hash_id(data['Subdistrict'])
            del data['Subdistrict']
            
        if 'Street' in data and 'StreetID' not in data:
            data['StreetID'] = text_to_hash_id(data['Street'])
            del data['Street']
        
        # Add derived features if not present
        if 'RoomDensity' not in data and 'TotalRooms' in data and 'Area_SqM' in data:
            data['RoomDensity'] = data['TotalRooms'] / data['Area_SqM']
        
        if 'BedroomRatio' not in data and 'Bedrooms' in data and 'TotalRooms' in data:
            data['BedroomRatio'] = data['Bedrooms'] / data['TotalRooms'] if data['TotalRooms'] > 0 else 0
        
        if 'FloorRatio' not in data and 'Floor' in data and 'TotalFloors' in data:
            data['FloorRatio'] = data['Floor'] / data['TotalFloors'] if data['TotalFloors'] > 0 else 0
        
        if 'IsTopFloor' not in data and 'Floor' in data and 'TotalFloors' in data:
            data['IsTopFloor'] = 1 if data['Floor'] == data['TotalFloors'] else 0
        
        if 'IsGroundFloor' not in data and 'Floor' in data:
            data['IsGroundFloor'] = 1 if data['Floor'] <= 1 else 0
        
        # Add default values for missing features
        for feature, value in DEFAULT_VALUES.items():
            if feature not in data:
                data[feature] = value
        
        # Add time-based features
        if 'DayOfWeek' not in data:
            from datetime import datetime
            current_date = datetime.now()
            data['DayOfWeek'] = current_date.weekday()
        if 'DayOfMonth' not in data:
            if 'DayOfWeek' in data:  # if we already created current_date
                data['DayOfMonth'] = current_date.day
            else:
                from datetime import datetime
                data['DayOfMonth'] = datetime.now().day
        if 'IsWeekend' not in data:
            if 'DayOfWeek' in data:
                data['IsWeekend'] = 1 if data['DayOfWeek'] >= 5 else 0
            else:
                from datetime import datetime
                data['IsWeekend'] = 1 if datetime.now().weekday() >= 5 else 0
        if 'ListingAge' not in data:
            data['ListingAge'] = 0  # Assume it's a new listing
        
        # Add transformed features
        if 'LogArea' not in data and 'Area_SqM' in data:
            import numpy as np
            data['LogArea'] = np.log1p(data['Area_SqM'])
        
        # Add premium area indicator
        if 'IsPremiumArea' not in data:
            data['IsPremiumArea'] = 0  # Default value
        
        # Convert to DataFrame
        property_df = pd.DataFrame([data])
        
        # Make prediction
        prediction_log = model.predict(property_df)[0]
        
        # Convert back from log scale
        prediction = np.expm1(prediction_log)
        
        return prediction
    
    return predict_property_price

def get_example_property():
    """
    Get an example property for prediction testing.
    
    Returns:
    --------
    dict
        Example property data.
    """
    return {
        'Area_SqM': 75,
        'Bedrooms': 2,
        'TotalRooms': 3,
        'Floor': 5,
        'TotalFloors': 9,
        'PropertyType': 'Apartment',
        'ListingYear': 2023,
        'ListingMonth': 4,
        'ListingSeason': 'Spring',
        'District': 'Vake-Saburtalo',  # Transliterated
        'Subdistrict': 'Saburtalo',    # Transliterated
        'Street': 'Nutsubidze St.'     # Transliterated
    }