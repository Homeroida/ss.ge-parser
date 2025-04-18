"""
Prediction script for the real estate price prediction model.
"""

import os
import sys
import json
import joblib
import argparse
import pandas as pd

from config import MODEL_DIR, MODEL_FILENAME

def load_model(model_path=None):
    """
    Load the trained model.
    
    Parameters:
    -----------
    model_path : str, optional
        Path to the model file. If None, uses the default path.
        
    Returns:
    --------
    object
        Loaded model.
    """
    if model_path is None:
        # Find the most recent model file if MODEL_FILENAME contains a date pattern
        if '%' in MODEL_FILENAME:
            model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith('real_estate_model_') and f.endswith('.joblib')]
            if not model_files:
                raise FileNotFoundError(f"No model files found in {MODEL_DIR}")
            
            # Sort by date (assuming YYYYMMDD format in filename)
            model_files.sort(reverse=True)
            model_path = os.path.join(MODEL_DIR, model_files[0])
        else:
            model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from {model_path}")
    return joblib.load(model_path)

def predict_from_file(input_file, model=None, output_file=None):
    """
    Make predictions for properties in a CSV file.
    
    Parameters:
    -----------
    input_file : str
        Path to the input CSV file.
    model : object, optional
        Trained model. If None, the model is loaded from the default path.
    output_file : str, optional
        Path to the output CSV file. If None, appends '_predictions' to the input filename.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with predictions.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Load the model if not provided
    if model is None:
        model = load_model()
    
    # Load the properties from CSV
    properties_df = pd.read_csv(input_file)
    print(f"Loaded {len(properties_df)} properties from {input_file}")
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(properties_df)
    
    # Convert log predictions back to original scale
    import numpy as np
    predictions_orig = np.expm1(predictions)
    
    # Add predictions to the DataFrame
    properties_df['Predicted_Price_USD'] = predictions_orig
    
    # Save predictions if output_file is provided
    if output_file is None:
        # Create output filename from input filename
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_predictions.csv"
    
    properties_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    
    return properties_df

def predict_single_property(property_data, model=None):
    """
    Make a prediction for a single property.
    
    Parameters:
    -----------
    property_data : dict
        Dictionary containing property features.
    model : object, optional
        Trained model. If None, the model is loaded from the default path.
        
    Returns:
    --------
    float
        Predicted property price.
    """
    # Load the model if not provided
    if model is None:
        model = load_model()
    
    # Convert property data to DataFrame
    property_df = pd.DataFrame([property_data])
    
    # Make prediction
    prediction_log = model.predict(property_df)[0]
    
    # Convert back from log scale
    import numpy as np
    prediction = np.expm1(prediction_log)
    
    return prediction

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict real estate prices')
    parser.add_argument('--input', '-i', type=str, help='Input CSV file with properties')
    parser.add_argument('--output', '-o', type=str, help='Output CSV file for predictions')
    parser.add_argument('--model', '-m', type=str, help='Path to model file')
    
    args = parser.parse_args()
    
    if args.input:
        # Batch prediction from CSV file
        predict_from_file(args.input, output_file=args.output, model=load_model(args.model) if args.model else None)
    else:
        # Interactive mode for single property prediction
        print("Interactive prediction mode")
        print("Enter property details:")
        
        property_data = {}
        property_data['Area_SqM'] = float(input("Area (square meters): "))
        property_data['Bedrooms'] = int(input("Number of bedrooms: "))
        property_data['TotalRooms'] = int(input("Total number of rooms: "))
        property_data['Floor'] = int(input("Floor: "))
        property_data['TotalFloors'] = int(input("Total floors in building: "))
        property_data['District'] = input("District: ")
        property_data['Subdistrict'] = input("Subdistrict: ")
        property_data['Street'] = input("Street: ")
        
        # Add season
        month = int(input("Month (1-12): "))
        if month in [12, 1, 2]:
            property_data['ListingSeason'] = 'Winter'
        elif month in [3, 4, 5]:
            property_data['ListingSeason'] = 'Spring'
        elif month in [6, 7, 8]:
            property_data['ListingSeason'] = 'Summer'
        else:
            property_data['ListingSeason'] = 'Fall'
        
        # Make prediction
        price = predict_single_property(property_data, model=load_model(args.model) if args.model else None)
        
        print(f"\nPredicted price: ${price:.2f}")