"""
Prediction script for the real estate price prediction model.
"""

import os
import sys
import json
import joblib
import argparse
import pandas as pd
import numpy as np
import hashlib
from datetime import datetime

from config import MODEL_DIR, MODEL_FILENAME, DEFAULT_VALUES

def text_to_hash_id(text):
    """
    Convert text to a stable hash ID.
    
    Parameters:
    -----------
    text : str
        Text to convert to a hash ID.
        
    Returns:
    --------
    int
        Hash ID.
    """
    return int(hashlib.md5(str(text).encode()).hexdigest(), 16) % 10000  # Limit to 10000 unique values

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

def prepare_property_features(property_data):
    """
    Prepare property features for prediction by applying the same transformations
    that were used during training.
    
    Parameters:
    -----------
    property_data : dict or pandas.DataFrame
        Property data to prepare for prediction.
        
    Returns:
    --------
    pandas.DataFrame
        Prepared property data.
    """
    # Convert to DataFrame if it's a dict
    if isinstance(property_data, dict):
        df = pd.DataFrame([property_data])
    else:
        df = property_data.copy()
    
    # 1. Convert location text to hash IDs
    if 'District' in df.columns and 'DistrictID' not in df.columns:
        df['DistrictID'] = df['District'].apply(text_to_hash_id)
    
    if 'Subdistrict' in df.columns and 'SubdistrictID' not in df.columns:
        df['SubdistrictID'] = df['Subdistrict'].apply(text_to_hash_id)
    
    if 'Street' in df.columns and 'StreetID' not in df.columns:
        df['StreetID'] = df['Street'].apply(text_to_hash_id)
    
    # 2. Create room density (rooms per area)
    if 'RoomDensity' not in df.columns and 'TotalRooms' in df.columns and 'Area_SqM' in df.columns:
        df['RoomDensity'] = df['TotalRooms'] / df['Area_SqM']
    
    # 3. Create bedroom ratio (bedrooms as a proportion of total rooms)
    if 'BedroomRatio' not in df.columns and 'Bedrooms' in df.columns and 'TotalRooms' in df.columns:
        df['BedroomRatio'] = df['Bedrooms'] / df['TotalRooms']
        df['BedroomRatio'] = df['BedroomRatio'].fillna(0)  # Handle division by zero
    
    # 4. Create floor position indicator
    if 'FloorRatio' not in df.columns and 'Floor' in df.columns and 'TotalFloors' in df.columns:
        df['FloorRatio'] = df['Floor'] / df['TotalFloors']
        df['FloorRatio'] = df['FloorRatio'].fillna(0)  # Handle division by zero
    
    # 5. Create is top floor indicator
    if 'IsTopFloor' not in df.columns and 'Floor' in df.columns and 'TotalFloors' in df.columns:
        df['IsTopFloor'] = (df['Floor'] == df['TotalFloors']).astype(int)
    
    # 6. Create is ground floor indicator
    if 'IsGroundFloor' not in df.columns and 'Floor' in df.columns:
        df['IsGroundFloor'] = (df['Floor'] <= 1).astype(int)
    
    # 7. Create premium area indicator (default to 0 since we can't calculate without the full dataset)
    if 'IsPremiumArea' not in df.columns:
        df['IsPremiumArea'] = 0
    
    # 8. Create time-based features if ListingDate is present
    if 'ListingDate' in df.columns:
        df['ListingDate'] = pd.to_datetime(df['ListingDate'])
        df['DayOfWeek'] = df['ListingDate'].dt.dayofweek
        df['DayOfMonth'] = df['ListingDate'].dt.day
        df['IsWeekend'] = ((df['DayOfWeek'] >= 5) & (df['DayOfWeek'] <= 6)).astype(int)
        df['ListingYear'] = df['ListingDate'].dt.year
        df['ListingMonth'] = df['ListingDate'].dt.month
        
        # ListingAge (assume today is the reference date)
        last_date = datetime.now()
        df['ListingAge'] = (last_date - df['ListingDate']).dt.days
    else:
        # If ListingDate is not present, add defaults
        current_date = datetime.now()
        df['DayOfWeek'] = current_date.weekday()
        df['DayOfMonth'] = current_date.day
        df['IsWeekend'] = 1 if current_date.weekday() >= 5 else 0
        df['ListingAge'] = 0  # Assume it's a new listing
        
        # Add missing ListingYear and ListingMonth
        if 'ListingYear' not in df.columns:
            df['ListingYear'] = current_date.year
        if 'ListingMonth' not in df.columns:
            df['ListingMonth'] = current_date.month
    
    # 9. Apply log transformation to numerical features
    if 'LogArea' not in df.columns and 'Area_SqM' in df.columns:
        df['LogArea'] = np.log1p(df['Area_SqM'])
    
    # 10. Add default price statistics
    for feature, value in DEFAULT_VALUES.items():
        if feature not in df.columns:
            df[feature] = value
    
    # 11. Drop columns that were excluded during training
    columns_to_drop = ['Price_USD', 'LogPrice', 'PricePerSqm_USD', 'LogPricePerSqm', 
                        'ListingDate', 'District', 'Subdistrict', 'Street']
    df = df.drop([col for col in columns_to_drop if col in df.columns], axis=1)
    
    return df

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
    
    # Prepare features for prediction
    prepared_df = prepare_property_features(properties_df)
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(prepared_df)
    
    # Convert log predictions back to original scale
    predictions_orig = np.expm1(predictions)
    
    # Add predictions to the original DataFrame
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
    
    # Prepare features for prediction
    prepared_df = prepare_property_features(property_data)
    
    # Make prediction
    prediction_log = model.predict(prepared_df)[0]
    
    # Convert back from log scale
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
        
        # Add month and year
        month = int(input("Month (1-12): "))
        property_data['ListingMonth'] = month
        property_data['ListingYear'] = datetime.now().year
        
        # Add season based on month
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