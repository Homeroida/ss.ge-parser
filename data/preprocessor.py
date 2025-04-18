"""
Data preprocessing functions for the real estate price prediction model.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import sys
import os

# Add the project root to the path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (COLS_TO_DROP_MODELING, TEST_SIZE, RANDOM_STATE, 
                   OUTLIER_PERCENTILE, LOG_TARGET_COLUMN)

def preprocess_data(df):
    """
    Clean the data and handle missing values and outliers.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data to preprocess.
        
    Returns:
    --------
    pandas.DataFrame
        Preprocessed data.
    """
    print("\nPreprocessing data...")
    # Create a copy to avoid modifying the original data
    df_clean = df.copy()
    
    # Convert ListingDate to datetime
    df_clean['ListingDate'] = pd.to_datetime(df_clean['ListingDate'])
    
    # Handle missing values
    print("Handling missing values...")
    # For District (relatively few missing values)
    df_clean['District'].fillna(df_clean['Subdistrict'], inplace=True)  # Use subdistrict as a proxy
    
    # For Street (very few missing values)
    df_clean['Street'].fillna('Unknown', inplace=True)
    
    # For StreetNumber (many missing values)
    # Since ~74% are missing, we'll drop this column as it's unlikely to be useful
    if 'StreetNumber' in df_clean.columns:
        df_clean.drop('StreetNumber', axis=1, inplace=True)
    
    # For Floor and TotalFloors (just 1 missing value each)
    df_clean['Floor'].fillna(df_clean['Floor'].median(), inplace=True)
    df_clean['TotalFloors'].fillna(df_clean['TotalFloors'].median(), inplace=True)
    
    # Check for duplicates
    duplicate_count = df_clean.duplicated().sum()
    print(f"Found {duplicate_count} duplicate rows")
    
    # Remove duplicates if any
    if duplicate_count > 0:
        df_clean = df_clean.drop_duplicates()
        
    # Drop PropertyType column if present
    if 'PropertyType' in df_clean.columns:
        print("Dropping PropertyType column as requested")
        df_clean = df_clean.drop('PropertyType', axis=1)    
    
    # Handle outliers in key numerical columns
    print("Handling outliers...")
    
    # Apply outlier handling to key numerical columns
    for column in ['Price_USD', 'PricePerSqm_USD', 'Area_SqM', 'Bedrooms', 'TotalRooms', 'Floor', 'TotalFloors']:
        if column in df_clean.columns:
            df_clean = _handle_outliers(df_clean, column)
    
    # Drop irrelevant columns
    if 'ID' in df_clean.columns:
        df_clean = df_clean.drop(['ID'], axis=1)  # ID is just a record identifier
    
    print(f"Data shape after preprocessing: {df_clean.shape}")
    return df_clean

def _handle_outliers(df, column):
    """
    Cap outliers at 99th percentile to preserve data while reducing extreme values.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data containing the column.
    column : str
        Column name to handle outliers for.
        
    Returns:
    --------
    pandas.DataFrame
        Data with outliers handled.
    """
    upper_limit = df[column].quantile(OUTLIER_PERCENTILE)
    print(f"  Capping {column} at {upper_limit}")
    df[column] = np.where(df[column] > upper_limit, upper_limit, df[column])
    return df

def prepare_data_for_modeling(df, categorical_features=None):
    """
    Split the data and create preprocessing pipelines.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data to prepare for modeling.
    categorical_features : list, optional
        List of categorical feature names. If None, uses default from config.
        
    Returns:
    --------
    tuple
        Tuple containing (X_train, X_test, y_train, y_test, preprocessor, numerical_features, categorical_features).
    """
    print("\nPreparing data for modeling...")
    # Copy data to avoid modifying the original
    df_model = df.copy()
    
    # If categorical_features is None, use the config list (but check if columns exist)
    if categorical_features is None:
        from config import CATEGORICAL_FEATURES
        # Filter to only include columns that actually exist in the dataframe
        categorical_features = [col for col in CATEGORICAL_FEATURES if col in df_model.columns]
    
    # Ensure target column exists
    if LOG_TARGET_COLUMN not in df_model.columns:
        if 'Price_USD' in df_model.columns:
            print(f"Creating log-transformed target column: {LOG_TARGET_COLUMN}")
            df_model[LOG_TARGET_COLUMN] = np.log1p(df_model['Price_USD'])
        else:
            raise ValueError("Target column not found in data")
    
    # Remove any analysis-only columns if they exist
    if 'PriceBracket' in df_model.columns:
        df_model = df_model.drop(['PriceBracket'], axis=1)
    if 'AreaBracket' in df_model.columns:
        df_model = df_model.drop(['AreaBracket'], axis=1)
    
    # Drop columns not needed for modeling
    cols_to_drop = [col for col in COLS_TO_DROP_MODELING if col in df_model.columns]
    X = df_model.drop(cols_to_drop, axis=1)
    y = df_model[LOG_TARGET_COLUMN]  # Use log-transformed price as target
    
    # Identify numerical features by excluding categorical ones
    numerical_features = [col for col in X.columns if col not in categorical_features]
    
    print(f"Categorical features: {categorical_features}")
    print(f"Numerical features (first 5): {numerical_features[:5]}...")
    print(f"Total features: {len(categorical_features) + len(numerical_features)}")
    

    # Split data without stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
    )
    
    # Create preprocessing pipelines for numerical and categorical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessors
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, preprocessor, numerical_features, categorical_features