"""
Data loading functions for the real estate price prediction model.
"""

import pandas as pd
import os

def load_data(file_path):
    """
    Load data from CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file.
        
    Returns:
    --------
    pandas.DataFrame
        Loaded data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    return pd.read_csv(file_path)

def explore_data(df):
    """
    Perform initial data exploration.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data to explore.
        
    Returns:
    --------
    dict
        Dictionary containing exploration results.
    """
    exploration_results = {
        'shape': df.shape,
        'dtypes': df.dtypes,
        'summary': df.describe(),
        'missing_values': df.isnull().sum(),
        'head': df.head()
    }
    
    # Print exploration results
    print(f"Dataset shape: {exploration_results['shape']}")
    print("\nFirst 5 rows:")
    print(exploration_results['head'])
    print("\nData types:")
    print(exploration_results['dtypes'])
    print("\nSummary statistics:")
    print(exploration_results['summary'])
    print("\nMissing values per column:")
    print(exploration_results['missing_values'])
    
    return exploration_results

def load_and_explore_data(file_path):
    """
    Load and perform initial data exploration.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file.
        
    Returns:
    --------
    pandas.DataFrame
        Loaded data.
    """
    print("Loading data...")
    df = load_data(file_path)
    explore_data(df)
    return df