"""
Feature engineering functions for the real estate price prediction model.
"""

import pandas as pd
import numpy as np
import hashlib

def engineer_features(df):
    """
    Create new features to improve model performance.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data to engineer features for.
        
    Returns:
    --------
    pandas.DataFrame
        Data with engineered features.
    """
    print("\nEngineering features...")
    # Create a copy
    df_engineered = df.copy()
    
    # 1. Age of listing (days since listing)
    last_date = df_engineered['ListingDate'].max()
    df_engineered['ListingAge'] = (last_date - df_engineered['ListingDate']).dt.days
    
    # 2. Room density (rooms per area)
    df_engineered['RoomDensity'] = df_engineered['TotalRooms'] / df_engineered['Area_SqM']
    
    # 3. Bedroom ratio (bedrooms as a proportion of total rooms)
    df_engineered['BedroomRatio'] = df_engineered['Bedrooms'] / df_engineered['TotalRooms']
    df_engineered['BedroomRatio'] = df_engineered['BedroomRatio'].fillna(0)  # Handle division by zero
    
    # 4. Floor position indicator (relative floor position in the building)
    df_engineered['FloorRatio'] = df_engineered['Floor'] / df_engineered['TotalFloors']
    df_engineered['FloorRatio'] = df_engineered['FloorRatio'].fillna(0)  # Handle division by zero
    
    # 5. Is top floor? (binary feature)
    df_engineered['IsTopFloor'] = (df_engineered['Floor'] == df_engineered['TotalFloors']).astype(int)
    
    # 6. Is ground floor? (binary feature)
    df_engineered['IsGroundFloor'] = (df_engineered['Floor'] <= 1).astype(int)
    
    # 7. Premium area indicator (based on price per sqm)
    median_price_per_sqm = df_engineered.groupby('District')['PricePerSqm_USD'].transform('median')
    df_engineered['IsPremiumArea'] = (df_engineered['PricePerSqm_USD'] > median_price_per_sqm).astype(int)
    
    # 8. Handle Georgian text with hash function to create stable IDs
    df_engineered['DistrictID'] = df_engineered['District'].apply(text_to_hash_id)
    df_engineered['SubdistrictID'] = df_engineered['Subdistrict'].apply(text_to_hash_id)
    df_engineered['StreetID'] = df_engineered['Street'].apply(text_to_hash_id)
    
    # 9. Additional time-based features
    df_engineered['DayOfWeek'] = df_engineered['ListingDate'].dt.dayofweek
    df_engineered['DayOfMonth'] = df_engineered['ListingDate'].dt.day
    df_engineered['IsWeekend'] = ((df_engineered['DayOfWeek'] >= 5) & (df_engineered['DayOfWeek'] <= 6)).astype(int)
    
    # 10. Apply log transformation to numerical features with skewed distributions
    df_engineered['LogPrice'] = np.log1p(df_engineered['Price_USD'])  # Target transformation
    df_engineered['LogArea'] = np.log1p(df_engineered['Area_SqM'])
    df_engineered['LogPricePerSqm'] = np.log1p(df_engineered['PricePerSqm_USD'])
    
    # 11. Calculate district and subdistrict price statistics
    df_engineered = add_location_price_statistics(df_engineered)
    
    
    # In feature_engineer.py
    df_engineered['LowPriceProperty'] = (df_engineered['Price_USD'] < 50000).astype(int)
    
    
    print(f"Created {len(df_engineered.columns) - len(df.columns)} new features")
    print(f"Final data shape: {df_engineered.shape}")
    
    return df_engineered

def text_to_hash_id(text):
    """
    Convert text (potentially in Georgian) to a stable hash ID.
    
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

def add_location_price_statistics(df):
    """
    Add price statistics by district and subdistrict.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data to add location price statistics to.
        
    Returns:
    --------
    pandas.DataFrame
        Data with location price statistics added.
    """
    # Calculate district price statistics
    district_price_stats = df.groupby('DistrictID')['PricePerSqm_USD'].agg(['mean', 'median', 'std']).reset_index()
    district_price_stats.columns = ['DistrictID', 'DistrictAvgPrice', 'DistrictMedianPrice', 'DistrictPriceStd']
    
    # Calculate subdistrict price statistics
    subdistrict_price_stats = df.groupby('SubdistrictID')['PricePerSqm_USD'].agg(['mean', 'median', 'std']).reset_index()
    subdistrict_price_stats.columns = ['SubdistrictID', 'SubdistrictAvgPrice', 'SubdistrictMedianPrice', 'SubdistrictPriceStd']
    
    # Merge these statistics back to the main dataframe
    df = pd.merge(df, district_price_stats, on='DistrictID', how='left')
    df = pd.merge(df, subdistrict_price_stats, on='SubdistrictID', how='left')
    
    return df