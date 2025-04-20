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


# Example usage:
# example_property = {'Area_SqM': 75, 'Bedrooms': 2, 'TotalRooms': 3, 'Floor': 5, 'TotalFloors': 9, 'PropertyType': 'Apartment', 'ListingYear': 2023, 'ListingMonth': 4, 'ListingSeason': 'Spring', 'District': 'Vake-Saburtalo', 'Subdistrict': 'Saburtalo', 'Street': 'Nutsubidze St.'}
# predicted_price = predict_property_price(example_property)
# print(f'Predicted price: ${predicted_price:.2f}')
