import pandas as pd
import re

# Load the CSV file
df = pd.read_csv('property_prediction_data_final.csv', encoding='utf-8')

# Georgian to English translation dictionary
georgian_to_english = {
    # Districts (რაიონები)
    'ვაკე': 'Vake',
    'საბურთალო': 'Saburtalo',
    'დიდუბე': 'Didube',
    'გლდანი': 'Gldani',
    'ნაძალადევი': 'Nadzaladevi',
    'ისანი': 'Isani',
    'სამგორი': 'Samgori',
    'კრწანისი': 'Krtsanisi',
    'მთაწმინდა': 'Mtatsminda',
    'ჩუღურეთი': 'Chughureti',
    
    # Subdistricts/Neighborhoods (უბნები)
    'ნუცუბიძის პლატო': 'Nutsubidze Plateau',
    'დიღომი': 'Dighomi',
    'ვაჟა-ფშაველა': 'Vazha-Pshavela',
    'დოლიძე': 'Dolidze',
    'წერეთელი': 'Tsereteli',
    'მარჯანიშვილი': 'Marjanishvili',
    'რუსთაველი': 'Rustaveli',
    
    # Seasons (სეზონები)
    'ზაფხული': 'Summer',
    'შემოდგომა': 'Autumn',
    'ზამთარი': 'Winter',
    'გაზაფხული': 'Spring'
}

# Function to detect Georgian script
def contains_georgian(text):
    if not isinstance(text, str):
        return False
    return bool(re.search('[\u10A0-\u10FF]', text))

# Print original data sample to verify Georgian characters
print("Original data sample:")
sample_rows = df.head(5)
print(sample_rows)

# Print columns containing Georgian text
columns_with_georgian = []
for column in df.columns:
    if df[column].apply(contains_georgian).any():
        columns_with_georgian.append(column)
        
print(f"\nColumns containing Georgian text: {columns_with_georgian}")

# Print unique Georgian values for diagnosis
for column in columns_with_georgian:
    georgian_values = df[column][df[column].apply(contains_georgian)].unique()
    print(f"\nUnique Georgian values in {column} column:")
    for value in georgian_values[:10]:  # Print first 10 to avoid too much output
        print(f"  {value}")

# Function to translate Georgian text
def translate_georgian(text):
    if not isinstance(text, str):
        return text
    
    # Try direct translation from dictionary
    if text in georgian_to_english:
        return georgian_to_english[text]
    
    # If text contains Georgian characters but no direct match, print for diagnosis
    if contains_georgian(text):
        print(f"No translation found for: {text}")
    
    return text

# Columns to translate (excluding PropertyType)
columns_to_translate = ['District', 'Subdistrict', 'Street', 'ListingSeason']

# Apply translations only to columns that exist and contain Georgian
for column in columns_to_translate:
    if column in df.columns and column in columns_with_georgian:
        print(f"Translating {column} column...")
        df[column] = df[column].apply(translate_georgian)

# Save the translated data to a new CSV file
output_file = 'property_prediction_data_english.csv'
df.to_csv(output_file, index=False, encoding='utf-8')

print(f"\nTranslation completed and saved to '{output_file}'")

# Check if translation worked
print("\nSample of translated data:")
print(df.head(5))