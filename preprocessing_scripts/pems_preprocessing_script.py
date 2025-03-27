# this is the preprocessing script for the pems data
# what this script is doing is makingthe data more standardized and ready to be trained on
# this handles null values (the method is DEBATABLE AND SUBJECT TO CHANGE)
# and converts data types to be appropriate
# and normalizing average speeds
import pandas as pd
import sys

def preprocess_pems_data(input_path, output_path):
    df = pd.read_csv(input_path)

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%m/%d/%Y %H:%M:%S')

    #  categorical data is treated right :)
    df['District'] = df['District'].astype('category')
    df['Freeway #'] = df['Freeway #'].astype('category')
    df['Direction of Travel'] = df['Direction of Travel'].astype('category')
    df['Lane Type'] = df['Lane Type'].astype('category')

    # conversion of numerical data
    numeric_fields = ['Station Length', 'Samples', '% Observed', 'Total Flow', 'Avg Occupancy', 'Avg Speed']
    for field in numeric_fields:
        df[field] = pd.to_numeric(df[field], errors='coerce')

    # handling missing values by MEAN (common method, SUBJECT TO CHANGE, DW)
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    categorical_cols = df.select_dtypes(include=['category']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Normalize data (can totally skip this too)
    df['Avg Speed'] = (df['Avg Speed'] - df['Avg Speed'].min()) / (df['Avg Speed'].max() - df['Avg Speed'].min())

    # saivng
    df.to_csv(output_path, index=False)

    print("Data preprocessing complete here yay!")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <input_path> <output_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    preprocess_pems_data(input_path, output_path)
