# data preprocessing script for CitySim
# note: this is a SKELETON script, meaning i have used arbitrary names
# and basic ideas for this script. We will need to tailor it a bit more
# once I recieve access to the real dataset. 
import pandas as pd

df = pd.read_csv('citysim_dataset.csv')

print(df.head())

# Removing any duplicate entries
df.drop_duplicates(inplace=True)

# checking for any missing values in critical columns 
df.fillna(method='ffill', inplace=True)  

# Ensure all vehicle identifiers are consistent
df['carId'] = df['carId'].astype(str).str.pad(width=6, side='left', fillchar='0')

# Validating the ranges of coordinates (both pixel and geographic)
df = df[(df['carCenterX'] >= 0) & (df['carCenterX'] <= max_image_width)]
df = df[(df['carCenterY'] >= 0) & (df['carCenterY'] <= max_image_height)]

# standardizing
df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
df['heading'] = pd.to_numeric(df['heading'], errors='coerce')

# Saving
df.to_csv('citysim_cleaned.csv', index=False)

