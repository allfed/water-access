import pandas as pd

# Assuming the data is in a CSV file named 'data.csv'
file_path = '/Users/kevin/Documents/ProgrammingIsFun/ALLFED/Water/water-access-gis/water-access/data/original_data/WHO Household Water Data - 2023 Data.csv'

# Read the data into a DataFrame
df = pd.read_csv(file_path)

# Replace '>99' with 100 and '<1' with 0
df.replace({'<1': 0, '>99': 100}, inplace=True)

# Convert relevant columns to numeric for proper calculations
columns_to_convert = ['TOTALPiped', 'RURALPiped', 'URBANPiped']
df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')

# Fill missing rural and urban piped data where total piped is 100
df.loc[(df['TOTALPiped'] == 100) & (df['RURALPiped'].isna()), 'RURALPiped'] = 100
df.loc[(df['TOTALPiped'] == 100) & (df['URBANPiped'].isna()), 'URBANPiped'] = 100

# Keep only the most recent year for each country
most_recent_df = df.sort_values('Year').drop_duplicates('Country', keep='last')

# Extract only the needed columns
final_df = most_recent_df[['Country', 'Year', 'TOTALPiped', 'RURALPiped', 'URBANPiped']]

# # Display the result
# print(final_df)
