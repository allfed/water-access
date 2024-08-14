import pandas as pd

def process_data(file_path):
    # Read the data into a DataFrame
    df = pd.read_csv(file_path)

    # Replace '>99' with 100 and '<1' with 0
    df.replace({'<1': 0, '>99': 100}, inplace=True)

    # Convert relevant columns to numeric for proper calculations
    columns_to_convert = ['TOTALPiped', 'RURALPiped', 'URBANPiped', 'Population \r\n(thousands)', '% urban']
    df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')

    # Fill missing rural and urban piped data where total piped is 100
    df.loc[(df['TOTALPiped'] == 100) & (df['RURALPiped'].isna()), 'RURALPiped'] = 100
    df.loc[(df['TOTALPiped'] == 100) & (df['URBANPiped'].isna()), 'URBANPiped'] = 100

    # Keep only the most recent year for each country
    most_recent_df = df.sort_values('Year').drop_duplicates('Country', keep='last')

    # Recalculate TOTALPiped
    most_recent_df['TOTALPiped_Recalculated'] = (
        (most_recent_df['% urban'] / 100) * most_recent_df['URBANPiped'] +
        ((100 - most_recent_df['% urban']) / 100) * most_recent_df['RURALPiped']
    )

    # Check for discrepancies
    most_recent_df['Discrepancy'] = most_recent_df['TOTALPiped'] - most_recent_df['TOTALPiped_Recalculated']

    # Extract the needed columns, including Population and % urban
    final_df = most_recent_df[['Country', 'Year', 'Population \r\n(thousands)', '% urban', 
                               'TOTALPiped', 'RURALPiped', 'URBANPiped', 'TOTALPiped_Recalculated', 'Discrepancy']]
    
    return final_df

# in