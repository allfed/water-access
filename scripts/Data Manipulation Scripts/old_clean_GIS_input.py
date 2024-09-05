import pandas as pd
""" 
DEPRECATED: This script is no longer used as the GIS data is now cleaned in the data processing pipeline.
"""


columns_to_keep = [
    "fid",
    "URBAN_1",
    "dtw_1",
    "pop_count_15_1",
    "slope_1",
    "ISOCODE",
    "grip_1_1",
    "grip_2_1",
    "grip_3_1",
    "grip_4_1",
    "grip_5_1",
    "shapeID",
    "shapeName"
]

input_file = '../../data/GIS/gis_data_adm1.csv'
output_file = '../../data/GIS/GIS_stripped_back.csv'
chunk_size = 300000  # Adjust based on your system's memory

# Initialize the CSV writer
with pd.read_csv(input_file, usecols=columns_to_keep, chunksize=chunk_size) as reader:
    for i, chunk in enumerate(reader):
        if i == 0:
            chunk.to_csv(output_file, index=False, mode='w')
        else:
            chunk.to_csv(output_file, index=False, header=False, mode='a')

print(f"Filtered data saved to {output_file}")
