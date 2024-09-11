import pandas as pd
""" 


"""

input_file = '../../data/GIS/updated_GIS_output.csv'


# read the data
df = pd.read_csv(input_file)

# drop the columns 'shapeType'
df = df.drop(['shapeType'], axis=1)

# drop all columns except id and left, top, right, bottom
df = df[['id', 'left', 'top', 'right', 'bottom']]

# save the data
df.to_csv('../../results/GIS_data_country_slice_for_results.csv', index=False)

print('Data saved!\n')
