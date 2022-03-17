import pandas as pd

# can read csv, xlsx, txt

# data_frame = pd.read_csv('pokemon_data.csv')
data_frame = pd.read_excel('pokemon_data.xlsx')
# data_frame = pd.read_csv('pokemon_data.txt', delimiter='\t')
# print(data_frame)

# read headers
print(data_frame.columns)

# read specified column
# print(data_frame['Name'][0:5])

# read row
# print(data_frame.head(4))

# print(data_frame.loc[data_frame['Type 1'] == "Grass"])
# print(data_frame.iloc[1:4])

# describe statistical data
print(data_frame.describe())

# sort ascending=True when 1 and =False when 0
print(data_frame.sort_values(['Type 1', 'HP'], ascending=[1,0]))

# add a column
# data_frame['Total'] = data_frame['HP'] + data_frame['Attack'] + data_frame['Defense']
# or
# adds all the rows up from column 4 through 9 and adds horizontally when axis=1 and vertically when =0
data_frame['Total'] = data_frame.iloc[:, 4:10].sum(axis=1)

# remove a column
data_frame = data_frame.drop(columns=['Total'])

# send data to excel sheet
# data_frame.to_excel('modified.xlsx', index=False)

# filtering data
new_df = data_frame.loc[(data_frame['Type 1'] == 'Grass') & (data_frame['Type 2'] == 'Poison') & (data_frame['HP'] > 70)]
# new_df = new_df.reset_index(drop=True)
# or
new_df.reset_index(drop=True, inplace=True)
new_df.to_csv('newDF.csv')

data_frame.loc[~data_frame['Name'].str.contains('Mega')]

# conditional changes 
data_frame.loc[data_frame['Type 1'] == 'Fire', 'Type 1'] = 'Flamer'

# aggregate stats
data_frame.groupby(['Type 1']).mean().sort_values('Attack', ascending=False)