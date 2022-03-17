import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("iris-data-1.csv")

print(df)

print("--------------------------------------------------------")

print(df.index)
print(df.columns)
print(df.values)

print("--------------------------------------------------------")

print("Length of dataset: ", len(df))

print("--------------------------------------------------------")

print(df.tail(51))

print("--------------------------------------------------------")

print(df['species'])

print("--------------------------------------------------------")

print(df['petal_length'].iloc[len(df)-1])

print("--------------------------------------------------------")

print(df.mean(axis=0, skipna=True, numeric_only=True))

print("--------------------------------------------------------")

print(df.std(axis=0, skipna=True, numeric_only=True))

print("--------------------------------------------------------")

print(df['petal_length'].head(100).mean())

print("--------------------------------------------------------")

print(df.max(axis=0, skipna=True, numeric_only=True))

print("--------------------------------------------------------")

print(df.min(axis=0, skipna=True, numeric_only=True))

print("--------------------------------------------------------")

print(df['species'].value_counts())

print("--------------------------------------------------------")

drop_duplicates_df = df.drop_duplicates(subset=["petal_length"], keep='last')
print(drop_duplicates_df)

print("--------------------------------------------------------")

print(len(drop_duplicates_df))

print("--------------------------------------------------------")

print(drop_duplicates_df.mean(axis=0, skipna=True, numeric_only=True))

print("--------------------------------------------------------")

print(drop_duplicates_df['species'].value_counts())

print("--------------------------------------------------------")
print("part 2 #1")
# my_plot = drop_duplicates_df.plot()
# plt.tight_layout()
# plt.show()

print("--------------------------------------------------------")
print("part 2 #2")
# my_plot_graph = drop_duplicates_df.plot.bar()
# plt.show()

print("--------------------------------------------------------")

print("part 2 #4")
# drop_duplicates_df['petal_length'].plot.hist()
# plt.show()

print("--------------------------------------------------------")
print("part 2 #5")
# drop_duplicates_df.value_counts().plot.bar()
# plt.show()

print("--------------------------------------------------------")
print("part 3 #1")
df2 = pd.read_csv("iris-data-2.csv")
print(df2)

print("--------------------------------------------------------")
print("part 3 #2")
drop_duplicates_df2 = df2.drop_duplicates(subset=["ID"], keep='first')
print(drop_duplicates_df2.shape)

print("--------------------------------------------------------")
print("part 3 #3")
drop_duplicates_df2['color'].value_counts().plot.bar()
plt.show()

print("--------------------------------------------------------")
print("part 3 #4")
df2_remove_na = drop_duplicates_df2.dropna()
print('Missing values: ', drop_duplicates_df2['color'].size - df2_remove_na['color'].size)

print("--------------------------------------------------------")
print("part 3 #5")
df2_consistant = df2_remove_na.replace(to_replace=['Bluee', 'Blue', 'U', 'None', 'Nan', 'none'], value=['blue', 'blue', None, None, None, None])
df2_consistant = df2_consistant.dropna()
df2_consistant['color'].value_counts().plot.bar()
plt.show()