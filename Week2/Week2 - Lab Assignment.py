'''
Lab 2

Alex Lux
Jeffery Foyil
'''

######### Part 1 ###########


'''
    1) Download the iris-data-1 from Canvas, use pandas.read_csv to load it.

'''
import pandas as pd

df = pd.read_csv("iris-data-1.csv")


'''
    2) after loading the data:
       print out the index, columns, and values information
       print out the length of the dataset 
       print out the last 50 data points
       print out the labels (species)
       print out the petal_length for the last sample
'''
print("--------------------------------------------------------")
print('#2')
print(df.index)
print(df.columns)
print(df.values)

print("--------------------------------------------------------")

print("Length of dataset: ", len(df))

print("--------------------------------------------------------")

print(df[:98:-1])

print("--------------------------------------------------------")

print(df['species'])

print("--------------------------------------------------------")

print(df['petal_length'].iloc[len(df)-1])


'''
    3) print out the mean and std of each feature (sepal_length, sepal_width, petal_length, petal_width)
    print out the mean of petal_length for first 100 samples
    print out the maximum and minimum values for each feature

'''

print("--------------------------------------------------------")
print('#3')
print(df.mean(axis=0, skipna=True, numeric_only=True))

print("--------------------------------------------------------")

print(df.std(axis=0, skipna=True, numeric_only=True))

print("--------------------------------------------------------")

print(df['petal_length'].head(100).mean())

print("--------------------------------------------------------")

print(df.max(axis=0, skipna=True, numeric_only=True))

print("--------------------------------------------------------")

print(df.min(axis=0, skipna=True, numeric_only=True))

'''
    4)  print out the frequency count of each class (setosa, versicolor, virginica)
    Hint: use pandas’ function value_counts 

'''
print("--------------------------------------------------------")
print('#4')
print(df['species'].value_counts())


'''
    5) Use pandas.DataFrame.drop_duplicates to drop duplications in "petal_length" feature (keep the last instance) and print out the resulted data
    print out the length of the resulted dataset
    print out the mean of each feature
    print out the frequency count of each label (=class)

'''
print("--------------------------------------------------------")
print('#5')
drop_duplicates_df = df.drop_duplicates(subset=["petal_length"], keep='last')
print(drop_duplicates_df)

print("--------------------------------------------------------")

print(len(drop_duplicates_df))

print("--------------------------------------------------------")

print(drop_duplicates_df.mean(axis=0, skipna=True, numeric_only=True))

print("--------------------------------------------------------")

print(drop_duplicates_df['species'].value_counts())

######### Part 2 ###########
'''
    1)  Use pandas.DataFrame.plot() to plot all of the columns in a single graph. What are the X axis and Y axis in the resulted graph?
    
    x-axis: integer number of samples 0 - 149 (150 total - sample size)
    y-axis: integer number representing the length and width values for the sepal and petal
'''
import matplotlib.pyplot as plt
print("--------------------------------------------------------")
print("part 2 #1")
drop_duplicates_df.plot()
plt.tight_layout()
plt.show()

'''
    2)  plot the bar graph of your data
    Hint: pandas.DataFrame.plot.bar()
    
'''
print("--------------------------------------------------------")
print("part 2 #2")
drop_duplicates_df.plot(kind='bar')
plt.show()


'''
    4)  plot the histogram graph for "petal_length" feature
    Hint: pandas.DataFrame.plot.histograms()
    
'''
print("--------------------------------------------------------")

print("part 2 #4")
drop_duplicates_df['petal_length'].plot.hist()
plt.show()


'''
    5)  Use the bar graph to show the frequency of each label (class)
'''
print("--------------------------------------------------------")
print("part 2 #5")
drop_duplicates_df['species'].value_counts().plot.bar()
plt.show()


######### Part 3 ###########

'''
    1) Download the iris-data-2 from Canvas, use pandas.read_csv to load it.

'''
print("--------------------------------------------------------")
print("part 3 #1")
df2 = pd.read_csv("iris-data-2.csv")
print(df2)

'''
    2) Use pandas.DataFrame.drop_duplicates to drop duplications in "ID" (keep the first instance) and save the resulted data frame in a new datafram (df). Print out the shape of df.

'''
print("--------------------------------------------------------")
print("part 3 #2")
drop_duplicates_df2 = df2.drop_duplicates(subset=["ID"], keep='first')
print(drop_duplicates_df2.shape)


'''
    3)  plot the bar graph for df['color']
    Hint: pandas.DataFrame.plot.bar()
    
'''
print("--------------------------------------------------------")
print("part 3 #3")
drop_duplicates_df2['color'].value_counts().plot.bar()
plt.show()


'''
    4)  How many missing values we have in 'color' column?
    Hint: pandas.DataFrame.dropna()
    
'''
print("--------------------------------------------------------")
print("part 3 #4")
df2_remove_na = drop_duplicates_df2.dropna()
print('Missing values: ', drop_duplicates_df2['color'].size - df2_remove_na['color'].size)

'''
    5)  Make the values in 'color' column to be consistant and remove the unkown values
    Hint: pandas.DataFrame.replace()
       
'''
print("--------------------------------------------------------")
print("part 3 #5")
df2_consistant = df2_remove_na.replace(to_replace=['Bluee', 'Blue', 'U', 'None', 'Nan', 'none', 'Red'], value=['blue', 'blue', None, None, None, None, 'red'])
df2_consistant = df2_consistant.dropna()
df2_consistant['color'].value_counts().plot.bar()
plt.show()



