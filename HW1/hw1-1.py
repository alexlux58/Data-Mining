# HW #1
# Date: 2/16/2022
# Name: Alex Lux

'''
1. Write a python code to implement KNN.
Note: in this question, you are NOT allowed to use Scikit-Learn package!
'''

'''
(a) Step 1 : Write a function to calculate pair-wise distance between two data points. (args:
datapoint1, datapoint2, dist-fcn) “dist-fcn” should be determined by the user (can be
either “Manhattan” or “Euclidean”)
'''
import numpy as np
from collections import Counter

# K nearest neighbor classifier class
class KNN:
    def __init__(self, k):
        self.k = k
    
    # fits the training samples and labels
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    # function to predict new samples and its labels
    def predict(self, X, dist_fcn):
        predicted_labels = [self._predict(x, dist_fcn) for x in X]
        return np.array(predicted_labels)
        
    def _predict(self, x, dist_fcn):
        # compute distances 
        distances = [self.distance(x, x_train, dist_fcn) for x_train in self.X_train ]
        # get k nearest neighbors and their labels
        k_indeces = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indeces]
        # majority vote, get the most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
        
    # function calculates manhattan or euclidean distance
    def distance(self, datapoint1, datapoint2, dist_fcn):
        if dist_fcn.lower() == "manhattan":
            return np.sum(abs(datapoint1 - datapoint2))
        elif dist_fcn.lower() == "euclidean":
            return np.sqrt(np.sum((datapoint1 - datapoint2)**2))
        else:
            print("INCORRECT entry for distance function.")

'''
(b) Step 2 : Load iris dataset (iris-dataset-1) using the Pandas package. The test dataset
includes indices: 5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145 and the rest
of the data points will be your training dataset.
'''
import pandas as pd
iris_df = pd.read_csv("iris-data-1.csv")

test_X = []   
test_y = []
train_X = []
train_y = []

for i in range(len(iris_df)):
    if (i % 5 == 0) and (i != 150) and (i != 0):
        test_X.append(iris_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].iloc[i])
        test_y.append(iris_df['species'].iloc[i])
    else:
        train_X.append(iris_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].iloc[i])
        train_y.append(iris_df['species'].iloc[i])

X_train = np.array(train_X)
y_train = np.array(train_y)
X_test = np.array(test_X)
y_test = np.array(test_y)


print("X_train:\n")
print(X_train)
print("y_train\n")
print(y_train)
print("X_test\n")
print(X_test)
print("y_test\n")
print(y_test)


'''
(c) Step 3: Set K = 11 and use your function in Step 1 to find the K closest data points to
the test samples and predict the label for each.
'''

distance_function = input("Enter Your Choice for the Distance Function: <manhattan> <euclidean>:\n")

clf = KNN(k=11)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test, distance_function)

'''
(d) Step 4: Calculate the accuracy.
'''

accuracy = np.sum(predictions == y_test) / len(y_test)
percent_accuracy = accuracy * 100
print(f"Accuracy = {accuracy:.3f} (% {percent_accuracy:.1f})")
