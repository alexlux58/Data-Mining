import numpy as np
import pandas as pd
import heapq
from sklearn.metrics import accuracy_score


def distance_fcn(datapoint1, datapoint2, distfcn='Euclidean'):
    
    """
     datapoint1: np.array
     datapoint2: np.array
     distfcn: str
    """
 
    if distfcn == 'Manhattan':
        distance = sum(np.absolute(datapoint1 - datapoint2))
    else:
        distance = sum(pow((datapoint1-datapoint2),2))
        distance  = np.sqrt(distance)
    return distance



    
data = pd.read_csv('iris-data-1.csv', delimiter = ',')
Y_test = data.loc[[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140], ['species']]
X_test = data.loc[[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140], ['sepal_length','sepal_width','petal_length','petal_width']]
Y_test = Y_test.reset_index(drop = True)
X_test = X_test.reset_index(drop = True)

Y_train = data[~data.index.isin(test_data.index)]['species']
X_train = data[~data.index.isin(test_data.index)][['sepal_length','sepal_width','petal_length','petal_width']]
Y_train = Y_train.reset_index(drop = True)
X_train = X_train.reset_index(drop = True)

distfcn = 'Euclidean'
K = 7
prediction= []

for i in range(len(X_test)):
    distance_list = []
    for j in range(len(X_train)):
        pairwise_dis = distance_fcn(X_test.values[i], X_train.values[j], distfcn)
        distance_list.append(pairwise_dis)
  
    neighbors_index = heapq.nsmallest(K, range(len(distance_list)), distance_list.__getitem__)
    pred = Y_train.loc[neighbors_index]
    top_pred = pd.value_counts(pred).head(1)
    value, count = top_pred.index[0], top_pred.iat[0]
    prediction.append(value)
        
    

print (accuracy_score(Y_test, prediction))
  