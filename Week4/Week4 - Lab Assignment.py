'''
Lab 4
'''

######### Part 1 ###########


'''
    1) Load the digits dataset from sklearn. Split your data into test set(%20) and train set(%80) randomly (random_state = 123).
    
'''
from cProfile import label
from sklearn.datasets import load_digits
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

digits_df = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits_df.data, digits_df.target,test_size=0.2, random_state=123)
    
    
'''
    2) Plot the first sample of the train set and the first sample of the test set. 
    
'''
import matplotlib.pyplot as plt

plt.matshow(digits_df.images[0])
plt.show()
    
'''    
    3) Train a KNN classifier with your training data. You need to use CV techniques to tune the following hyper-params:
        a) metric = {chebyshev, euclidean, manhattan}
        b) k = {1, 3, 5, 7, 9, 11, 13, 15}
    
    3-1) Use hold-out validation method to tune the hyper-params. (use 30% of your training data as a test set).  
    3-2) Use 5Fold-CV validation method to tune the hyper-params. (there are multiple ways for implementing it with sklearn).
    3-2-1) For each metric (e.g. chebyshev) plot the results of classifiers (e.g. F1-score or accuracy) vs k. 
'''
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

X2_train, X2_test, y2_train, y2_test = train_test_split(X_train, y_train, test_size=0.3)

cheby_f1_scores = {}
euclid_f1_scores = {}
manhat_f1_scores = {}

# max_f1_from_all = np.zeros((16, 1, 3))

class MaxF1Score:
    def __init__(self, index, f1, metric):
        self.index = index
        self.f1 = f1
        self.metric = metric
        
max_f1_scores = []

for i in range(1,16,2):
    neighbor_cheby = KNeighborsClassifier(n_neighbors=(i), metric="chebyshev")
    neighbor_cheby.fit(X2_train, y2_train)
    pred_cheby = neighbor_cheby.predict(X2_test)
    cheby_f1 = f1_score(y2_test, pred_cheby, average='macro')
    cheby_f1_scores[i] = cheby_f1


    neighbor_euclid = KNeighborsClassifier(n_neighbors=(i), metric="euclidean")
    neighbor_euclid.fit(X2_train, y2_train)
    pred_euclid = neighbor_euclid.predict(X2_test)
    euclid_f1 = f1_score(y2_test, pred_euclid, average='macro')
    euclid_f1_scores[i] = euclid_f1

    neighbor_manhat = KNeighborsClassifier(n_neighbors=(i), metric="manhattan")
    neighbor_manhat.fit(X2_train, y2_train)
    pred_manhat = neighbor_manhat.predict(X2_test)
    manhat_f1 = f1_score(y2_test, pred_manhat, average='macro')
    manhat_f1_scores[i] = manhat_f1

    if (cheby_f1_scores.get(i) > euclid_f1_scores.get(i)) and (cheby_f1_scores.get(i) > manhat_f1_scores.get(i)):
        # max_f1_from_all[i][0][0] = cheby_f1_scores.get(i)
        max_f1_scores.append(MaxF1Score(i, cheby_f1_scores.get(i), "chebyshev"))
    elif (euclid_f1_scores.get(i) > cheby_f1_scores.get(i)) and (euclid_f1_scores.get(i) > manhat_f1_scores.get(i)):
        # max_f1_from_all[i][0][1] = euclid_f1_scores.get(i)
        max_f1_scores.append(MaxF1Score(i, euclid_f1_scores.get(i), "euclidean"))
    else:
        # max_f1_from_all[i][0][2] = manhat_f1_scores.get(i)
        max_f1_scores.append(MaxF1Score(i, manhat_f1_scores.get(i), "manhattan"))

print("\nchebyshev f1 scores:\n")
print(cheby_f1_scores)
print("\neuclidean f1 scores:\n")
print(euclid_f1_scores)
print("\nmanhattan f1 scores:\n")
print(manhat_f1_scores)

print("\nBEST F1 FOR EACH K\n")
for score in max_f1_scores:
    print(f"k = {score.index}, f1 = {score.f1}, metric = {score.metric}")

print()

cheby_ls = sorted(cheby_f1_scores.items())
x_cheby, y_cheby = zip(*cheby_ls)
plt.plot(x_cheby,y_cheby, label="CHEBY")

euclid_ls = sorted(euclid_f1_scores.items())
x_euclid, y_euclid = zip(*euclid_ls)
plt.plot(x_euclid,y_euclid, label="EUCLID")

manhat_ls = sorted(manhat_f1_scores.items())
x_manhat, y_manhat = zip(*manhat_ls)
plt.plot(x_manhat,y_manhat, label="MANHAT")

plt.legend()
plt.show()

from sklearn.model_selection import cross_validate
from sklearn import linear_model

lasso = linear_model.Lasso()
cv_score = cross_validate(lasso, X2_train, y2_train, scoring='neg_mean_absolute_error',cv=5, return_train_score=True)
print("\n", cv_score)

'''   
    4) Test your trained best classifiers in previous part.
'''
# YOUR CODE GOES HERE  


######### Part 2 ###########

'''   
    1) We want to see how normalization of the features affect the results in previous part. We will try two different normailizer fom sklearn:
    
    1-1) Use StandardScaler() to normalize your training data. Repeat Q3 and Q4 in part 1.
    1-2) Use  MinMaxScaler() to normalize your training data. Repeat Q3 and Q4 in part 1.
'''

# YOUR CODE GOES HERE  