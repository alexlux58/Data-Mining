'''
Alex Lux
'''

##########Part 0 ###########
'''
    1)  from sklearn.datasets import load_digits  (Each datapoint is a 8x8 image of
a digit)
    breifly explain: what are the features in this classification problem and how 
many features do we have?
    Find the distribution of the lables. 
    Use plot command to visualize the first five samples in the dataset. What are 
their lables?
    Split your data into train(80% of data) and test(20% of data) via random 
selection 
'''
from sklearn.datasets import load_digits
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

digits = load_digits()

print("FEAUTRE_NAMES(64 features):\n", digits.feature_names) # 64 features

X = pd.DataFrame(digits.data)
y = pd.DataFrame(digits.target)
# print(X.shape)
# print(X.columns)

'''
There are 64 total features, each with integer values from [0,16].
Their names are integers from [0,63).
'''
print("\nDistribution of the Labels:\n", y.value_counts())

fig = plt.figure(figsize=(6, 6))  # figure size in inches
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(5):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
    # label the image with the target value
    ax.text(0, 7, str(digits.target[i]))
    
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report as creport
from sklearn.metrics import confusion_matrix as cmatrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")

from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report as creport
from sklearn.metrics import confusion_matrix as cmatrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")

##########Part 1 ###########
'''
    1)  Try LogisticRegression from sklearn.linear_model
        Try to tune the hyperparameters (only change these params: penalty, C, 
Solver) via hold-out CV (30% for validation).
        candidate values: 
        penalties = [ 'l1', 'l2', 'elasticnet', 'none' ]
        solvers = [ 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga' ]
        C = [1 , 10 , 0.1]
        
        What is the class_weight param? Do you need to modify that? Why?
    
'''
print("======================================= 1 ====================================================")
#hold-out validation. Do train_test_split on your training data:
X_train_new, X_valid, y_train_new, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=123)
P = [ 'l1', 'l2', 'elasticnet', 'none' ]
S = [ 'newton-cg', 'lbfgs', 'sag', 'liblinear', 'saga' ]
C = [1 , 10 , 0.1]

results = {}
for p in P:
  for c in C:
    for s in S:
      try:
        lr = LogisticRegression(penalty=p, C=c, solver=s)
        lr.fit(X_train_new, y_train_new)
        pred = lr.predict(X_valid)
        results[(p, c, s)] = f1_score(y_valid, pred, average='weighted')
      except:
        continue

print("Best method to use determined by hold-out validation:", max(results, key=results.get))

#You can use the hold-out dataset (the smaller subset) for the training, OR the bigger subset for training. Up to you!
lr_best = LogisticRegression(penalty='none', C=0.1, solver='sag')
lr_best.fit(X_train_new, y_train_new)
pred_best = lr_best.predict(X_valid)
print(classification_report(y_valid, pred_best, digits=4))
print("class_weight is the parameter that associates weights with classes.")
print("The default is that each class has weight 1.")
print("I don't need to modify that since my classes are balanced.")

print("======================================= 2 ====================================================")

'''
    2)  Try LinearSVC from sklearn.svm
    Try to tune the hyperparameters (only change these params: penalty, C, loss) 
via hold-out CV (30% for validation).
    penalties = [ 'l1', 'l2', 'elasticnet', 'none' ]
    solvers = [ 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga' ]
    C = [1 , 10 , 0.1]
'''
X_train_svc, X_test_svc, y_train_svc, y_test_svc = train_test_split(X_train, y_train, test_size=0.3, random_state=123)

parameters = {'penalty': ['l1', 'l2', 'elasticnet', 'none']
              , 'loss': ['hinge', 'squared_hinge']
              , 'C': [1 , 10 , 0.1]
              , 'random_state': [123]}
svc = LinearSVC()
clf = GridSearchCV(svc, parameters, cv=5)
clf.fit(X_train_svc, y_train_svc)

print("\nBest params:", clf.best_params_)
print("\nBest score:", clf.best_score_)
best_estimator = clf.best_estimator_

y_pred_svc = clf.predict(X_test_svc)
print("SVC classification report:\n", creport(y_true=y_test_svc, y_pred=y_pred_svc, digits=4))

'''
    3)  Try SVC from sklearn.svm (this classifier can also be used with linear 
kernel == LinearSVC)
    Try to tune the hyperparameters (only change these params: 
decision_function_shape, C, kernel, degree) via hold-out CV (30% for validation).
    penalties = [ 'l1', 'l2', 'elasticnet', 'none' ]
    solvers = [ 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga' ]
    C = [1 , 10 , 0.1]
'''

print("======================================= 3 ====================================================")

from sklearn.svm import SVC
from sklearn.metrics import f1_score
import numpy as np
from sklearn.metrics import classification_report

X_train_vanilla_svc, X_test_vanilla_svc, y_train_vanilla_svc, y_test_vanilla_svc = train_test_split(X_train, y_train, test_size=0.3, random_state=123)

decision_function_shape = ['ovo', 'ovr']
kernel = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
degree = [3, 4, 5, 6, 7]
C = [1 , 10 , 0.1]

results_f1_vanilla_svc = {}
max_keys = {}

for d in decision_function_shape:
    for k in kernel:
        for dgre in degree:
            for c in C:
                vanilla_svc_clf = SVC(C=c, kernel=k, degree=dgre, decision_function_shape=d)
                if k == "precomputed":
                    kernel_train = np.dot(X_train_vanilla_svc, X_train_vanilla_svc.T) # linear kernel
                    vanilla_svc_clf.fit(kernel_train, y_train_vanilla_svc.values.ravel())
                    kernel_test = np.dot(X_test_vanilla_svc, X_train_vanilla_svc.T)
                    pred = vanilla_svc_clf.predict(kernel_test)
                else:
                    vanilla_svc_clf.fit(X_train_vanilla_svc, y_train_vanilla_svc.values.ravel())
                    pred = vanilla_svc_clf.predict(X_test_vanilla_svc)
                results_f1_vanilla_svc[(d,k, dgre,c)] = f1_score(y_test_vanilla_svc, pred, average='weighted')
                max_keys = {'c': c, 'd': d, 'dgre': dgre, 'k': k}
                        
max_key_f1_vanilla_svc = max(results_f1_vanilla_svc, key=results_f1_vanilla_svc.get)
print(f"Best Hyperparameters = (kernel, decision_function_shape, degree, C): {max_key_f1_vanilla_svc}")
print(f"F1-SCORE: {results_f1_vanilla_svc[max_key_f1_vanilla_svc]}")

best_vanilla_svc_clf = SVC(C=max_keys['c'], kernel=max_keys['k'], degree=max_keys['dgre'], decision_function_shape=max_keys['d'])
best_pred = np.ndarray(shape=[6,6])
if k == "precomputed":
    kernel_train = np.dot(X_train_vanilla_svc, X_train_vanilla_svc.T) # linear kernel
    best_vanilla_svc_clf.fit(kernel_train, y_train_vanilla_svc.values.ravel())
    kernel_test = np.dot(X_test_vanilla_svc, X_train_vanilla_svc.T)
    best_pred = best_vanilla_svc_clf.predict(kernel_test)
else:
    best_vanilla_svc_clf.fit(X_train_vanilla_svc, y_train_vanilla_svc.values.ravel())
    best_pred = best_vanilla_svc_clf.predict(X_test_vanilla_svc)
print("\nCLASSIFICATION REPORT [VANILLA SVC]")
print(classification_report(y_test_vanilla_svc, best_pred, digits=4))
print("===========================================================================================")

##########Part 2 ###########
'''
    1)  Test your trained models in part1: Q1, Q2, and Q3 with the test set and 
pick the best model. Try to analyze the confusion matrix and explain which classes 
are mostly confused with each other.
'''

##########Part 3 ###########
'''
    1)  Repeat part 1 and 2 with Normalized data
'''
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)

#hold-out validation. Do train_test_split on your training data:
X_train_new, X_valid, y_train_new, y_valid = train_test_split(X_train_norm, y_train, test_size=0.3, random_state=123)

P = [ 'l1', 'l2', 'elasticnet', 'none' ]
S = [ 'newton-cg', 'lbfgs', 'sag', 'liblinear', 'saga' ]
C = [1 , 10 , 0.1]

results = {}
for p in P:
  for c in C:
    for s in S:
      try:
        lr = LogisticRegression(penalty=p, C=c, solver=s)
        lr.fit(X_train_new, y_train_new)
        pred = lr.predict(X_valid)
        results[(p, c, s)] = f1_score(y_valid, pred, average='weighted')
      except:
        continue

print("Best method to use determined by hold-out validation:", max(results, key=results.get))

#You can use the hold-out dataset (the smaller subset) for the training, OR the bigger subset for training. Up to you!
lr_best = LogisticRegression(penalty='l2', C=10, solver='sag')
lr_best.fit(X_train_new, y_train_new)
pred_best = lr_best.predict(X_valid)
print(classification_report(y_valid, pred_best, digits=4))

print("class_weight is the parameter that associates weights with classes.")
print("The default is that each class has weight 1.")
print("I don't need to modify that since my classes are balanced.")


# LinearSVC
ss = StandardScaler()
ntrain = ss.fit_transform(X_train)
ntest = ss.transform(X_test)

X_train_svc, X_test_svc, y_train_svc, y_test_svc = train_test_split(ntrain, y_train, test_size=0.3, random_state=123)

parameters = {'penalty': ['l1', 'l2', 'elasticnet', 'none']
              , 'loss': ['hinge', 'squared_hinge']
              , 'C': [1 , 10 , 0.1]
              , 'random_state': [123]}
svc = LinearSVC()
clf = GridSearchCV(svc, parameters, cv=5)
clf.fit(X_train_svc, y_train_svc)

print("\nBest params:", clf.best_params_)
print("\nBest score:", clf.best_score_)
best_estimator = clf.best_estimator_

y_pred_svc = clf.predict(X_test_svc)
print("DTC classification report:\n", creport(y_true=y_test_svc, y_pred=y_pred_svc, digits=4))

##########Part 4 ###########
'''     
    1)  Try MLPClassifier from sklearn.neural_network
        (a NN with two hidden layers, each with 100 nodes)
        Use 20% of your training data as the validation set to tune other hyper-
parameters (e.g. activation, solver). Try different values and pick the best one.
        
    3)  Print classification report for the test set.
'''
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification

X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(X_train, y_train, test_size=0.2, random_state=123)
parameters = {'activation': ['identity', 'logistic', 'tanh', 'relu'], 
              'solver': ['lbfgs', 'sgd', 'adam'], 
              'learning_rate': ['constant', 'invscaling', 'adaptive'], 
              'random_state': [123]}
nn = MLPClassifier()
nn = GridSearchCV(nn, parameters, cv=5)
nn.fit(X_train_nn, y_train_nn)
print("\nBest params:", nn.best_params_)
print("\nBest score\n:", nn.best_score_)
best_estimator = nn.best_estimator_
nn_pred = nn.predict(X_test)
print("Classification report:\n", creport(y_test, nn_pred, digits=4))