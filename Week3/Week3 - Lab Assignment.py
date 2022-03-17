'''
Lab 3
Alex Lux
Rene Escamilla
'''

######### Part 1 ###########
import pandas as pd

'''
    1) Download the iris-data-1 from Canvas, use pandas.read_csv to load it.

'''
print("-------------------------------------------------------------")
df = pd.read_csv("iris-data-1.csv")
print("Part 1 #1")
print("IRIS DATA FRAME:\n")
print(df)


'''
    2) Split your data into test set(%30) and train set(%70) randomly. (Hint: you can use scikit-learn package tools for doing this)
    
'''
from sklearn.model_selection import train_test_split
import numpy as np

X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]] 
y = df["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


'''    
    3) Use KNeighborsClassifier from scikit-learn package. Train a KKN classifier using your training dataset  (K = 3, Euclidean distance).   
    
'''
from sklearn.neighbors import KNeighborsClassifier

kNeighbor = KNeighborsClassifier(n_neighbors=3)
kNeighbor.fit(X_train, y_train)

'''   
    4) Test your classifier (Hint: use predict method) and report the performance (report accuracy, recall, precision, and F1score). (Hint: use classification_report from scikit learn)
'''
from sklearn.metrics import classification_report, mean_absolute_error
print("-------------------------------------------------------------")
print("Part 1 #4")
print("CLASSIFICATION REPORT (K=3):\n")
pred = kNeighbor.predict(X_test)
print(classification_report(y_test, pred))

'''   
    5) report micro-F1score, macro-F1score, and weighted F1-score.
'''

from sklearn.metrics import f1_score
print("\nPart 1 #5")
print("#5 F1-score")
micro_f1_kNeighbor = f1_score(y_test, pred, average='micro')
macro_f1_kNeighbor = f1_score(y_test, pred, average='macro')
weighted_f1_kNeighbor = f1_score(y_test, pred, average='weighted')
print(f"MICRO F1: {micro_f1_kNeighbor:.3f}")
print(f"MACRO F1: {macro_f1_kNeighbor:.3f}")
print(f"WEIGHTED F1: {weighted_f1_kNeighbor:.3f}")

'''    
    6) Repeat Q3, Q4, and Q5 for "manhattan" distance function

'''
print("-------------------------------------------------------------")
print("Part 1 #6")
kNeighbor_manhat = KNeighborsClassifier(n_neighbors=3, metric="manhattan")
kNeighbor_manhat.fit(X_train, y_train)
pred_manhat = kNeighbor_manhat.predict(X_test)
print("CLASSIFICATION REPORT (K=3, DISTANCE FORMULA=MANHATTAN):\n")
print(classification_report(y_test, pred_manhat))

print("\n#6 F1-score")
micro_f1_manhattan = f1_score(y_test, pred_manhat, average='micro')
macro_f1_manhattan = f1_score(y_test, pred_manhat, average='macro')
weighted_f1_manhattan = f1_score(y_test, pred_manhat, average='weighted')
print(f"MICRO F1: {micro_f1_manhattan:.3f}")
print(f"MACRO F1: {macro_f1_manhattan:.3f}")
print(f"WEIGHTED F1: {weighted_f1_manhattan:.3f}")

'''   
    7) Compare your results in Q5 and Q6.

'''
print("-------------------------------------------------------------")
print("Part 1 #7 F1-SCORE COMPARISON (#5 vs #6):")
print(f"MICRO:\t   {micro_f1_kNeighbor:.3f}\t{micro_f1_manhattan:.3f}")
print(f"MACRO:\t   {macro_f1_kNeighbor:.3f}\t{macro_f1_manhattan:.3f}")
print(f"WEIGHTED:  {weighted_f1_kNeighbor:.3f}\t{weighted_f1_manhattan:.3f}")

'''
    8) Repeat Q3, Q4, Q5, Q6, and Q7 for K = 11.
'''
print("-------------------------------------------------------------")
print("Part 1 #8")
kNeighbor_eleven = KNeighborsClassifier(n_neighbors=11)
kNeighbor_eleven.fit(X_train, y_train)
pred_eleven = kNeighbor_eleven.predict(X_test)
print("CLASSIFICATION REPORT (K=11):\n")
print(classification_report(y_test, pred_eleven))

print("\nF1-scores")
micro_f1_k11 = f1_score(y_test, pred_eleven, average='micro')
macro_f1_k11 = f1_score(y_test, pred_eleven, average='macro')
weighted_f1_k11 = f1_score(y_test, pred_eleven, average='weighted')
print(f"MICRO F1: {micro_f1_k11:.3f}")
print(f"MACRO F1: {macro_f1_k11:.3f}")
print(f"WEIGHTED F1: {weighted_f1_k11:.3f}")

kNeighbor_eleven_manhat = KNeighborsClassifier(n_neighbors=11, metric="manhattan")
kNeighbor_eleven_manhat.fit(X_train, y_train)
pred_eleven_manhat = kNeighbor_eleven_manhat.predict(X_test)
print("\nCLASSIFICATION REPORT (K=11, DISTANCE FORMULA=MANHATTAN):\n")
print(classification_report(y_test, pred_eleven_manhat))

print("\nF1-Scores")
micro_f1_k11_manhattan = f1_score(y_test, pred_eleven_manhat, average='micro')
macro_f1_k11_manhattan = f1_score(y_test, pred_eleven_manhat, average='macro')
weighted_f1_k11_manhattan = f1_score(y_test, pred_eleven_manhat, average='weighted')
print(f"MICRO F1: {micro_f1_k11_manhattan:.3f}")
print(f"MACRO F1: {macro_f1_k11_manhattan:.3f}")
print(f"WEIGHTED F1: {weighted_f1_k11_manhattan:.3f}")

print("\nF1-SCORE COMPARISON (k11 vs k11 w/ manhattan):")
print(f"MICRO:\t   {micro_f1_k11:.3f}\t{micro_f1_k11_manhattan:.3f}")
print(f"MACRO:\t   {macro_f1_k11:.3f}\t{macro_f1_k11_manhattan:.3f}")
print(f"WEIGHTED:  {weighted_f1_k11:.3f}\t{weighted_f1_k11_manhattan:.3f}")

######### Part 2 ###########
'''
    0)  Repeat Q1 and Q2 in part 1.

'''
df = pd.read_csv("iris-data-1.csv")
X2 = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]] 
y2 = df["species"]
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.30)


'''
    1) Train a KKN classifier using your training dataset  (K = 7, Euclidean distance). 
    
    1-1) Test your classifier using predict_proba method. What is the difference between predict_proba and predict method?
    
    **predict(X): Predict the class labels for the provided data.

    **predict_proba(X): Return probability estimates for the test data X.

    1-2) report the performance based on your results in 1-1.
    
'''
print("-------------------------------------------------------------")
print("Part 2 #1-1, 1-2")
kNeighbor2 = KNeighborsClassifier(n_neighbors=7, metric="euclidean")
kNeighbor2.fit(X_train2, y_train2)
pred2 = kNeighbor2.predict_proba(X_test2)
maxes_pred = []
for i in range(len(pred2)):
    max = np.amax(pred2[i])
    for j in range(3):
        if max == pred2[i][j]:
            if j == 0:
                max_label = "setosa"
            elif j == 1:
                max_label = "versicolor"
            else:
                max_label = "virginica"
            maxes_pred.append(max_label)
print("\nCLASSIFICATION REPORT (K=7, DISTANCE FORMULA=EUCLIDEAN):\n")
print(classification_report(y_test2, maxes_pred))

######### Part 3 ###########

'''
    0) Repeat Q1 and Q2 in part 1.

'''
df = pd.read_csv("iris-data-1.csv")
X3 = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]] 
y3 = df["species"]
X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, test_size=0.30)

'''
    1) Use DecisionTreeClassifier from scikit-learn package. Train a DT classifier using your training dataset  (criterion='entropy', splitter= 'best'). 

'''
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(criterion='entropy', splitter='best')
clf.fit(X_train3, y_train3)

'''   
    2) Test your classifier (Hint: use predict method) and report the performance (report accuracy, recall, precision, and F1score). (Hint: use classification_report from scikit learn)
'''
pred_decision_tree = clf.predict(X_test3)
print("-------------------------------------------------------------")
print("\nCLASSIFICATION REPORT DECISION TREE(criterion='entropy', splitter='best'):\n")
print(classification_report(y_test3, pred_decision_tree))

'''   
    3) report micro-F1score, macro-F1score, and weighted F1-score
'''
print("\nF1-score Decision Tree\n")
micro_f1_DT = f1_score(y_test3, pred_decision_tree, average='micro')
macro_f1_DT = f1_score(y_test3, pred_decision_tree, average='macro')
weighted_f1_DT = f1_score(y_test3, pred_decision_tree, average='weighted')
print(f"MICRO F1: {micro_f1_DT:.3f}")
print(f"MACRO F1: {macro_f1_DT:.3f}")
print(f"WEIGHTED F1: {weighted_f1_DT:.3f}")

'''    
    4) Repeat Q1, Q2, and Q3 for "random" splitter.
'''
print("-------------------------------------------------------------")
print("Part 3 #4 Decision Tree\n")
clf2 = DecisionTreeClassifier(criterion='entropy', splitter='random')
clf2.fit(X_train3, y_train3)
pred_decision_tree2 = clf2.predict(X_test3)
print("\nCLASSIFICATION REPORT DECISION TREE(criterion='entropy', splitter='random'):\n")
print(classification_report(y_test3, pred_decision_tree2))

print("\nF1-score Decision Tree\n")
micro_f1_DT2 = f1_score(y_test3, pred_decision_tree2, average='micro')
macro_f1_DT2 = f1_score(y_test3, pred_decision_tree2, average='macro')
weighted_f1_DT2 = f1_score(y_test3, pred_decision_tree2, average='weighted')
print(f"MICRO F1: {micro_f1_DT2:.3f}")
print(f"MACRO F1: {macro_f1_DT2:.3f}")
print(f"WEIGHTED F1: {weighted_f1_DT2:.3f}")

'''   
    5) Compare your results in Q4 and Q3.

'''
print("\nF1-SCORE COMPARISON DECISION TREE(best vs. random):")
print(f"MICRO:\t   {micro_f1_DT:.3f}\t{micro_f1_DT2:.3f}")
print(f"MACRO:\t   {macro_f1_DT:.3f}\t{macro_f1_DT2:.3f}")
print(f"WEIGHTED:  {weighted_f1_DT:.3f}\t{weighted_f1_DT2:.3f}")

'''   
    6) Repeat Q2, Q3, Q4, and Q5 for criterion = "gini".

'''
print("-------------------------------------------------------------")
print("Part 3 #6")
clf3 = DecisionTreeClassifier(criterion='gini', splitter='best')
clf3.fit(X_train3, y_train3)
clf_pred3 = clf3.predict(X_test3)
print("\nCLASSIFICATION REPORT DECISION TREE(criterion='gini', splitter='best'):\n")
print(classification_report(y_test3, clf_pred3))
print("\nF1-score Decision Tree\n")
micro_f1_DT3 = f1_score(y_test3, clf_pred3, average='micro')
macro_f1_DT3 = f1_score(y_test3, clf_pred3, average='macro')
weighted_f1_DT3 = f1_score(y_test3, clf_pred3, average='weighted')
print(f"MICRO F1: {micro_f1_DT3:.3f}")
print(f"MACRO F1: {macro_f1_DT3:.3f}")
print(f"WEIGHTED F1: {weighted_f1_DT3:.3f}")

clf4 = DecisionTreeClassifier(criterion='gini', splitter='random')
clf4.fit(X_train3, y_train3)
clf_pred4 = clf4.predict(X_test3)
print("\nCLASSIFICATION REPORT DECISION TREE(criterion='gini', splitter='random'):\n")
print(classification_report(y_test3, clf_pred4))
print("\nF1-score Decision Tree\n")
micro_f1_DT4 = f1_score(y_test3, clf_pred4, average='micro')
macro_f1_DT4 = f1_score(y_test3, clf_pred4, average='macro')
weighted_f1_DT4 = f1_score(y_test3, clf_pred4, average='weighted')
print(f"MICRO F1: {micro_f1_DT4:.3f}")
print(f"MACRO F1: {macro_f1_DT4:.3f}")
print(f"WEIGHTED F1: {weighted_f1_DT4:.3f}")

print("\nF1-SCORE COMPARISON DECISION TREE GINI(best vs. random):")
print(f"MICRO:\t   {micro_f1_DT3:.3f}\t{micro_f1_DT4:.3f}")
print(f"MACRO:\t   {macro_f1_DT3:.3f}\t{macro_f1_DT4:.3f}")
print(f"WEIGHTED:  {weighted_f1_DT3:.3f}\t{weighted_f1_DT4:.3f}")