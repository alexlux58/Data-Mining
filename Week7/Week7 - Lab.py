from sklearn.datasets import load_wine
df = load_wine(as_frame=True)

# print(dir(df))
# print(df.DESCR)
# print("-------------------")
# print(df)

from sklearn.model_selection import train_test_split

X, Y = df.data, df.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.25, random_state=2)

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3, metric="manhattan")
neigh.fit(X_train, Y_train)
pred = neigh.predict(X_test)

from sklearn.metrics import classification_report, f1_score

print(classification_report(Y_test, pred, digits=2))

X_train_new, X_valid, Y_train_new, Y_valid = train_test_split(X_train, Y_train, test_size=.2, random_state=2)
K = [1, 5, 11, 17]
M = ["manhattan", "chebyshev", "hamming"]
W = ["uniform", "distance"]

results = {}
for k in K:
    for m in M:
        for w in W:
            neigh_new = KNeighborsClassifier(n_neighbors=k, metric=m, weights=w)
            neigh_new.fit(X_train_new, Y_train_new)
            pred_new = neigh_new.predict(X_valid)
            results[(k,m,w)] = f1_score(Y_valid, pred_new, average='weighted')

max_key = max(results, key=results.get)
print(max_key)
print(results[max_key])

# print(results)

neigh_fin = KNeighborsClassifier(n_neighbors=5, metric="manhattan", weights="distance")
neigh_fin.fit(X_train_new, Y_train_new)
pred_fin = neigh_fin.predict(X_test)
print(classification_report(Y_test, pred_fin))

from sklearn.model_selection import GridSearchCV

parameters = {'n_neighbors':[1, 5, 11, 17], 'metric': ["manhattan", "chebyshev", "hamming"], 'weights': ["uniform", "distance"]}
knn = KNeighborsClassifier()
clf = GridSearchCV(knn, parameters, cv=5)
clf.fit(X_train, Y_train)
print(clf.best_params_)
knn_best = clf.best_estimator_
prediction = knn_best.predict(X_test)
print(classification_report(Y_test, prediction))

from sklearn.tree import DecisionTreeClassifier

criterion = ["gini", "entropy"]
splitter = ["best", "random"]
paramsDT = {"criterion": criterion, "splitter": splitter}

from sklearn.metrics import f1_score

dtclf = DecisionTreeClassifier()
clf = GridSearchCV(dtclf, paramsDT, cv=5)
clf.fit(X_train, Y_train)
print(clf.best_params_)
dt_best = clf.best_estimator_
prediction = dt_best.predict(X_test)
print("F1-SCORE (DT GRIDCV): ", f1_score(Y_test, prediction, average="weighted"))