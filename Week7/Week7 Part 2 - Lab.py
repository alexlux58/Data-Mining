import pandas as pd

CSV_LOCATION = "healthcare-dataset-stroke-data.csv"
df = pd.read_csv(CSV_LOCATION)
# print(df)

# Remove Missing Values N/A and Unknown
drop_list = ["id","gender","age","hypertension","heart_disease","ever_married","work_type","Residence_type","avg_glucose_level","bmi","smoking_status","stroke"]
df.dropna(inplace=True)
for dl in drop_list:
  df.drop(df.index[df[dl] == 'Unknown'], inplace=True)
df = df.reset_index(drop = True)
df = df.drop(columns=['id'])

# print(df.columns, sep='\n')
# print(df)

x = df[['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type',
        'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status', 'stroke']]

# print("age mean: ", x['age'].mean(), '\n' , x['stroke'].value_counts())

x_dummies = pd.get_dummies(x)
# print(x.columns)

# drop binary column because all the values in the columns "ever_married_Yes" 
# and "Residence_type_Urban" are not needed because "ever_married_No" 
# and "Residence_type_Rural" will give the values needed to make a prediction

x_dummies.drop(columns=["ever_married_Yes","Residence_type_Urban"], inplace=True)
# print(*x_dummies.columns, sep='\n')

# Down Sample
from sklearn.utils import resample

X_minority = x_dummies[x_dummies.stroke == 1]
X_majority = x_dummies[x_dummies.stroke == 0]
X_majority_downsampled = resample(X_majority, replace=False, n_samples=len(X_minority), random_state=0)

# Split data - train and test
from sklearn.model_selection import train_test_split

y, X = X_majority_downsampled["stroke"], X_majority_downsampled.drop(columns="stroke")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# print("X_train:\n", X_train)
# print("y_train:\n", y_train)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import  precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt  

K = [1, 3, 5, 7, 9, 11, 13, 15]
metric = ["chebyshev", "euclidean", "manhattan"]
W = ["uniform", "distance"]

results_f1 = {}
results_precision = {}
results_recall = {}
results_accuracy = {}
results_confusion = {}

for k in K:
    for m in metric:
        for w in W:
            neigh = KNeighborsClassifier(n_neighbors=k, metric=m, weights=w)
            neigh.fit(X_train, y_train)
            pred = neigh.predict(X_test)
            results_f1[(k,m,w)] = f1_score(y_test, pred, average='weighted')
            results_precision[(k,m,w)] = precision_score(y_test, pred, average='weighted')
            results_recall[(k,m,w)] = recall_score(y_test, pred, average='weighted')
            results_accuracy[(k,m,w)] = accuracy_score(y_test, pred)
            results_confusion[(k,m,w)] = confusion_matrix(y_test, pred)
            
            # ROC curve needs to be debugged
            # print("\t\t{}".format(roc_curve(y_test, y_pred, pos_label=1)))
            # metric_dict[m]["roc_curve"].append(roc_curve(y_test, y_pred, pos_label=1))

max_key_f1 = max(results_f1, key=results_f1.get)
print(f"(K, metric, W): {max_key_f1}")
print(f"F1-SCORE:\n{results_f1[max_key_f1]}")

max_key_precision = max(results_precision, key=results_precision.get)
print(f"(K, metric, W): {max_key_precision}")
print(f"PRECISION:\n{results_precision[max_key_precision]}")

max_key_recall = max(results_recall, key=results_recall.get)
print(f"(K, metric, W): {max_key_recall}")
print(f"RECALL:\n{results_recall[max_key_recall]}")

max_key_accuracy = max(results_accuracy, key=results_accuracy.get)
print(f"(K, metric, W): {max_key_accuracy}")
print(f"ACCURACY:\n{results_accuracy[max_key_accuracy]}")

max_key_confusion = max(results_confusion, key=results_confusion.get)
print(f"(K, metric, W): {max_key_confusion}")
print(f"CONFUSION MATRIX:\n{results_confusion[max_key_confusion]}")

print(results_f1)