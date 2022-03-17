import pandas as pd
import numpy as np

df = pd.read_table('SMSSpamCollection', sep='\t', header=None, names=['label', 'sms_message'])
# print(df.head(5))

df['label'] = df.label.map({'ham': 0, 'spam': 1})
# print(df.head(5))

# print(df.shape)

Y_test = df.loc[[x for x in range(0,df.shape[0], 10)], ['label']]
X_test = df.loc[[x for x in range(0,df.shape[0], 10)], ['sms_message']]
Y_test = Y_test.reset_index(drop = True)
X_test = X_test.reset_index(drop = True)
        
# print("test_X:\n", X_test, "\ntest_y:\n", Y_test)

Y_train = df[~df.index.isin(Y_test.index)]['label']
X_train = df[~df.index.isin(X_test.index)][['sms_message']]
Y_train = Y_train.reset_index(drop = True)
X_train = X_train.reset_index(drop = True)

# print("train_X:\n", X_train, "\ntrain_y:\n", Y_train)

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

# print(X_train['sms_message'])

training_data = cv.fit_transform(X_train['sms_message'])
testing_data = cv.transform(X_test['sms_message'])

# frequency_matrix = pd.DataFrame(training_data, columns=cv.get_feature_names())
# print(training_data)

from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
nb.fit(training_data, Y_train)
predictions = nb.predict(testing_data)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print('Accuracy score: ', format(accuracy_score(Y_test, predictions)))
print('Precision score: ', format(precision_score(Y_test, predictions)))
print('Recall score: ', format(recall_score(Y_test, predictions)))
print('F1 score: ', format(f1_score(Y_test, predictions)))