import pandas as pd
import numpy as np

df = pd.read_table('SMSSpamCollection', sep='\t', header=None, names=['label', 'sms_message'])

df['label'] = df.label.map({'ham': 0, 'spam': 1})

Y_test = df.loc[[x for x in range(0,df.shape[0], 10)], ['label']]
X_test = df.loc[[x for x in range(0,df.shape[0], 10)], ['sms_message']]
Y_test = Y_test.reset_index(drop = True)
X_test = X_test.reset_index(drop = True)

Y_train = df[~df.index.isin(Y_test.index)]['label']
X_train = df[~df.index.isin(X_test.index)][['sms_message']]
Y_train = Y_train.reset_index(drop = True)
X_train = X_train.reset_index(drop = True)

from sklearn.feature_extraction.text import CountVectorizer

# default values: lowercase = True, 
# token_pattern = (?u)\\b\\w\\w+\\b, ignores all punctuation marks and treats them as delimiters, 
# while accepting alphanumeric strings of length greater than or equal to 2, as individual tokens or words.
cv = CountVectorizer()

training_data = cv.fit_transform(X_train['sms_message'])
testing_data = cv.transform(X_test['sms_message'])

# print(training_data)

from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB(alpha=0.2)
nb.fit(training_data, Y_train)
predictions = nb.predict(testing_data)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

accuracy = accuracy_score(Y_test, predictions)
precision = precision_score(Y_test, predictions)
recall = recall_score(Y_test, predictions)
f1 = f1_score(Y_test, predictions)
confusion_mat = confusion_matrix(Y_test, predictions)

print(f"Accuracy score: {accuracy:.3f}\n")
print(f"Precision score: {precision:.3f}\n")
print(f"Recall score: {recall:.3f}\n")
print(f"F1 score: {f1:.3f}\n")
print(f"Confusion Matrix:\n {confusion_mat}")