import pandas as pd

df = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['label', 'sms'])

df['label'] = df.label.map({'ham': 0, 'spam': 1})

test = df.loc[[x for x in range(0,df.shape[0], 10)], ['label', 'sms']]
test = test.reset_index(drop = True)

train = df[~df.index.isin(test.index)][['label', 'sms']]
train = train.reset_index(drop = True)

X_train, y_train = train.sms, train.label
X_test, y_test = test.sms, test.label

from sklearn.feature_extraction.text import CountVectorizer

# default values: lowercase = True, 
# token_pattern = (?u)\\b\\w\\w+\\b, ignores all punctuation marks and treats them as delimiters, 
# while accepting alphanumeric strings of length greater than or equal to 2, as individual tokens or words.
cv = CountVectorizer()
train_X = cv.fit_transform(X_train)

train_X_df = pd.DataFrame(train_X.A, columns=cv.get_feature_names_out())
train_X_df.insert(0, "SMS", X_train.tolist())
train_X_df.insert(0, "Class", y_train.values.tolist())

# Ham and Spam Messages
ham_sms = train_X_df[train_X_df["Class"] == 0]
spam_sms = train_X_df[train_X_df["Class"] == 1]

# P(Ham) and P(Spam)
y_train_p_ham = len(ham_sms) / len(train_X_df)
y_train_p_spam = len(spam_sms) / len(train_X_df)

# Number of ham words
n_words_per_ham_message = ham_sms['SMS'].apply(len)
n_ham = n_words_per_ham_message.sum()

# Number of spam words
n_words_per_spam_message = spam_sms['SMS'].apply(len)
n_spam = n_words_per_spam_message.sum()

unique_words = cv.get_feature_names_out()

# P(words|ham) and P(words|spam)
p_words_ham = {word:0 for word in unique_words}
p_words_spam = {word:0 for word in unique_words}

alpha = 0.20
N = 20000

# P(words|spam) or P(words|ham)
for word in unique_words:
  n_word_given_ham = ham_sms[word].sum()
  p_word_given_ham = (n_word_given_ham + alpha) / (n_ham + alpha * N)
  p_words_ham[word] = p_word_given_ham

  n_word_given_spam = spam_sms[word].sum()
  p_word_given_spam = (n_word_given_spam + alpha) / (n_spam + alpha * N)
  p_words_spam[word] = p_word_given_spam
  
predictions = []
for message in X_test:
  sms = message.split()
  p_ham_words_given = 1
  p_spam_words_given = 1
  for word in sms:
    if word in p_words_ham: 
      p_ham_words_given *= p_words_ham[word]
    if word in p_words_spam:
      p_spam_words_given *= p_words_spam[word]

  p_ham_words_given *= y_train_p_ham
  p_spam_words_given *= y_train_p_spam

  if p_ham_words_given > p_spam_words_given:
    predictions.append(0)
  elif p_ham_words_given < p_spam_words_given:
    predictions.append(1)
  else:
    predictions.append(0)

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
    
print(f"\nAccuracy: {accuracy_score(y_test, predictions):.3f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, predictions)}")
print(f"\nPrecision Score: {precision_score(y_test, predictions, average='weighted'):.3f}")
print(f"\nRecall Score: {recall_score(y_test, predictions, average='weighted'):.3f}")
print(f"\nF1 Score: {f1_score(y_test, predictions, average='weighted'):.3f}")