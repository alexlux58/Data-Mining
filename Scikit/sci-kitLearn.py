import json
import random

class Sentiment:
    NEGATIVE = "NEGATIVE"
    POSITIVE = "POSITIVE"
    NEUTRAL = "NEUTRAL"

class Review:
    def __init__(self, text, score):
        self.text = text
        self.score = score
        self.sentiment = self.get_sentiment()

    def get_sentiment(self):
        if self.score <= 2:
            return Sentiment.NEGATIVE
        elif self.score == 3:
            return Sentiment.NEUTRAL
        else:
            return Sentiment.POSITIVE

class ReviewContainer:
    def __init__(self, reviews):
        self.reviews = reviews
    
    def get_text(self):
        return [x.text for x in self.reviews]

    def get_sentiment(self):
        return [x.sentiment for x in self.reviews]
    
    def evenly_distribute(self):
        negative = list(filter(lambda x: x.sentiment == Sentiment.NEGATIVE, self.reviews))
        positive = list(filter(lambda x: x.sentiment == Sentiment.POSITIVE, self.reviews))
        positive_shrunk = positive[:len(negative)]
        self.reviews = negative + positive_shrunk
        random.shuffle(self.reviews)
        '''
        print(negative[0].text)
        print(len(negative))
        print(len(positive))
        '''

file_name = "Books_small_10000.json"

reviews = []

with open(file_name) as f:
    for line in f:
        review = json.loads(line)
        reviews.append(Review(review['reviewText'], review['overall']))

# print(reviews[5].score)

from sklearn.model_selection import train_test_split

training, test = train_test_split(reviews, test_size=.33, random_state=42)

train_container = ReviewContainer(training)
train_container.evenly_distribute()

test_container = ReviewContainer(test)
test_container.evenly_distribute()

# print(training[0].text)

train_x = train_container.get_text()
train_y = train_container.get_sentiment()

test_x = test_container.get_text()
test_y = test_container.get_sentiment()

print(train_y.count(Sentiment.POSITIVE))
print(train_y.count(Sentiment.NEGATIVE))

# print(train_x[0])
# print(train_y[0])

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# (term frequency inverse document frequency) = tfidf

# vectorizor = CountVectorizer()
vectorizor = TfidfVectorizer()

# corpus = train_x
# ---TRANSFORMER---
# standardization: mean = 0, standard deviation = 1
# .fit() computes mean and standard deviation (in the case of standard scalar transformer)
# .transform() applies the transformer formula to each data point 
# .fit() and .transform() -> training
# .transform() -> test
# the above method helps overcome over fiting data

# ---MODEL---
# models learn: parameters and weights (models do not change data)
# .fit() -> training
# .predict() -> test/new data

train_x_vectors = vectorizor.fit_transform(train_x)
test_x_vectors = vectorizor.transform(test_x)

# print(test_x_vectors[0])

# TODO: Make Multi threaded

from sklearn import svm
# svm (support vector machine), SVC (support vector classifier)
clf_svm = svm.SVC(kernel="linear")
clf_svm.fit(train_x_vectors, train_y)
print(clf_svm.predict(test_x_vectors[0]))

from sklearn.tree import DecisionTreeClassifier
clf_dec = DecisionTreeClassifier()
clf_dec.fit(train_x_vectors, train_y)
print(clf_dec.predict(test_x_vectors[0]))

from sklearn.naive_bayes import GaussianNB
clf_gnb = GaussianNB()
clf_gnb.fit(train_x_vectors.toarray(), train_y)
print(clf_gnb.predict(test_x_vectors[0].toarray()))

from sklearn.linear_model import LogisticRegression
clf_log = LogisticRegression(max_iter=500)
clf_log.fit(train_x_vectors, train_y)
print(clf_log.predict(test_x_vectors[0]))

test_set = ["very good", "very bad", "I do not recommend"]
new_set = vectorizor.transform(test_set)
print(clf_svm.predict(new_set))

# Mean Accuracy
print(clf_svm.score(test_x_vectors.toarray(), test_y))
print(clf_dec.score(test_x_vectors.toarray(), test_y))
print(clf_gnb.score(test_x_vectors.toarray(), test_y))
print(clf_log.score(test_x_vectors.toarray(), test_y))


# F1 Scores
from sklearn.metrics import f1_score
print(f1_score(test_y, clf_svm.predict(test_x_vectors.toarray()), average=None, labels=[Sentiment.POSITIVE ,Sentiment.NEGATIVE]))
print(f1_score(test_y, clf_dec.predict(test_x_vectors.toarray()), average=None, labels=[Sentiment.POSITIVE ,Sentiment.NEGATIVE]))
print(f1_score(test_y, clf_gnb.predict(test_x_vectors.toarray()), average=None, labels=[Sentiment.POSITIVE ,Sentiment.NEGATIVE]))
print(f1_score(test_y, clf_log.predict(test_x_vectors.toarray()), average=None, labels=[Sentiment.POSITIVE ,Sentiment.NEGATIVE]))

from sklearn.model_selection import GridSearchCV

parameters = {
    'kernel': ('linear', 'rbf'), 
    'C': (1,4,8,16,32)
    }

svc = svm.SVC()
clf = GridSearchCV(svc, parameters, cv=5)
clf.fit(train_x_vectors, train_y)
print(clf.score(test_x_vectors.toarray(), test_y))

import pickle

with open('sentiment_classifier.pkl', 'wb') as f:
    pickle.dump(clf, f)

with open('sentiment_classifier.pkl', 'rb') as f:
    loaded_clf = pickle.load(f)
    print(loaded_clf.predict(test_x_vectors[0]))