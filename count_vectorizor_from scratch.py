documents = ['Hello, how are you!',
             'Win money, win from home.',
             'Call me now.',
             'Hello, Call hello you tomorrow?']

lower_case_documents = [i.lower() for i in documents]
    
print(lower_case_documents)

import string

sans_punctuation_documents = [i.translate(str.maketrans('', '', string.punctuation)) for i in lower_case_documents]
    
print(sans_punctuation_documents)

preprocessed_documents = [i.split() for i in sans_punctuation_documents]
    
print(preprocessed_documents)

import pprint
from collections import Counter

frequency_list = [Counter(i) for i in preprocessed_documents]
    
pprint.pprint(frequency_list)