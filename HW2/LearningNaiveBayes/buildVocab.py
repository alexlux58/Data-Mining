import pandas as pd
import nltk
from nltk.corpus import words

vocubulary = {}
data = pd.read_csv("emails.csv")
nltk.download('words')
set_words = set(words.words())

def build_vocabulary(current_email):
    idx = len(vocubulary)
    
    for word in current_email:
        if word.lower() not in vocubulary and word.lower() in set_words:
            vocubulary[word] = idx
            idx += 1
            
if __name__ == '__main__':
    for i in range(data.shape[0]):
        current_email = data.iloc[i,0].split()
        print(current_email)
        print(f'Current email is {i}/{data.shape[0]} and the length of vocabulary is {len(vocubulary)}')
        build_vocabulary(current_email)
    
    file = open("vocabulary.txt", "w")
    file.write(str(vocubulary))
    file.close()