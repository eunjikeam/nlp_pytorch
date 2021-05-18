# -*- coding: utf-8 -*-
# One-Hot Encoding 
# code by @eunjikeam

import numpy as np

def BoW(sentence, word_dict):
    token = sentence.split() # split sentence
    bow = list(np.zeros(len(word_dict), dtype = int)) # bag of words
    for w in token:
        if w not in word_dict.keys():  # out of vocabulary
            word_dict['unknown'] = word_dict['unknown'] + 1
        else:
            bow[word_dict[w]] = bow[word_dict[w]] + 1
    return bow

if __name__ == '__main__':
    sentence = "I have a book and I like romance"
    token = sentence.split()
    print('sentence : ', sentence)

    # create vocabulary
    word_set = list(set(token))
    word_dict = {w:i for i, w in enumerate(word_set)}
    word_dict['unknown'] = len(word_dict)
    print('Vocabulary : ', word_dict)

    #BoW
    print('BoW of sentence : ', BoW(sentence, word_dict))
