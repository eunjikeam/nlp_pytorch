# -*- coding: utf-8 -*-
# One-Hot Encoding 
# code by @eunjikeam

import numpy as np

def one_hot_encoding(word, word_dict):
    one_hot_vector = np.zeros(len(word_dict)) # 단어사전 길이만크의 0벡터 생성
    one_hot_vector[word_dict[word]] = 1 # 해당하는 단어의 index 자리에 1 부여
    return one_hot_vector

if __name__ == '__main__':
    
    sentence = "I have a book and I like romance"
    token = sentence.split()
    
    # create vocabulary
    word_set = list(set(token))
    word_dict = {w:i for i, w in enumerate(word_set)}

    print('One-hot vector of "I" is ', one_hot_encoding('I', word_dict))
    
    word_list = []
    for w in token:
        word_list.append(one_hot_encoding(w, word_dict))
    
    print('One-hot vector of full sentence is \n', np.array(word_list))