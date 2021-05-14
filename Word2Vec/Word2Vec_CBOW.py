# -*- coding: utf-8 -*-
# Word2Vec CBOW
# code by @eunjikeam

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

def one_hot_encoding(word, word_dict):
    return np.eye(len(word_dict))[word_dict[word]]

def get_batch(dataset, batch_size):
    inputs = []
    labels = []
    random_index = np.random.choice(range(len(dataset)), batch_size, replace = False)
    
    for i in random_index:
        inputs.append(dataset[i][0])
        labels.append(dataset[i][1])
        
    return inputs, labels

# Model
class CBOW(nn.Module):
    def __init__(self, voc_size, embedding_size):
        super(CBOW, self).__init__()
        self.W = nn.Embedding(voc_size, embedding_size)
        self.WT = nn.Linear(embedding_size, voc_size, bias = False)
    
    def forward(self, X):
        # [batch_size, window_size*2]
        # One_hot_encoding : [batch_size, window_size*2, voc_size]
        p_layer = self.W(X) # projection_layer : [batch_size, window_size*2, embedding_size]
        p_layer = p_layer.mean(dim = 1) # mean_Weight = [batch_size, embedding_size]
        output = self.WT(p_layer)
        return output
    
if __name__ == '__main__':
    corpus = [
        'drink cold milk',
        'drink cold water',
        'drink cold cola',
        'drink sweet juice',
        'drink sweet cola',
        'eat delicious bacon',
        'eat sweet mango',
        'eat delicious cherry',
        'eat sweet apple',
        'juice with sugar',
        'cola with sugar',
        'mango is fruit',
        'apple is fruit',
        'cherry is fruit',
        'Berlin is Germany',
        'Boston is USA',
        'Mercedes from Germany',
        'Mercedes is car',
        'Ford from USA',
        'Ford is car'
    ]
    
    # create word dictionary
    word_list = ' '.join(corpus).split()
    word_set = list(set(word_list))
    word_dict = {w:i for i, w in enumerate(word_set)}
    
    window_size = 2 # set window size
    
    dataset = [] # create dataset
    for i in range(window_size,len(word_list)-window_size):
        context = [word_dict[word_list[i-ws]] for ws in range(window_size,0, -1)] + \
        [word_dict[word_list[i+ws]] for ws in range(1,window_size+1)] # left, right context
        target = word_dict[word_list[i]]
        dataset.append([context, target])
        
    batch_size = 10
    voc_size = len(word_dict)
    embedding_size = 2
    epochs = 10000
    
    model = CBOW(voc_size, embedding_size)
    criterion = nn.CrossEntropyLoss() # softmax
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # train
    for epoch in range(epochs):
        input_batch, label_batch = get_batch(dataset, batch_size)
        input_batch = torch.LongTensor(input_batch)
        label_batch = torch.LongTensor(label_batch)

        optimizer.zero_grad()
        output = model(input_batch)

        loss = criterion(output, label_batch)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 1000 == 0:
            print("Epoch : {}/{}, cost = {:.6f}".format(epoch+1, epochs, loss))
            
    W, _ = model.parameters()
    for i, label in enumerate(word_dict):
        x, y = W[i][0].item(), W[i][1].item()
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.show()