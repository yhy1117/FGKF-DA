# -*- coding:utf-8 -*-
import numpy as np
from collections import defaultdict
filename='embedding/Wiki100.txt'

def loadWord2Vec(filename):
    vocab = []
    embd = []
    word2idx = defaultdict(int)
    idx = 0
    fr = open(filename, 'r', encoding='utf')
    for line in fr:
        if len(line) > 10:
            row = line.strip().split(' ')
            word2idx[row[0]] = idx
            vocab.append(row[0])
            embd.append(row[1:])
            idx += 1
    fr.close()
    return word2idx,vocab,embd

word2idx,vocab,embd = loadWord2Vec(filename)
vocab_size = len(vocab)
embedding_dim = len(embd[0])

embedding = np.asarray(embd, 'float32')