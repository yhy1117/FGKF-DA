# -*- coding:utf-8 -*-
import numpy as np
from collections import defaultdict
filename='embedding/character.vec'


def loadWord2Vec(filename):
    vocab = []
    embd = []
    word2idx = defaultdict(int)
    idx = 0
    fr = open(filename,'r')
    line = fr.readline().strip()
    word_dim = 100
    #vocab.append("unk")
    #embd.append([0]*word_dim)
    for line in fr :
        row = line.strip().split(' ')
        word2idx[row[0]] = idx
        vocab.append(row[0])
        embd.append(row[1:])
        idx += 1
    fr.close()
    #embd = np.asarray(embd)
    #embd = embd.astype('float32')
    return word2idx,vocab,embd

word2idx,vocab,embd = loadWord2Vec(filename)
vocab_size = len(vocab)
embedding_dim = len(embd[0])
embedding = np.asarray(embd)
embedding = embedding.astype('float32')



