# -*- coding:utf-8 -*-
import numpy as np
from config import BI_GRAM
from config import WORD_SINGLE
from collections import defaultdict
filename='embedding/character.vec'
#filename = 'embedding/gold_ctb/ctb_gold_20000_embed.txt'
frequency = 3#只保留词频大于15的bigram

def loadWord2Vec(filename,bigram=BI_GRAM):
    vocab = []
    embd = []
    char2idx = defaultdict(int)
    idx = 0
    fr = open(filename,'r',encoding='utf')
    line = fr.readline().strip()
    word_dim = 100
    #vocab.append("unk")
    #embd.append([0]*word_dim)
    for line in fr :
        row = line.strip().split(' ')
        char2idx[row[0]] = idx
        vocab.append(row[0])
        embd.append(row[1:])
        idx += 1
    fr.close()
    #embd = np.asarray(embd)
    #embd = embd.astype('float32')
    if bigram:
        fw = open(WORD_SINGLE, 'r', encoding='utf')
        for line in fw:
            mean_vec = np.zeros(word_dim)
            word = line.split(', ')[0]
            freq = int(line.split(', ')[1].split(')')[0])
            if freq > frequency:
                vocab.append(word)
                for ch in word:
                    if ch in char2idx:
                        mean_vec += np.asarray(embd[char2idx[ch]], 'float32')
                    else:
                        mean_vec += np.asarray(embd[char2idx['<OOV>']], 'float32')

                word_vec = mean_vec / 2.0
                char2idx[word] = idx
                embd.append(word_vec)
                idx += 1
        fw.close()

    return char2idx,vocab,embd

char2idx,vocab,embd = loadWord2Vec(filename)
vocab_size = len(vocab)
embedding_dim = len(embd[0])
embedding = np.asarray(embd)
embedding = embedding.astype('float32')


