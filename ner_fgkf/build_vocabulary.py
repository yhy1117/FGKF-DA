# -*- coding:utf-8 -*-
import numpy as np
import embed

def build_vocabulary(sent, label, pred, max_sentence_length):
    x_text_idx = []
    y_label_idx = []
    y_label_pred_idx = []
    # count = 0
    for x in sent:
        x_idx = np.zeros(max_sentence_length)
        for i in range(min(len(x),max_sentence_length)):
            x_mid = x[i]
            x_idx[i] = embed.word2idx[x_mid]
        x_text_idx.append(x_idx)
    train_x = np.array(x_text_idx)

    for y in label:
        y_idx = np.zeros(max_sentence_length)
        for i in range(min(len(y),max_sentence_length)):
            y_idx[i] = y[i]
        y_label_idx.append(y_idx)
    train_label = np.array(y_label_idx)

    for y_pred in pred:
        y_pred_idx = np.zeros([max_sentence_length, 7])
        for i in range(min(len(y_pred), max_sentence_length)):
            y_pred_idx[i] = y_pred[i]
        y_pred_idx = y_pred_idx.flatten()
        y_label_pred_idx.append(y_pred_idx)
    train_pred = np.array(y_label_pred_idx)

    return train_x,train_label,train_pred

def build_vocabulary_t(sent, label, max_sentence_length):
    x_text_idx = []
    y_label_idx = []
    # count = 0
    for x in sent:
        x_idx = np.zeros(max_sentence_length)
        for i in range(min(len(x),max_sentence_length)):
            x_mid = x[i]
            x_idx[i] = embed.word2idx[x_mid]
        x_text_idx.append(x_idx)
    train_x = np.array(x_text_idx)

    for y in label:
        y_idx = np.zeros(max_sentence_length)
        for i in range(min(len(y),max_sentence_length)):
            y_idx[i] = y[i]
        y_label_idx.append(y_idx)
    train_label = np.array(y_label_idx)

    return train_x,train_label