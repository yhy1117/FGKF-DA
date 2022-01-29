# -*- coding:utf-8 -*-
import numpy as np
import embed

def build_bi_vocabulary(char, label, pred, max_sentence_length, bi_gram=True ):
    x_text_idx = []
    y_label_idx = []
    y_label_pred_idx = []

    # count = 0
    for x in char:
        x_idx = np.zeros(max_sentence_length * 5)
        for k in range(min(len(x), max_sentence_length)):
            x_mid = x[k]
            x_mid_index = embed.char2idx[x_mid]
            if k == 0 and len(x) > 1:
                x_back = x[k + 1]
                x_for_char_index = embed.char2idx['</s>']
                x_back_char_index = embed.char2idx[x_back]
                x_for_index = embed.char2idx['</s>'+x_mid]
                x_back_index = embed.char2idx[x_mid + x_back]
            elif k == len(x) - 1:
                x_for = x[k - 1]
                x_for_char_index = embed.char2idx[x_for]
                x_back_char_index = embed.char2idx['</s>']
                x_for_index = embed.char2idx[x_for + x_mid]
                x_back_index = embed.char2idx[x_mid+'</s>']
            else:
                x_for = x[k-1]
                x_back = x[k+1]
                x_for_char_index = embed.char2idx[x_for]
                x_back_char_index = embed.char2idx[x_back]
                x_for_index = embed.char2idx[x_for+x_mid]
                x_back_index = embed.char2idx[x_mid+x_back]
            x_idx[5*k] = x_for_char_index
            x_idx[5*k+1] = x_for_index
            x_idx[5*k+2] = x_mid_index
            x_idx[5*k+3] = x_back_index
            x_idx[5*k+4] = x_back_char_index
        x_text_idx.append(x_idx)

    train_x = np.array(x_text_idx)


    for y in label:
        y_idx = np.zeros(max_sentence_length)
        for i in range(min(len(y),max_sentence_length)):
            y_idx[i] = y[i]
        y_label_idx.append(y_idx)
    train_label = np.array(y_label_idx)

    for y_pred in pred:
        y_pred_idx = np.zeros([max_sentence_length, 4])
        for i in range(min(len(y_pred), max_sentence_length)):
            y_pred_idx[i] = y_pred[i]
        y_label_pred_idx.append(y_pred_idx)
    train_pred = np.array(y_label_pred_idx)

    return train_x,train_label,train_pred

def build_bi_vocabulary_t(char, label, max_sentence_length, bi_gram=True ):
    x_text_idx = []
    y_label_idx = []
    y_label_pred_idx = []

    # count = 0
    for x in char:
        x_idx = np.zeros(max_sentence_length * 5)
        for k in range(min(len(x), max_sentence_length)):
            x_mid = x[k]
            x_mid_index = embed.char2idx[x_mid]
            if k == 0 and len(x) > 1:
                x_back = x[k + 1]
                x_for_char_index = embed.char2idx['</s>']
                x_back_char_index = embed.char2idx[x_back]
                x_for_index = embed.char2idx['</s>'+x_mid]
                x_back_index = embed.char2idx[x_mid + x_back]
            elif k == len(x) - 1:
                x_for = x[k - 1]
                x_for_char_index = embed.char2idx[x_for]
                x_back_char_index = embed.char2idx['</s>']
                x_for_index = embed.char2idx[x_for + x_mid]
                x_back_index = embed.char2idx[x_mid+'</s>']
            else:
                x_for = x[k-1]
                x_back = x[k+1]
                x_for_char_index = embed.char2idx[x_for]
                x_back_char_index = embed.char2idx[x_back]
                x_for_index = embed.char2idx[x_for+x_mid]
                x_back_index = embed.char2idx[x_mid+x_back]
            x_idx[5*k] = x_for_char_index
            x_idx[5*k+1] = x_for_index
            x_idx[5*k+2] = x_mid_index
            x_idx[5*k+3] = x_back_index
            x_idx[5*k+4] = x_back_char_index
        x_text_idx.append(x_idx)

    train_x = np.array(x_text_idx)


    for y in label:
        y_idx = np.zeros(max_sentence_length)
        for i in range(min(len(y),max_sentence_length)):
            y_idx[i] = y[i]
        y_label_idx.append(y_idx)
    train_label = np.array(y_label_idx)

    return train_x, train_label