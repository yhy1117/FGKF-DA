import numpy as np

#from keras.utils.np_utils import to_categorical


def define_tags(label):
    tag = -1
    if label == 'B':
        tag = 0
    elif label == 'M':
        tag = 1
    elif label == 'E':
        tag = 2
    elif label == 'S':
        tag = 3
    return tag

def tag2label(tag):
    label = 'X'
    if tag == 0:
        label = 'B'
    elif tag == 1:
        label = 'M'
    elif tag == 2:
        label = 'E'
    elif tag == 3:
        label = 'S'
    return label

def load_data_base(path):
    texts = []
    labels = []
    sentence = []
    sentence_lable = []
    file = open(path, 'r', encoding='utf')
    while 1:
        line_t = file.readline()
        if not line_t:
            break
        else:
            if len(line_t) < 2:
                texts.append(sentence)
                #sentence_lable = to_categorical(sentence_lable)
                labels.append(sentence_lable)
                sentence = []
                sentence_lable =[]
            else:
                char = line_t.split('\t')[0]
                label = line_t.split('\t')[1].split('\n')[0]
                label = define_tags(label)
                sentence.append(char)
                sentence_lable.append(label)
    file.close()
    #print(np.array(texts).shape)
    return texts, labels

def load_datas(path):
    texts = []
    labels = []
    sentence_len = []
    sentence = []
    sentence_lable = []
    length = 0
    file = open(path, 'r', encoding='utf')
    while 1:
        line_t = file.readline()
        if not line_t:
            break
        else:
            if len(line_t) < 2:
                texts.append(sentence)
                #sentence_lable = to_categorical(sentence_lable)
                labels.append(sentence_lable)
                sentence_len.append(length)
                length = 0
                sentence = []
                sentence_lable =[]
            else:
                length += 1
                char = line_t.split('\t')[0]
                label = line_t.split('\t')[1].split('\n')[0]
                label = define_tags(label)
                sentence.append(char)
                sentence_lable.append(label)
    file.close()
    #print(np.array(texts).shape)
    return texts, labels, sentence_len
