def define_tags(label):
    tag = 17
    if label == 'NN':
        tag = 0
    elif label == 'VV':
        tag = 1
    elif label == 'PU':
        tag = 2
    elif label == 'AD':
        tag = 3
    elif label == 'NR':
        tag = 4
    elif label == 'CD':
        tag = 5
    elif label == 'P':
        tag = 6
    elif label == 'M':
        tag = 7
    elif label == 'DEC':
        tag = 8
    elif label == 'JJ':
        tag = 9
    elif label == 'PN':
        tag = 10
    elif label == 'NT':
        tag = 11
    elif label == 'LC':
        tag = 12
    elif label == 'VA':
        tag = 13
    elif label == 'CC':
        tag = 14
    elif label == 'DT':
        tag = 15
    elif label == 'AS':
        tag = 16
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

def load_datas(path):
    texts = []
    labels = []
    sentence = []
    sentence_lable = []
    file = open(path, 'r')
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
                word = line_t.strip().split('\t')[0]
                label = line_t.strip().split('\t')[1]
                label = define_tags(label)
                sentence.append(word)
                sentence_lable.append(label)
    file.close()
    #print(np.array(texts).shape)
    return texts, labels
