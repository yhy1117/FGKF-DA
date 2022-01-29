def define_tags(label):
    tag = 0
    if label == 'O':
        tag = 0
    elif label == 'B-PER':
        tag = 1
    elif label == 'I-PER':
        tag = 2
    elif label == 'B-LOC':
        tag = 3
    elif label == 'I-LOC':
        tag = 4
    elif label == 'B-ORG':
        tag = 5
    elif label == 'I-ORG':
        tag = 6

    return tag


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
