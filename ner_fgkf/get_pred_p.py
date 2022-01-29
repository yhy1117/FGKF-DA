# char \t prob[18]
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
            if len(line_t) < 4:
                texts.append(sentence)
                #sentence_lable = to_categorical(sentence_lable)
                labels.append(sentence_lable)
                sentence = []
                sentence_lable =[]
            else:
                char = line_t.split('\t')[0]
                prob = []
                p = line_t.split('\t')[1]
                for i in range(7):
                    prob.append(float(p.split(';')[i]))
                #label = define_tags(label)
                sentence.append(char)
                sentence_lable.append(prob)
    file.close()
    #print(np.array(texts).shape)
    return labels


