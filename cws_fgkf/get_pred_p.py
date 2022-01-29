import numpy as np

#from keras.utils.np_utils import to_categorical


def load_datas(path):
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
                prob = []
                char = line_t.split('\t')[0]
                p = line_t.split('\t')[1].split('\n')[0]
                p_B = float(p.split(';')[0])
                p_M = float(p.split(';')[1])
                p_E = float(p.split(';')[2])
                p_S = float(p.split(';')[3])
                prob.append(p_B)
                prob.append(p_M)
                prob.append(p_E)
                prob.append(p_S)
                #label = define_tags(label)
                sentence.append(char)
                sentence_lable.append(prob)
    file.close()
    #print(np.array(texts).shape)
    return labels


#labels = load_datas('E:/Python_Projects/seg_multi_loss/data/NLPCC/pred/CTB_train_CTB_dev_pred_p.txt')
#print(labels)
