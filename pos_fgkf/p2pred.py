

def prob2pred(in_f,in_p,out_f):
    #in_f = open(config.T_TEST_DATA_UNI, 'r', encoding='utf')
    #in_p = open(config.T_TEST_PRED_P, 'r', encoding='utf')
    #out_f = open(config.TRAIN_DATA_UNI_PRED, 'w', encoding='utf')
    default_p = '0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;1'
    row = 0
    correct = 0
    sent = 0
    length = 0
    while 1:
        line = in_f.readline()
        if not line:
            break
        else:
            p_line = in_p.readline()
            if not p_line:
                break
            row += 1
            correct = row
            if len(line) > 3 and length < 60:
                length += 1
                char = line.split('\t')[0]
                out_f.write(char)
                out_f.write('\t')
                out_f.write(p_line)
            elif len(line) >3 and length >= 60:
                length += 1
                while 1:
                    char = line.split('\t')[0]
                    out_f.write(char)
                    out_f.write('\t')
                    out_f.write(default_p)
                    out_f.write('\n')
                    line = in_f.readline()
                    if len(line) > 3:
                        length += 1
                    else:
                        length = 0
                        out_f.write('\n')
                        sent += 1
                        break
            else:
                length = 0
                out_f.write('\n')
                sent += 1
                while 1:
                    if row % 61 == 0:
                        break
                    p_line = in_p.readline()
                    if not p_line:
                        break
                    row += 1
                    correct = row

    in_f.close()
    out_f.close()