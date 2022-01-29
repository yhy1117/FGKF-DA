# -*- coding:utf-8 -*-:w
import os

#Baseline
DROP_SINGLE = 0.2
LSTM_NET = True
STACK_STATUS =False
BI_DIRECTION = True
BI_GRAM = True
dense_hidden = [100, 2]

DIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(DIR, 'data')
MODEL_DIR = os.path.join(DIR, 'models')

TRAIN_DATA_UNI = os.path.join(DATADIR, 'nlpcc/train_1000.pos')
DEV_DATA_UNI = os.path.join(DATADIR, 'nlpcc/dev.pos')
TEST_DATA_UNI = os.path.join(DATADIR, 'nlpcc/test.pos')

TRAIN_DATA_UNI_PRED = os.path.join(DATADIR, 'nlpcc/pred/p_train_1000_cap_warm.txt')
DEV_DATA_UNI_PRED = os.path.join(DATADIR, 'nlpcc/pred/p_dev.txt')
TEST_DATA_UNI_PRED = os.path.join(DATADIR, 'nlpcc/pred/p_test.txt')

T_TRAIN_DATA_UNI = os.path.join(DATADIR, 'ctb6/train.pos')
T_DEV_DATA_UNI = os.path.join(DATADIR, 'nlpcc/dev.pos')
T_TEST_DATA_UNI = os.path.join(DATADIR, 'nlpcc/test.pos')

TEST_PRED_TAG = os.path.join(DIR, 'mid_result/NLPCC/cap_multi_warm_tag.txt')

TEST_PRED_P = os.path.join(DIR, 'mid_result/NLPCC/cap_multi_warm_p.txt')
T_TEST_PRED_P = os.path.join(DIR, 'mid_result/NLPCC/teacher_ctb_cap_multi_warm_p.txt')
T_TEST_SENT_P = os.path.join(DIR,'mid_result/NLPCC/cap_multi_sent_warm_p.txt')
ATTENTION_FILE = os.path.join(DIR, 'mid_result/NLPCC/cap_multi_warm_attn.txt')
DEV_LOSS_FILE = os.path.join(DIR, 'mid_result/NLPCC/dev_loss/cap_multi_warm_dev_loss.txt')


DATA_FILE = ['data']
MAX_LEN = 60
