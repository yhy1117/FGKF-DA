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

TRAIN_DATA_UNI = os.path.join(DATADIR, 'weibo/train.ner')
DEV_DATA_UNI = os.path.join(DATADIR, 'weibo/dev.ner')
TEST_DATA_UNI = os.path.join(DATADIR, 'weibo/test.ner')

TRAIN_DATA_UNI_PRED = os.path.join(DATADIR, 'weibo/pred/p_train_cap.txt')
DEV_DATA_UNI_PRED = os.path.join(DATADIR, 'weibo/pred/p_dev.txt')
TEST_DATA_UNI_PRED = os.path.join(DATADIR, 'weibo/pred/p_test.txt')

T_TRAIN_DATA_UNI = os.path.join(DATADIR, 'msr/train.ner')
T_DEV_DATA_UNI = os.path.join(DATADIR, 'weibo/dev.ner')
T_TEST_DATA_UNI = os.path.join(DATADIR, 'weibo/train.ner')

TEST_PRED_TAG = os.path.join(DIR, 'mid_result/weibo/cap_multi_tag.txt')

TEST_PRED_P = os.path.join(DIR, 'mid_result/weibo/cap_multi_p.txt')
T_TEST_PRED_P = os.path.join(DIR, 'mid_result/weibo/teacher_ctb_cap_multi_p.txt')
T_TEST_SENT_P = os.path.join(DIR,'mid_result/weibo/cap_multi_sent_p.txt')
ATTENTION_FILE = os.path.join(DIR, 'mid_result/weibo/cap_multi_attn.txt')
DEV_LOSS_FILE = os.path.join(DIR, 'mid_result/weibo/cap_multi_dev_loss.txt')


DATA_FILE = ['data']
MAX_LEN = 100
