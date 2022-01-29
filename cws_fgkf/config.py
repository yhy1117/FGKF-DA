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

WORD_VEC_100 = os.path.join(MODEL_DIR, 'vec100.txt')

TRAIN_PATH = os.path.join(DATADIR, 'train')
DEV_PATH = os.path.join(DATADIR, 'dev')
TEST_PATH = os.path.join(DATADIR, 'test')
WORD_SINGLE = os.path.join(DATADIR, 'freq.txt')

WORD_DICT = os.path.join(DATADIR, 'word_dictionary.txt')

TRAIN_DATA_UNI = os.path.join(DATADIR, 'ZX/ZX.train_new.seg')
DEV_DATA_UNI = os.path.join(DATADIR, 'ZX/ZX.dev.seg')
TEST_DATA_UNI = os.path.join(DATADIR, 'ZX/ZX.test.seg')

T_TRAIN_DATA_UNI = os.path.join(DATADIR, 'ctb6/ctb6.train.seg.clear')
T_DEV_DATA_UNI = os.path.join(DATADIR, 'ZX/ZX.dev.seg')
T_TEST_DATA_UNI = os.path.join(DATADIR, 'ZX/ZX.train_new.seg')

TRAIN_DATA_UNI_PRED = os.path.join(DATADIR, 'ZX/pred/teacher_ctb/p_ZX_ctb6_train_capsule_caps_num_60_100_step_multi_p.txt')
DEV_DATA_UNI_PRED = os.path.join(DATADIR, 'ZX/pred/teacher_ctb/p_ZX_dev.txt')
TEST_DATA_UNI_PRED = os.path.join(DATADIR, 'ZX/pred/teacher_ctb/p_ZX_test.txt')

TEST_GOLD_SENT = os.path.join(DATADIR, 'ZX/ZX_test_gold_sentence.txt')
TEST_PRED_SENT = os.path.join(DATADIR, 'ZX/ZX_test_pred_sentence.txt')

TEST_PRED_TAG = os.path.join(DIR, 'mid_result/ZX/ZX_ctb6_joint_train_alpha_0.5_capsule_caps_num_60_100_step_multi_tag.txt')
TEST_PRED_LABEL = os.path.join(DIR, 'mid_result/ZX/CTB_pretrain_pred_label.txt')

TEST_PRED_P = os.path.join(DIR, 'mid_result/ZX/teacher_ctb_pred_ZX_test_alpha_0.5_capsule_caps_num_60_100_step_multi_p.txt')
T_TEST_PRED_P = os.path.join(DIR, 'mid_result/ZX/teacher_ctb_pred_ZX_train_alpha_0.5_capsule_caps_num_60_100_step_multi_p.txt')
T_TEST_SENT_P = os.path.join(DIR,'mid_result/ZX/ZX_multi_sent_alpha.txt')
ATTENTION_FILE = os.path.join(DIR, 'mid_result/ZX/ZX_multi_char_alpha.txt')
DEV_LOSS_FILE = os.path.join(DIR, 'mid_result/ZX/ZX_multi_dev_loss.txt')

TRAIN_DATA_BI = os.path.join(DATADIR, 'train.csv')
DEV_DATA_BI = os.path.join(DATADIR, 'dev.csv')
TEST_DATA_BI = os.path.join(DATADIR, 'test.csv')

DATA_FILE = ['data']
MAX_LEN = 100
