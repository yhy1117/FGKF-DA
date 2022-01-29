# -*- coding:utf-8 -*-
import os
import datetime
import numpy as np
import tensorflow as tf

import config as config
from config import DROP_SINGLE, BI_DIRECTION, BI_GRAM, STACK_STATUS, LSTM_NET, TEST_PRED_TAG

from model import Model
import load_data
import embed
import build_vocabulary
import data_helpers
import get_pred_p

# Data parameters
tf.flags.DEFINE_integer("word_dim", 100, "word_dim")
tf.flags.DEFINE_integer("lstm_dim", 100, "lstm_dim")
tf.flags.DEFINE_integer("num_classes", 7, "num_classes")

# model names
tf.flags.DEFINE_string("model_name", "ner_"+"weibo_capsule", "model name")
tf.flags.DEFINE_string("rout_mode", "rout", "rout or Rrout")
tf.flags.DEFINE_string("attn_mode","matrix dot","dot,matrix dot or mlp")
tf.flags.DEFINE_string("model_type","type III", "type I,II,III")

# Model Hyperparameters[t]
tf.flags.DEFINE_float("lr", 0.01, "learning rate (default: 0.01)")
tf.flags.DEFINE_float("dropout_keep_prob", DROP_SINGLE, "Dropout keep probability (default: 0.8)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.100, "L2 regularizaion lambda (default: 0.5)")
tf.flags.DEFINE_float("clip", 5, "grident clip")
tf.flags.DEFINE_float("alpha", 0.5, "Student reference seg loss rate (default: 0.5)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 40)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("alpha_decay", 0, "Student reference seg loss rate decay (default: 0)")
tf.flags.DEFINE_integer("out_caps_num", 60, "Number of output capsules (default: 30)")
tf.flags.DEFINE_integer("rout_iter", 3, "Iterations of routing (default:3)")

# Misc Parameters
tf.flags.DEFINE_boolean("is_block", False, "teacher gradient block or not")
tf.flags.DEFINE_boolean("embed_status", False, "embed_status")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log pl:acement of ops on devices")

FLAGS = tf.flags.FLAGS
#FLAGS._parse_flags()

if FLAGS.embed_status is False:
    init_embedding = None
else:
    print('get initialized embedding...')
    init_embedding = embed.embedding

#print("\nParameters:")
#for attr, value in sorted(FLAGS.__flags.items(),reverse=True):
#    print("{}={}".format(attr.upper(), value))
#print("")

print('load student data...')
#load train data
print('loading train data...')
train_char_s, train_label_s = load_data.load_datas(config.TRAIN_DATA_UNI)
#train_pred = get_pred_p.load_datas(config.TRAIN_DATA_UNI_PRED)

# Build vocabulary
print("Build vocabulary...")
max_sentene_length = config.MAX_LEN
#train_x_s, train_label_s, train_pred = build_bi_vocab.build_bi_vocabulary(train_char_s, train_label_s, train_pred, max_sentene_length, BI_GRAM)

#load dev data
print('loading dev data...')
dev_char_s, dev_label_s = load_data.load_datas(config.DEV_DATA_UNI)
dev_pred = get_pred_p.load_datas(config.DEV_DATA_UNI_PRED)
#max_dev_sentene_length_s = max([len(x) for x in dev_char_s])
dev_x_s, dev_label_s, dev_pred = build_vocabulary.build_vocabulary(dev_char_s, dev_label_s, dev_pred, max_sentene_length)

#load test data
print('loading test data...')
test_char_s, test_label_s = load_data.load_datas(config.TEST_DATA_UNI)
test_pred = get_pred_p.load_datas(config.TEST_DATA_UNI_PRED)
#max_test_sentene_length = max([len(x) for x in test_char])
test_x_s, test_label_s, test_pred = build_vocabulary.build_vocabulary(test_char_s, test_label_s, test_pred, max_sentene_length)

print('load teacher data...')
print('loading train data...')
train_char_t, train_label_t = load_data.load_datas(config.T_TRAIN_DATA_UNI)

# Build vocabulary
print("Build vocabulary...")
max_sentene_length = config.MAX_LEN
train_x_t, train_label_t = build_vocabulary.build_vocabulary_t(train_char_t, train_label_t, max_sentene_length)

#load dev data
print('loading dev data...')
dev_char_t, dev_label_t = load_data.load_datas(config.T_DEV_DATA_UNI)
#max_dev_sentene_length_s = max([len(x) for x in dev_char_s])
dev_x_t, dev_label_t = build_vocabulary.build_vocabulary_t(dev_char_t, dev_label_t, max_sentene_length)

#load test data
print('loading test data...')
test_char_t, test_label_t = load_data.load_datas(config.T_TEST_DATA_UNI)
#max_test_sentene_length = max([len(x) for x in test_char])
test_x_t, test_label_t = build_vocabulary.build_vocabulary_t(test_char_t, test_label_t, max_sentene_length)
best_pval = np.zeros(2)
best_rval = np.zeros(2)
best_fval = np.zeros(2)
dev_loss_list = []
# Training
# ==================================================
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)

    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # build model
        model = Model(batch_size=FLAGS.batch_size,
                      vocab_size=embed.vocab_size,
                      word_dim=FLAGS.word_dim,
                      lstm_dim=FLAGS.lstm_dim,
                      num_classes=FLAGS.num_classes,
                      lr=FLAGS.lr,
                      clip=FLAGS.clip,
                      l2_reg_lambda=FLAGS.l2_reg_lambda,
                      init_embedding=init_embedding,
                      bi_gram=BI_GRAM,
                      stack=STACK_STATUS,
                      lstm_net=LSTM_NET,
                      bi_direction=BI_DIRECTION,
                      alpha=FLAGS.alpha,
                      alpha_decay=FLAGS.alpha_decay,
                      out_caps_num=FLAGS.out_caps_num,
                      rout_iter=FLAGS.rout_iter,
                      rout_mode=FLAGS.rout_mode,
                      dense_hidden=config.dense_hidden,
                      attn_mode=FLAGS.attn_mode,
                      model_type=FLAGS.model_type)

        # Output directory for models
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "models", FLAGS.model_name))
        print("Writing to {}\n".format(out_dir))
        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        filename_t = 'Base_line_' + 'teacher'
        filename_s = 'Base_line_' + 'student'
        checkpoint_prefix_t = os.path.join(checkpoint_dir, filename_t)
        checkpoint_prefix_s = os.path.join(checkpoint_dir, filename_s)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step_student(type, index_batch, x_batch, y_batch, y_pred_batch, y_class_batch, seq_len_batch):
            step, loss, loss_ref, loss_adv, acc = model.train_step_student(sess, type,  index_batch, x_batch, y_batch, y_pred_batch, y_class_batch, seq_len_batch, FLAGS.dropout_keep_prob)

            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, loss_ref {:g}, loss_adv {:g}, d_acc {:g}".format(time_str, step, loss, loss_ref, loss_adv, acc))

            return step

        def train_step_teacher(x_batch, y_batch, y_class_batch, seq_len_batch):
            step, loss, loss_adv, acc = model.train_step_teacher(sess, x_batch, y_batch, y_class_batch, seq_len_batch, FLAGS.dropout_keep_prob)

            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, loss_adv {:g}, d_acc {:g}".format(time_str, step, loss, loss_adv, acc))

            return step

        def evaluate_word_PRF(y_pred, y, test = False):
            cor_num = 0
            yp_wordnum = y_pred.count(1) + y_pred.count(3) + y_pred.count(5)
            yt_wordnum = y.count(1) + y.count(3) + y.count(5)
            start = 0
            for i in range(len(y)):
                if (int(y[i]) == 1 or int(y[i]) == 3 or int(y[i]) == 5) and int(y[i]) == int(y_pred[i]):
                    flag = True
                    for j in range(i+1, len(y)):
                        if int(y[j]) == 1 or int(y[j]) == 3 or int(y[j]) == 5 or int(y[j]) == 0:
                            if int(y_pred[j]) != 2 and int(y_pred[j]) != 4 and int(y_pred[j]) != 6:
                                flag = True
                            else:
                                flag = False
                            break
                        if int(y[j]) != int(y_pred[j]):
                            flag = False
                            break
                    if flag == True:
                        cor_num += 1
            print('correct: ', cor_num)
            print('pred: ', yp_wordnum)
            print('gold: ', yt_wordnum)
            P = cor_num / (float(yp_wordnum)+ 1e-6)
            R = cor_num / float(yt_wordnum)
            F = 2 * P * R / (P + R + 1e-6)
            print('P: ',P)
            print('R: ',R)
            print('F: ',F)
            if test:
                return P, R, F
            else:
                return F

        def final_test_step_student(x, label, pred, epoch, test=False, bigram=False):
            N = x.shape[0]
            y_true, y_pred = model.fast_all_predict_sdudent(sess, N, x, label, pred, epoch)
            if test:
                print('Test:')
            else:
                print('Dev')
            return y_pred, y_true

        def final_test_step_p_student(x, label, pred, epoch, test=False, bigram=False):
            N = x.shape[0]
            y_true, y_pred = model.fast_all_predict_p_student(sess, N, x, label, pred, epoch, bigram=BI_GRAM)
            if test:
                print('Test:')
            else:
                print('Dev')
            return y_pred, y_true

        def final_test_step_teacher(x, label, epoch, test=False, bigram=False):
            N = x.shape[0]
            y_true, y_pred = model.fast_all_predict_teacher(sess, N, x, label, epoch, bigram=BI_GRAM)
            if test:
                print('Test:')
            else:
                print('Dev')
            return y_pred, y_true

        def final_test_step_p_teacher(x, label, epoch, test=False, bigram=False):
            N = x.shape[0]
            y_true, y_pred = model.fast_all_predict_p_teacher(sess, N, x, label, epoch, bigram=BI_GRAM)
            if test:
                print('Test:')
            else:
                print('Dev')
            return y_pred, y_true

        #train loop
        best_accuracy = [0.0] * 2
        best_step = [0] * 2
        done = False
        #p, r, f = 0.0, 0.0, 0.0
        for i in range(FLAGS.num_epochs):
            if done:
                break
            print('Episode: ', i)
            print('Training teacher...')
            batches_t = data_helpers.batch_iter(list(zip(train_x_t, train_label_t)), FLAGS.batch_size, 1)
            for batch in batches_t:
                x_batch, y_batch = zip(*batch)
                seq_len_batch = [int(len(x)) for x in x_batch]
                y_class_t = [0] * len(seq_len_batch)
                x_batch = np.asarray(x_batch, dtype=np.float32)
                current_step = train_step_teacher(x_batch, y_batch, y_class_t, seq_len_batch)
                if current_step % FLAGS.evaluate_every == 0:
                    yp, yt = final_test_step_teacher(dev_x_t, dev_label_t, epoch=1, bigram=BI_GRAM)
                    print('Teacher:')
                    tmpacc = evaluate_word_PRF(yp,yt)
                    if best_accuracy[0] < tmpacc:
                        best_accuracy[0] = tmpacc
                        best_step[0] = current_step
                        yp_test, yt_test = final_test_step_p_teacher(test_x_t, test_label_t, epoch=1, test=True, bigram=BI_GRAM)
                        best_pval[0], best_rval[0], best_fval[0] = evaluate_word_PRF(yp_test, yt_test, test=True)
                        path = saver.save(sess, checkpoint_prefix_t)
                        print('Saved model checkpoint to {}\n'.format(path))

                    print('Training student...')
                    train_pred = get_pred_p.load_datas(config.TRAIN_DATA_UNI_PRED)
                    train_x_s, train_label_s, train_pred = build_vocabulary.build_vocabulary(train_char_s, train_label_s, train_pred, max_sentene_length)
                    index = []
                    dex = 0
                    ind = np.zeros(config.MAX_LEN)
                    for i in range(len(train_x_s)):
                        index.append(np.add(ind, dex))
                        dex += 1
                    index = np.array(index)
                    batches_s = data_helpers.batch_iter(list(zip(index, train_x_s, train_label_s, train_pred)), FLAGS.batch_size, 1)
                    for batch in batches_s:
                        index_batch, x_batch, y_batch, pred_batch = zip(*batch)
                        pred_batch = np.reshape(pred_batch, [FLAGS.batch_size, -1, 7])
                        seq_len_batch = [int(len(x)) for x in x_batch]
                        y_class_s = [1] * len(seq_len_batch)
                        x_batch = np.asarray(x_batch, dtype=np.float32)
                        current_step = train_step_student(FLAGS.model_type, index_batch, x_batch, y_batch, pred_batch, y_class_s, seq_len_batch)
                        if current_step % FLAGS.evaluate_every == 0:
                            yp, yt = final_test_step_student(dev_x_s, dev_label_s, dev_pred, epoch=1, bigram=BI_GRAM)
                            print('Student:')
                            tmpacc = evaluate_word_PRF(yp, yt)
                            if best_accuracy[1] < tmpacc:
                                best_accuracy[1] = tmpacc
                                best_step[1] = current_step
                                yp_test, yt_test = final_test_step_student(test_x_s, test_label_s, test_pred, epoch=1, test=True, bigram=BI_GRAM)
                                best_pval[1], best_rval[1], best_fval[1] = evaluate_word_PRF(yp_test, yt_test, test=True)
                                path = saver.save(sess, checkpoint_prefix_s)
                                print('Saved model checkpoint to {}\n'.format(path))

                            if current_step - best_step[1] > 2000:
                                print("Dev acc is not getting better in 2000 steps, triggers normal early stop")
                                done = True
                                break

        module_file = tf.train.latest_checkpoint(checkpoint_dir)
        saver.restore(sess, module_file)
        yp, yt = final_test_step_p_student(test_x_s, test_label_s, test_pred, epoch=1, test=True, bigram=BI_GRAM)
        evaluate_word_PRF(yp, yt)

        f = open(TEST_PRED_TAG, 'w')
        for i in range(len(yp)):
            f.write(str(yp[i]))
            f.write('\n')
        f.close()

        print('best P:', best_pval[1])
        print('best R:', best_rval[1])
        print('best F:', best_fval[1])
