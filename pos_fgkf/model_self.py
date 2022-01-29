# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import data_helpers
from config import TEST_PRED_P
import config as config
import p2pred
import math
import TfUtils
from TfUtils import masked_softmax
from capsule_masked import Capusule

class Model(object):
    def __init__(self, batch_size, vocab_size, word_dim, lstm_dim, num_classes,
               l2_reg_lambda, lr, clip, init_embedding, bi_gram, stack,
               lstm_net, bi_direction, alpha, alpha_decay, out_caps_num, rout_iter, rout_mode, dense_hidden, attn_mode, model_type):
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.word_dim = word_dim
        self.lstm_dim = lstm_dim
        self.num_classes = num_classes
        self.l2_reg_lambda = l2_reg_lambda
        self.lr = lr
        self.clip = clip
        self.bi_gram = bi_gram
        self.stack = stack
        self.lstm_net = lstm_net
        self.bi_direction = bi_direction
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.out_caps_num = out_caps_num
        self.rout_iter = rout_iter
        self.rout_mode = rout_mode
        self.dense_hidden = dense_hidden
        self.attn_mode = attn_mode
        self.model_type = model_type

        if init_embedding is None:
            self.init_embedding = np.zeros([vocab_size, word_dim], dtype=np.float32)
        else:
            self.init_embedding = init_embedding

        #placeholders
        self.x = tf.placeholder(tf.int32, [None, None], name='x')
        self.y = tf.placeholder(tf.int32, [None, None], name='y')
        self.y_pred = tf.placeholder(tf.float32, [None, None, None], name='y_pred')
        self.y_class = tf.placeholder(tf.int32, [None])
        self.seq_len = tf.placeholder(tf.int32, [None], name='seq_len')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.is_train = tf.placeholder(dtype=tf.bool, name='is_train')

        #embedding layer
        with tf.device('/cpu:0'),tf.variable_scope("embedding"):
            self.embedding = tf.Variable(self.init_embedding ,name="embedding")
            self.embedded_chars = tf.nn.embedding_lookup(self.embedding, self.x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        with tf.variable_scope("softmax_student"):
            if self.bi_direction:
                self.W_s = tf.get_variable(
                    shape=[lstm_dim * 2, num_classes],
                    initializer=tf.truncated_normal_initializer(stddev=0.01),
                    name="weights",
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda))
            else:
                self.W_s = tf.get_variable(
                    shape=[lstm_dim, num_classes],
                    initializer=tf.truncated_normal_initializer(stddev=0.01),
                    name="weights",
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda))

            self.b_s = tf.Variable(
                tf.zeros([num_classes],
                         name="bias"))

        with tf.variable_scope("lstm_student"):
            if self.lstm_net is False:
                self.fw_cell_s = tf.nn.rnn_cell.GRUCell(self.lstm_dim)
                self.bw_cell_s = tf.nn.rnn_cell.GRUCell(self.lstm_dim)
            else:
                self.fw_cell_s = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_dim)
                self.bw_cell_s = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_dim)

        with tf.variable_scope("forward_student"):
            seq_len = tf.cast(self.seq_len, tf.int64)
            #x = tf.nn.embedding_lookup(self.embedding, self.x)
            x = self.embedded_chars_expanded
            x = tf.nn.dropout(x, self.dropout_keep_prob)

            size_s = tf.shape(x)[0]

            x = tf.reshape(x, [size_s, -1, word_dim])

            if self.bi_direction:
                (forward_output, backward_output), _ = tf.nn.bidirectional_dynamic_rnn(
                    self.fw_cell_s,
                    self.bw_cell_s,
                    x,
                    dtype=tf.float32,
                    sequence_length=seq_len,
                    scope='layer_1'
                )
                output = tf.concat(axis=2, values=[forward_output, backward_output])
                if self.stack:
                    (forward_output, backward_output), _ = tf.nn.bidirectional_dynamic_rnn(
                        self.fw_cell_s,
                        self.bw_cell_s,
                        output,
                        dtype=tf.float32,
                        sequence_length=seq_len,
                        scope='layer_2'
                    )
                    output = tf.concat(axis=2, values=[forward_output, backward_output])
            else:
                forward_output, _ = tf.nn.dynamic_rnn(
                    self.fw_cell_s,
                    x,
                    dtype=tf.float32,
                    sequence_length=seq_len,
                    scope='layer_1'
                )
                output = forward_output
                if self.stack:
                    forward_output, _ = tf.nn.dynamic_rnn(
                        self.fw_cell_s,
                        output,
                        dtype=tf.float32,
                        sequence_length=seq_len,
                        scope='layer_2'
                    )
                    output =forward_output

            self.output_student = output

            if self.bi_direction:
                output = tf.reshape(output, [-1, 2 * self.lstm_dim])
            else:
                output = tf.reshape(output, [-1, self.lstm_dim])

            self.matricized_unary_sores_s = tf.matmul(output, self.W_s) + self.b_s
            self.predictions_s = tf.argmax(self.matricized_unary_sores_s, 1, name="predictions")
            self.unary_scores_s = tf.reshape(self.matricized_unary_sores_s, [size_s, -1, num_classes])

        with tf.variable_scope("softmax_teacher"):
            if self.bi_direction:
                self.W_t = tf.get_variable(
                    shape=[lstm_dim * 2, num_classes],
                    initializer=tf.truncated_normal_initializer(stddev=0.01),
                    name="weights",
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda))
            else:
                self.W_t = tf.get_variable(
                    shape=[lstm_dim, num_classes],
                    initializer=tf.truncated_normal_initializer(stddev=0.01),
                    name="weights",
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda))

            self.b_t = tf.Variable(
                tf.zeros([num_classes],
                         name="bias"))

        with tf.variable_scope("lstm_teacher"):
            if self.lstm_net is False:
                self.fw_cell_t = tf.nn.rnn_cell.GRUCell(self.lstm_dim)
                self.bw_cell_t = tf.nn.rnn_cell.GRUCell(self.lstm_dim)
            else:
                self.fw_cell_t = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_dim)
                self.bw_cell_t = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_dim)

        with tf.variable_scope("forward_teacher"):
            seq_len = tf.cast(self.seq_len, tf.int64)
            #x = tf.nn.embedding_lookup(self.embedding, self.x)
            x = self.embedded_chars_expanded
            x = tf.nn.dropout(x, self.dropout_keep_prob)

            size_t = tf.shape(x)[0]
            size = x.shape[0]

            x = tf.reshape(x, [size_t, -1, word_dim])

            if self.bi_direction:
                (forward_output, backward_output), _ = tf.nn.bidirectional_dynamic_rnn(
                    self.fw_cell_t,
                    self.bw_cell_t,
                    x,
                    dtype=tf.float32,
                    sequence_length=seq_len,
                    scope='layer_1'
                )
                output = tf.concat(axis=2, values=[forward_output, backward_output])
                if self.stack:
                    (forward_output, backward_output), _ = tf.nn.bidirectional_dynamic_rnn(
                        self.fw_cell_t,
                        self.bw_cell_t,
                        output,
                        dtype=tf.float32,
                        sequence_length=seq_len,
                        scope='layer_2'
                    )
                    output = tf.concat(axis=2, values=[forward_output, backward_output])
            else:
                forward_output, _ = tf.nn.dynamic_rnn(
                    self.fw_cell_t,
                    x,
                    dtype=tf.float32,
                    sequence_length=seq_len,
                    scope='layer_1'
                )
                output = forward_output
                if self.stack:
                    forward_output, _ = tf.nn.dynamic_rnn(
                        self.fw_cell_t,
                        output,
                        dtype=tf.float32,
                        sequence_length=seq_len,
                        scope='layer_2'
                    )
                    output =forward_output

            self.output_teacher = output

            if self.bi_direction:
                output = tf.reshape(output, [-1, 2 * self.lstm_dim])
            else:
                output = tf.reshape(output, [-1, self.lstm_dim])

            self.matricized_unary_sores_t = tf.matmul(output, self.W_t) + self.b_t
            self.predictions_t = tf.argmax(self.matricized_unary_sores_t, 1, name="predictions")
            self.unary_scores_t = tf.reshape(self.matricized_unary_sores_t, [size_t, -1, num_classes])

        def routing_masked(output, seq_len, out_size, out_caps_num, iter=3, dropout=None, is_train=False):
            b_sz = tf.shape(output)[0]
            with tf.variable_scope('routing'):
                attn_ctx = Capusule(out_caps_num, out_size, iter)(output, seq_len)
                attn_ctx = tf.reshape(attn_ctx, shape=[b_sz, out_caps_num*out_size])
                if dropout is not None:
                    attn_ctx = tf.layers.dropout(attn_ctx, rate=dropout, training=is_train)
            return attn_ctx

        def reverse_routing_masked(output, seq_len, out_size, out_caps_num, iter=3, dropout=None, is_train=False):
            b_sz = tf.shape(output)[0]
            with tf.variable_scope('routing'):
                attn_ctx = Capusule(out_caps_num, out_size, iter)(output, seq_len, reverse_routing=True)
                attn_ctx = tf.reshape(attn_ctx, shape=[b_sz, out_caps_num * out_size])
                if dropout is not None:
                    attn_ctx = tf.layers.dropout(attn_ctx, rate=dropout, training=is_train)
            return attn_ctx

        def self_attention(output, seq_len, out_size, activation_fn=tf.tanh, dropout=None, is_train=False):
            with tf.variable_scope('self-attention'):
                context_vector = tf.get_variable(name='context_vector', shape=[out_size], dtype=tf.float32)
                mlp = tf.layers.dense(output, out_size, activation=activation_fn, name='mlp')
                attn = tf.tensordot(mlp, context_vector, axes=[[2], [0]])
                attn_normed1 = masked_softmax(attn, seq_len)

                attn_normed = tf.expand_dims(attn_normed1, axis=-1)
                attn_ctx = tf.matmul(mlp, attn_normed, transpose_a=True)
                attn_ctx = tf.squeeze(attn_ctx, axis=[2])
                if dropout is not None:
                    attn_ctx = tf.layers.dropout(attn_ctx, rate=dropout, training=is_train)
            return attn, attn_normed1, attn_ctx

        def domain_layer(output, seq_len):
            W_classifier = tf.get_variable(shape=[2*lstm_dim, 2], initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(2))), name='W_classifier')
            bias = tf.Variable(tf.zeros([2], name='class_bias'))
            output_avg = TfUtils.reduce_avg(output, seq_len, 1)
            logits = tf.matmul(output_avg, W_classifier) + bias
            return logits

        def Dloss(logits, y_class):
            labels = tf.to_int64(y_class)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='xentropy')
            D_loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
            return D_loss

        def Dense(inputs, dropout=None, is_train=False, activation=None):
            loop_input = inputs
            for i, hid_num in enumerate(self.dense_hidden):
                with tf.variable_scope("dense-layer-%d" % i):
                    loop_input = tf.layers.dense(loop_input, units=hid_num)
                if i < len(self.dense_hidden) - 1:
                    if dropout is not None:
                        loop_input = tf.layers.dropout(loop_input, rate=dropout, training=is_train)
                    loop_input = activation(loop_input)

            logits = loop_input
            return logits

        with tf.variable_scope("student_sent_rep"):
            self.w_char, self.w_multi, self.student_sent_rep = self_attention(self.output_student, seq_len, int(self.output_student.get_shape()[-1]),
                                                   dropout=0.2, is_train=self.is_train)

        with tf.variable_scope("teacher_sent_rep"):
            _, __, self.teacher_sent_rep = self_attention(self.output_teacher, seq_len, int(self.output_teacher.get_shape()[-1]),
                                                   dropout=0.2, is_train=self.is_train)

        with tf.variable_scope("attention"):

            if self.model_type == 'type I':
                self.alpha_new = self.alpha
            elif self.model_type == 'type II':
                # base = 1 / config.MAX_LEN
                # alpha = self.alpha
                # self.alpha_new = np.multiply(tf.exp(np.add(np.multiply(self.attn_normed, -1), base)), alpha)
                # self.alpha_seg = np.add(np.multiply(self.alpha_new, -1), 1) # 1-alpha
                attn = self.w_char
                self.W_alpha = tf.get_variable(
                    shape=[config.MAX_LEN, config.MAX_LEN],
                    initializer=tf.truncated_normal_initializer(stddev=0.01),
                    name="w_alpha",
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda))
                self.b_alpha = tf.Variable(
                    tf.zeros([config.MAX_LEN],
                             name="bias_alpha"))
                self.alpha_new = tf.nn.sigmoid(tf.add(tf.matmul(attn, self.W_alpha), self.b_alpha))
                self.attn_normed = attn
            elif self.model_type == 'type III':
                #self.attn_normed = tf.nn.softmax(attn, dim=1) # [batch_size, seq_len]
                attn = self.w_multi
                self.W_alpha = tf.get_variable(
                    shape=[config.MAX_LEN, config.MAX_LEN],
                    initializer=tf.truncated_normal_initializer(stddev=0.01),
                    name="w_alpha",
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda))
                self.b_alpha = tf.Variable(
                    tf.zeros([config.MAX_LEN],
                             name="bias_alpha"))
                self.alpha_new = tf.nn.sigmoid(tf.add(tf.matmul(attn, self.W_alpha), self.b_alpha))
                self.attn_normed = attn
            self.alpha_seg = np.add(np.multiply(self.alpha_new, -1), 1)


        with tf.variable_scope("loss_seg_student_params"):
            # CRF log likelihood
            unary_scores_0 = self.unary_scores_s[:, :, 0]
            unary_scores_1 = self.unary_scores_s[:, :, 1]
            unary_scores_2 = self.unary_scores_s[:, :, 2]
            unary_scores_3 = self.unary_scores_s[:, :, 3]
            weighted_unary_scores_0 = np.multiply(unary_scores_0, self.alpha_seg)
            weighted_unary_scores_1 = np.multiply(unary_scores_1, self.alpha_seg)
            weighted_unary_scores_2 = np.multiply(unary_scores_2, self.alpha_seg)
            weighted_unary_scores_3 = np.multiply(unary_scores_3, self.alpha_seg)
            weighted_unary_scores_0 = tf.expand_dims(weighted_unary_scores_0, -1)
            weighted_unary_scores_0 = tf.reshape(weighted_unary_scores_0, [size_s, -1, 1])
            weighted_unary_scores_1 = tf.expand_dims(weighted_unary_scores_1, -1)
            weighted_unary_scores_1 = tf.reshape(weighted_unary_scores_1, [size_s, -1, 1])
            weighted_unary_scores_2 = tf.expand_dims(weighted_unary_scores_2, -1)
            weighted_unary_scores_2 = tf.reshape(weighted_unary_scores_2, [size_s, -1, 1])
            weighted_unary_scores_3 = tf.expand_dims(weighted_unary_scores_3, -1)
            weighted_unary_scores_3 = tf.reshape(weighted_unary_scores_3, [size_s, -1, 1])

            unary_scores_4 = self.unary_scores_s[:, :, 4]
            unary_scores_5 = self.unary_scores_s[:, :, 5]
            unary_scores_6 = self.unary_scores_s[:, :, 6]
            unary_scores_7 = self.unary_scores_s[:, :, 7]
            weighted_unary_scores_4 = np.multiply(unary_scores_4, self.alpha_seg)
            weighted_unary_scores_5 = np.multiply(unary_scores_5, self.alpha_seg)
            weighted_unary_scores_6 = np.multiply(unary_scores_6, self.alpha_seg)
            weighted_unary_scores_7 = np.multiply(unary_scores_7, self.alpha_seg)
            weighted_unary_scores_4 = tf.expand_dims(weighted_unary_scores_4, -1)
            weighted_unary_scores_4 = tf.reshape(weighted_unary_scores_4, [size_s, -1, 1])
            weighted_unary_scores_5 = tf.expand_dims(weighted_unary_scores_5, -1)
            weighted_unary_scores_5 = tf.reshape(weighted_unary_scores_5, [size_s, -1, 1])
            weighted_unary_scores_6 = tf.expand_dims(weighted_unary_scores_6, -1)
            weighted_unary_scores_6 = tf.reshape(weighted_unary_scores_6, [size_s, -1, 1])
            weighted_unary_scores_7 = tf.expand_dims(weighted_unary_scores_7, -1)
            weighted_unary_scores_7 = tf.reshape(weighted_unary_scores_7, [size_s, -1, 1])

            unary_scores_8 = self.unary_scores_s[:, :, 8]
            unary_scores_9 = self.unary_scores_s[:, :, 9]
            unary_scores_10 = self.unary_scores_s[:, :, 10]
            unary_scores_11 = self.unary_scores_s[:, :, 11]
            weighted_unary_scores_8 = np.multiply(unary_scores_8, self.alpha_seg)
            weighted_unary_scores_9 = np.multiply(unary_scores_9, self.alpha_seg)
            weighted_unary_scores_10 = np.multiply(unary_scores_10, self.alpha_seg)
            weighted_unary_scores_11 = np.multiply(unary_scores_11, self.alpha_seg)
            weighted_unary_scores_8 = tf.expand_dims(weighted_unary_scores_8, -1)
            weighted_unary_scores_8 = tf.reshape(weighted_unary_scores_8, [size_s, -1, 1])
            weighted_unary_scores_9 = tf.expand_dims(weighted_unary_scores_9, -1)
            weighted_unary_scores_9 = tf.reshape(weighted_unary_scores_9, [size_s, -1, 1])
            weighted_unary_scores_10 = tf.expand_dims(weighted_unary_scores_10, -1)
            weighted_unary_scores_10 = tf.reshape(weighted_unary_scores_10, [size_s, -1, 1])
            weighted_unary_scores_11 = tf.expand_dims(weighted_unary_scores_11, -1)
            weighted_unary_scores_11 = tf.reshape(weighted_unary_scores_11, [size_s, -1, 1])

            unary_scores_12 = self.unary_scores_s[:, :, 12]
            unary_scores_13 = self.unary_scores_s[:, :, 13]
            unary_scores_14 = self.unary_scores_s[:, :, 14]
            unary_scores_15 = self.unary_scores_s[:, :, 15]
            weighted_unary_scores_12 = np.multiply(unary_scores_12, self.alpha_seg)
            weighted_unary_scores_13 = np.multiply(unary_scores_13, self.alpha_seg)
            weighted_unary_scores_14 = np.multiply(unary_scores_14, self.alpha_seg)
            weighted_unary_scores_15 = np.multiply(unary_scores_15, self.alpha_seg)
            weighted_unary_scores_12 = tf.expand_dims(weighted_unary_scores_12, -1)
            weighted_unary_scores_12 = tf.reshape(weighted_unary_scores_12, [size_s, -1, 1])
            weighted_unary_scores_13 = tf.expand_dims(weighted_unary_scores_13, -1)
            weighted_unary_scores_13 = tf.reshape(weighted_unary_scores_13, [size_s, -1, 1])
            weighted_unary_scores_14 = tf.expand_dims(weighted_unary_scores_14, -1)
            weighted_unary_scores_14 = tf.reshape(weighted_unary_scores_14, [size_s, -1, 1])
            weighted_unary_scores_15 = tf.expand_dims(weighted_unary_scores_15, -1)
            weighted_unary_scores_15 = tf.reshape(weighted_unary_scores_15, [size_s, -1, 1])

            unary_scores_16 = self.unary_scores_s[:, :, 16]
            unary_scores_17 = self.unary_scores_s[:, :, 17]
            weighted_unary_scores_16 = np.multiply(unary_scores_16, self.alpha_seg)
            weighted_unary_scores_17 = np.multiply(unary_scores_17, self.alpha_seg)
            weighted_unary_scores_16 = tf.expand_dims(weighted_unary_scores_16, -1)
            weighted_unary_scores_16 = tf.reshape(weighted_unary_scores_16, [size_s, -1, 1])
            weighted_unary_scores_17 = tf.expand_dims(weighted_unary_scores_17, -1)
            weighted_unary_scores_17 = tf.reshape(weighted_unary_scores_17, [size_s, -1, 1])

            self.weighted_unary_scores = tf.concat(axis=2, values=[weighted_unary_scores_0, weighted_unary_scores_1,
                                                                   weighted_unary_scores_2, weighted_unary_scores_3,
                                                                   weighted_unary_scores_4, weighted_unary_scores_5,
                                                                   weighted_unary_scores_6, weighted_unary_scores_7,
                                                                   weighted_unary_scores_8, weighted_unary_scores_9,
                                                                   weighted_unary_scores_10, weighted_unary_scores_11,
                                                                   weighted_unary_scores_12, weighted_unary_scores_13,
                                                                   weighted_unary_scores_14, weighted_unary_scores_15,
                                                                   weighted_unary_scores_16, weighted_unary_scores_17])

            log_likelihood_seg_s, self.transition_params_s = tf.contrib.crf.crf_log_likelihood(self.weighted_unary_scores,
                                                                                               self.y, self.seq_len)
            # logits_s = domain_layer(self.output_student, seq_len)
            logits_s = Dense(self.student_sent_rep, dropout=0.2, is_train=self.is_train, activation=tf.nn.tanh)
            self.logits_s =logits_s
            self.D_loss_student = Dloss(logits_s, self.y_class)
            self.prediction_s = tf.argmax(logits_s, axis=-1, output_type=self.y_class.dtype)
            self.accuracy_s = tf.reduce_mean(tf.cast(tf.equal(self.prediction_s, self.y_class), tf.float32))

        with tf.variable_scope("loss_seg_student"):
            self.loss_seg_student = tf.reduce_mean(-log_likelihood_seg_s)

        with tf.variable_scope("loss_ref"):
            pred_0 = self.y_pred[:, :, 0]
            pred_1 = self.y_pred[:, :, 1]
            pred_2 = self.y_pred[:, :, 2]
            pred_3 = self.y_pred[:, :, 3]
            weighted_pred_0 = np.multiply(pred_0, self.alpha_new)
            weighted_pred_1 = np.multiply(pred_1, self.alpha_new)
            weighted_pred_2 = np.multiply(pred_2, self.alpha_new)
            weighted_pred_3 = np.multiply(pred_3, self.alpha_new)
            weighted_pred_0 = tf.expand_dims(weighted_pred_0, -1)
            weighted_pred_0 = tf.reshape(weighted_pred_0, [size_s, -1, 1])
            weighted_pred_1 = tf.expand_dims(weighted_pred_1, -1)
            weighted_pred_1 = tf.reshape(weighted_pred_1, [size_s, -1, 1])
            weighted_pred_2 = tf.expand_dims(weighted_pred_2, -1)
            weighted_pred_2 = tf.reshape(weighted_pred_2, [size_s, -1, 1])
            weighted_pred_3 = tf.expand_dims(weighted_pred_3, -1)
            weighted_pred_3 = tf.reshape(weighted_pred_3, [size_s, -1, 1])

            pred_4 = self.y_pred[:, :, 4]
            pred_5 = self.y_pred[:, :, 5]
            pred_6 = self.y_pred[:, :, 6]
            pred_7 = self.y_pred[:, :, 7]
            weighted_pred_4 = np.multiply(pred_4, self.alpha_new)
            weighted_pred_5 = np.multiply(pred_5, self.alpha_new)
            weighted_pred_6 = np.multiply(pred_6, self.alpha_new)
            weighted_pred_7 = np.multiply(pred_7, self.alpha_new)
            weighted_pred_4 = tf.expand_dims(weighted_pred_4, -1)
            weighted_pred_4 = tf.reshape(weighted_pred_4, [size_s, -1, 1])
            weighted_pred_5 = tf.expand_dims(weighted_pred_5, -1)
            weighted_pred_5 = tf.reshape(weighted_pred_5, [size_s, -1, 1])
            weighted_pred_6 = tf.expand_dims(weighted_pred_6, -1)
            weighted_pred_6 = tf.reshape(weighted_pred_6, [size_s, -1, 1])
            weighted_pred_7 = tf.expand_dims(weighted_pred_7, -1)
            weighted_pred_7 = tf.reshape(weighted_pred_7, [size_s, -1, 1])

            pred_8 = self.y_pred[:, :, 8]
            pred_9 = self.y_pred[:, :, 9]
            pred_10 = self.y_pred[:, :, 10]
            pred_11 = self.y_pred[:, :, 11]
            weighted_pred_8 = np.multiply(pred_8, self.alpha_new)
            weighted_pred_9 = np.multiply(pred_9, self.alpha_new)
            weighted_pred_10 = np.multiply(pred_10, self.alpha_new)
            weighted_pred_11 = np.multiply(pred_11, self.alpha_new)
            weighted_pred_8 = tf.expand_dims(weighted_pred_8, -1)
            weighted_pred_8 = tf.reshape(weighted_pred_8, [size_s, -1, 1])
            weighted_pred_9 = tf.expand_dims(weighted_pred_9, -1)
            weighted_pred_9 = tf.reshape(weighted_pred_9, [size_s, -1, 1])
            weighted_pred_10 = tf.expand_dims(weighted_pred_10, -1)
            weighted_pred_10 = tf.reshape(weighted_pred_10, [size_s, -1, 1])
            weighted_pred_11 = tf.expand_dims(weighted_pred_11, -1)
            weighted_pred_11 = tf.reshape(weighted_pred_11, [size_s, -1, 1])

            pred_12 = self.y_pred[:, :, 12]
            pred_13 = self.y_pred[:, :, 13]
            pred_14 = self.y_pred[:, :, 14]
            pred_15 = self.y_pred[:, :, 15]
            weighted_pred_12 = np.multiply(pred_12, self.alpha_new)
            weighted_pred_13 = np.multiply(pred_13, self.alpha_new)
            weighted_pred_14 = np.multiply(pred_14, self.alpha_new)
            weighted_pred_15 = np.multiply(pred_15, self.alpha_new)
            weighted_pred_12 = tf.expand_dims(weighted_pred_12, -1)
            weighted_pred_12 = tf.reshape(weighted_pred_12, [size_s, -1, 1])
            weighted_pred_13 = tf.expand_dims(weighted_pred_13, -1)
            weighted_pred_13 = tf.reshape(weighted_pred_13, [size_s, -1, 1])
            weighted_pred_14 = tf.expand_dims(weighted_pred_14, -1)
            weighted_pred_14 = tf.reshape(weighted_pred_14, [size_s, -1, 1])
            weighted_pred_15 = tf.expand_dims(weighted_pred_15, -1)
            weighted_pred_15 = tf.reshape(weighted_pred_15, [size_s, -1, 1])

            pred_16 = self.y_pred[:, :, 16]
            pred_17 = self.y_pred[:, :, 17]
            weighted_pred_16 = np.multiply(pred_16, self.alpha_new)
            weighted_pred_17 = np.multiply(pred_17, self.alpha_new)
            weighted_pred_16 = tf.expand_dims(weighted_pred_16, -1)
            weighted_pred_16 = tf.reshape(weighted_pred_16, [size_s, -1, 1])
            weighted_pred_17 = tf.expand_dims(weighted_pred_17, -1)
            weighted_pred_17 = tf.reshape(weighted_pred_17, [size_s, -1, 1])

            self.weighted_pred = tf.concat(axis=2, values=[weighted_pred_0, weighted_pred_1,
                                                           weighted_pred_2, weighted_pred_3,
                                                           weighted_pred_4, weighted_pred_5,
                                                           weighted_pred_6, weighted_pred_7,
                                                           weighted_pred_8, weighted_pred_9,
                                                           weighted_pred_10, weighted_pred_11,
                                                           weighted_pred_12, weighted_pred_13,
                                                           weighted_pred_14, weighted_pred_15,
                                                           weighted_pred_16, weighted_pred_17])

            if self.model_type == 'type II':
                self.sent_p = tf.nn.softmax(self.logits_s, dim=1)
                log_soft = tf.nn.log_softmax(self.unary_scores_s)

                H_mid = -tf.reduce_mean(tf.multiply(self.weighted_pred, log_soft), axis=0)
                self.loss_ref = tf.reduce_sum(H_mid)
            elif self.model_type == 'type I':
                self.sent_p = tf.nn.softmax(self.logits_s, dim=1)
                sent_p_true = self.sent_p[:, 1]
                self.gammma = tf.get_variable(initializer=1.0, name='gamma')
                self.tao = tf.get_variable(initializer=0.5, name='tao')

                self.beta = tf.exp(np.multiply(np.add(self.tao, tf.multiply(sent_p_true, -1)), self.gammma))
                w_pred_p = []
                for j in range(batch_size):
                    w_pred_p.append(np.multiply(self.beta[j], self.y_pred[j]))

                log_soft = tf.nn.log_softmax(tf.clip_by_value(self.unary_scores_s, 1e-10, 1.0))
                H_mid = -tf.reduce_mean(tf.multiply(w_pred_p, log_soft), axis=0)
                self.loss_ref = tf.reduce_sum(H_mid)
                # self.loss_ref = tf.reduce_mean(tf.square(self.y_pred - self.unary_scores_s))
            elif self.model_type == 'type III':
                self.sent_p = tf.nn.softmax(self.logits_s, dim=1)
                sent_p_true = self.sent_p[:, 1]
                self.gammma = tf.get_variable(initializer=1.0, name='gamma')
                self.tao = tf.get_variable(initializer=0.5, name='tao')

                self.beta = tf.exp(np.multiply(np.add(self.tao, tf.multiply(sent_p_true, -1)), self.gammma))

                w_pred_p = []
                for j in range(batch_size):
                    w_pred_p.append(np.multiply(self.beta[j], self.weighted_pred[j]))

                log_soft = tf.nn.log_softmax(tf.clip_by_value(self.unary_scores_s, 1e-10, 1.0))
                H_mid = -tf.reduce_mean(tf.multiply(w_pred_p, log_soft), axis=0)
                self.loss_ref = tf.reduce_sum(H_mid)

        with tf.variable_scope("loss_student"):
            #alpha = self.alpha
            self.loss_student = self.loss_seg_student + self.loss_ref

        with tf.variable_scope("loss_teacher"):
            log_likelihood_seg_t, self.transition_params_t = tf.contrib.crf.crf_log_likelihood(self.unary_scores_t, self.y, self.seq_len)

            self.loss_teacher = tf.reduce_mean(-log_likelihood_seg_t)
            #logits_t = domain_layer(self.output_teacher, seq_len)
            logits_t = Dense(self.teacher_sent_rep, dropout=0.2, is_train=self.is_train, activation=tf.nn.tanh)
            self.D_loss_teacher = Dloss(logits_t, self.y_class)
            self.prediction_t = tf.argmax(logits_t, axis=-1, output_type=self.y_class.dtype)
            self.accuracy_t = tf.reduce_mean(tf.cast(tf.equal(self.prediction_t, self.y_class), tf.float32))

        def trainingDomain(loss):
            optimizer = tf.train.AdamOptimizer(self.lr)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            #tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='domain')
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), self.clip)
            train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

            return train_op, global_step

        with tf.variable_scope("train_student_ops"):
            self.optimizer_s = tf.train.AdamOptimizer(self.lr)

            self.global_step_s = tf.Variable(0, name="global_step", trainable=False)#全局步骤计数

            tvars = tf.trainable_variables()
            #tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='embedding') + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='lstm_student') + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='forward_student') + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='softmax_student') + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='loss_seg_student_params')
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss_student, tvars), self.clip)#对梯度进行截取
            self.train_op_s = self.optimizer_s.apply_gradients(zip(grads, tvars), global_step=self.global_step_s)#将算出的梯度应用到变量上
            self.domain_op_s, self.global_step_domian_s = trainingDomain(self.D_loss_student)

        with tf.variable_scope("train_teacher_ops"):
            self.optimizer_t = tf.train.AdamOptimizer(self.lr)

            self.global_step_t = tf.Variable(0, name="global_step", trainable=False)#全局步骤计数

            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss_teacher, tvars), self.clip)#对梯度进行截取
            self.train_op_t = self.optimizer_t.apply_gradients(zip(grads, tvars), global_step=self.global_step_t)#将算出的梯度应用到变量上
            self.domain_op_t, self.global_step_domain_t = trainingDomain(self.D_loss_teacher)

    def train_step_student(self, sess, type, index_batch, x_batch, y_batch, y_pred_batch, y_class_batch, seq_len_batch, dropout_keep_prob, is_train=True):

        feed_dict = {
            self.x: x_batch,
            self.y: y_batch,
            self.y_pred: y_pred_batch,
            self.y_class: y_class_batch,
            self.seq_len: seq_len_batch,
            self.dropout_keep_prob: dropout_keep_prob,
            self.is_train: is_train
        }
        if type == 'type II' or type == 'type III':
            loss_ref, attn, alphas, _, step, loss, __, step_adv, loss_adv, acc = sess.run([self.loss_ref, self.attn_normed, self.alpha_new, self.train_op_s, self.global_step_s, self.loss_student, self.domain_op_s, self.global_step_domian_s, self.D_loss_student, self.accuracy_s], feed_dict)

            f = open(config.ATTENTION_FILE, 'w')
            for index, attention, alpha in zip(index_batch, attn, alphas):
                f.write(str(index[0]))
                f.write('\n')
                for i in range(seq_len_batch[0]):
                    f.write(str(attention[i]))
                    f.write(';')
                    f.write(str(alpha[i]))
                    f.write('\n')
        elif type == 'type I':
            loss_ref, _, step, loss, __, step_adv, loss_adv, acc = sess.run(
                [self.loss_ref, self.train_op_s, self.global_step_s, self.loss_student,
                 self.domain_op_s, self.global_step_domian_s, self.D_loss_student, self.accuracy_s], feed_dict)

        return step, loss_ref, loss, loss_adv, acc

    def train_step_student_seg(self, sess, x_batch, y_batch, y_pred_batch, seq_len_batch, dropout_keep_prob, is_train=True):

        feed_dict = {
            self.x: x_batch,
            self.y: y_batch,
            self.y_pred: y_pred_batch,
            self.seq_len: seq_len_batch,
            self.dropout_keep_prob: dropout_keep_prob,
            self.is_train: is_train
        }
        _, step, loss,  = sess.run([self.train_op_s, self.global_step_s, self.loss_student], feed_dict)

        return step, loss

    def train_step_student_d(self, sess, x_batch, y_class_batch, seq_len_batch, dropout_keep_prob, is_train=True):

        feed_dict = {
            self.x: x_batch,
            self.y_class: y_class_batch,
            self.seq_len: seq_len_batch,
            self.dropout_keep_prob: dropout_keep_prob,
            self.is_train: is_train
        }
        _, step_adv, loss_adv, acc = sess.run([self.domain_op_s, self.global_step_domian_s, self.D_loss_student, self.accuracy_s], feed_dict)

        return step_adv, loss_adv, acc

    def train_step_teacher(self, sess, x_batch, y_batch, y_class_batch, seq_len_batch, dropout_keep_prob, is_train=True):

        feed_dict = {
            self.x: x_batch,
            self.y: y_batch,
            self.y_class: y_class_batch,
            self.seq_len: seq_len_batch,
            self.dropout_keep_prob: dropout_keep_prob,
            self.is_train: is_train
        }
        _, step, loss, __, step_adv, loss_adv, acc = sess.run([self.train_op_t, self.global_step_t, self.loss_teacher, self.domain_op_t, self.global_step_domain_t, self.D_loss_teacher, self.accuracy_t], feed_dict)

        return step, loss, loss_adv, acc

    def train_step_teacher_seg(self, sess, x_batch, y_batch, seq_len_batch, dropout_keep_prob, is_train=True):

        feed_dict = {
            self.x: x_batch,
            self.y: y_batch,
            self.seq_len: seq_len_batch,
            self.dropout_keep_prob: dropout_keep_prob,
            self.is_train: is_train
        }
        _, step, loss = sess.run([self.train_op_t, self.global_step_t, self.loss_teacher], feed_dict)

        return step, loss

    def train_step_teacher_d(self, sess, x_batch, y_class_batch, seq_len_batch, dropout_keep_prob, is_train=True):

        feed_dict = {
            self.x: x_batch,
            self.y_class: y_class_batch,
            self.seq_len: seq_len_batch,
            self.dropout_keep_prob: dropout_keep_prob,
            self.is_train: is_train
        }
        _, step_adv, loss_adv, acc = sess.run([self.domain_op_t, self.global_step_domain_t, self.D_loss_teacher, self.accuracy_t], feed_dict)

        return step_adv, loss_adv, acc

    def fast_all_predict_sdudent(self, sess, N, x, label, pred, epoch, is_train=False):
        y_pred, y_true = [], []
        batches = data_helpers.batch_iter(list(zip(x, label, pred)), len(x), epoch, shuffle=False)
        for batch in batches:
            x_batch, y_batch, y_pred_batch = zip(*batch)
            seq_len_batch = [int(len(x)) for x in x_batch]
            y_pred_batch = np.reshape(y_pred_batch, [-1, 60, self.num_classes])
            #infer predictions
            feed_dict = {
                self.x: x_batch,
                self.y: y_batch,
                self.y_pred: y_pred_batch,
                self.seq_len: seq_len_batch,
                self.dropout_keep_prob: 1.0,
                self.is_train: is_train
            }

            unary_scores, transition_params = sess.run([self.unary_scores_s, self.transition_params_s], feed_dict)


            for unary_scores_, y_, seq_len_ in zip(unary_scores, y_batch, seq_len_batch):
                #remove padding
                unary_scores_ = unary_scores_[:seq_len_]

                #compute the highest scoring sequence
                viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(unary_scores_, transition_params)

                y_pred += viterbi_sequence
                y_true += y_[:seq_len_].tolist()

        return y_true, y_pred


    def fast_all_predict_p_student(self, sess, N, x, label, pred, epoch, bigram, is_train=False):
        y_pred, y_true = [], []
        #num_batches = int((N - 5) / self.batch_size)
        #batches = data_helpers.batch_iter(
            #list(zip(x, label, pred)), self.batch_size, epoch)
        batches = data_helpers.batch_iter(list(zip(x, label, pred)), len(x), epoch, shuffle=False)
        f = open(TEST_PRED_P, 'w')
        for batch in batches:
            x_batch, y_batch, y_pred_batch = zip(*batch)
            seq_len_batch = [int(len(x)) for x in x_batch]
            y_pred_batch = np.reshape(y_pred_batch, [-1, 60, self.num_classes])
            #infer predictions
            feed_dict = {
                self.x: x_batch,
                self.y: y_batch,
                self.y_pred: y_pred_batch,
                self.seq_len: seq_len_batch,
                self.dropout_keep_prob: 1.0,
                self.is_train: is_train
            }

            unary_scores, transition_params = sess.run([self.unary_scores_s, self.transition_params_s], feed_dict)

            for unary_scores_, y_, seq_len_ in zip(unary_scores, y_batch, seq_len_batch):
                #remove padding
                unary_scores_ = unary_scores_[:seq_len_]

                #compute the highest scoring sequence
                viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(unary_scores_, transition_params)

                idx = 0
                for scores in unary_scores_:
                    p = np.exp(scores) / np.sum(np.exp(scores), axis=0)
                    #print('scores:', scores)
                    #print('p', p)
                    if idx < seq_len_:
                        for i in range(self.num_classes):
                            f.write(str(p[i]))
                            f.write(';')
                        f.write('\n')
                        idx += 1
                    else:
                        break
                f.write('\n')

                y_pred += viterbi_sequence
                y_true += y_[:seq_len_].tolist()
        f.close()
        return y_true, y_pred


    def fast_all_predict_teacher(self, sess, N, x, label, epoch, bigram, is_train=False):
        y_pred, y_true = [], []
        batches = data_helpers.batch_iter(list(zip(x, label)), int(len(x)/4), epoch, shuffle=False)
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            seq_len_batch = [int(len(x)) for x in x_batch]
            #infer predictions
            feed_dict = {
                self.x: x_batch,
                self.y: y_batch,
                self.seq_len: seq_len_batch,
                self.dropout_keep_prob: 1.0,
                self.is_train: is_train
            }

            unary_scores, transition_params = sess.run([self.unary_scores_t, self.transition_params_t], feed_dict)
            for unary_scores_, y_, seq_len_ in zip(unary_scores, y_batch, seq_len_batch):
                #remove padding
                unary_scores_ = unary_scores_[:seq_len_]
                #compute the highest scoring sequence
                viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(unary_scores_, transition_params)

                y_pred += viterbi_sequence
                y_true += y_[:seq_len_].tolist()

        return y_true, y_pred

    def fast_all_predict_p_teacher(self, sess, N, x, label, epoch, bigram, is_train=False):
        y_pred, y_true = [], []
        batches = data_helpers.batch_iter(list(zip(x, label)), int(len(x)/4), epoch, shuffle=False)
        f = open(config.T_TEST_PRED_P, 'w')
        f_1 = open(config.T_TEST_SENT_P, 'w', encoding='utf')
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            #print(y_batch)
            seq_len_batch = [int(len(x)) for x in x_batch]
            #infer predictions
            feed_dict = {
                self.x: x_batch,
                self.y: y_batch,
                self.seq_len: seq_len_batch,
                self.dropout_keep_prob: 1.0,
                self.is_train: is_train
            }

            logits, unary_scores, transition_params = sess.run([self.sent_p, self.unary_scores_t, self.transition_params_t], feed_dict)

            for logit, unary_scores_, y_, seq_len_ in zip(logits, unary_scores, y_batch, seq_len_batch):
                #remove padding
                unary_scores_ = unary_scores_[:seq_len_]
                #compute the highest scoring sequence
                viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(unary_scores_, transition_params)

                f_1.write(str(logit[0]))
                f_1.write(';')
                f_1.write(str(logit[1]))
                f_1.write('\n')

                idx = 0
                for scores in unary_scores_:
                    p = np.exp(scores) / np.sum(np.exp(scores), axis=0)
                    if idx < seq_len_:
                        for i in range(self.num_classes):
                            f.write(str(p[i]))
                            f.write(';')
                        f.write('\n')
                        idx += 1
                    else:
                        break
                f.write('\n')

                y_pred += viterbi_sequence
                y_true += y_[:seq_len_].tolist()
        f.close()
        f_1.close()
        in_f = open(config.T_TEST_DATA_UNI, 'r')
        in_p = open(config.T_TEST_PRED_P, 'r')
        out_f = open(config.TRAIN_DATA_UNI_PRED, 'w')
        p2pred.prob2pred(in_f, in_p, out_f)
        return y_true, y_pred
