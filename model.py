# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  model.py
@Description    :  
@CreateTime     :  2020/2/7 16:27
------------------------------------
@ModifyTime     :  
"""
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
import tensorflow.contrib.rnn as rnn
import data_utils


class Model(object):
    def __init__(self, config):
        self.config = config
        self.lr = config['lr']
        self.word_dim = config['word_dim']
        self.lstm_dim = config['lstm_dim']
        self.seg_dim = config['seg_dim']
        self.num_tags = config['num_tags']
        self.num_words = config['num_words']
        self.num_sges = 4

        self.global_step = tf.Variable(0, trainable=False)
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)
        self.best_test_f1 = tf.Variable(0.0, trainable=False)
        self.initializer = initializers.xavier_initializer()

        # 申请占位符
        # one-hot 编码
        self.word_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="wordInputs")
        self.seg_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="SegInputs")
        self.targets = tf.placeholder(dtype=tf.int32, shape=[None, None], name="Targets")
        self.dropout = tf.placeholder(dtype=tf.float32, name="Dropout")

        length = tf.reduce_sum(tf.sign(tf.abs(self.word_inputs)), reduction_indices=1)
        self.lengths = tf.cast(length, tf.int32)
        self.batch_size = tf.shape(self.word_inputs)[0]
        self.num_setps = tf.shape(self.word_inputs)[-1]

        # embedding层单词和分词信息
        embedding = self.embedding_layer(self.word_inputs, self.seg_inputs, config)

        # lstm输入层
        lstm_inputs = tf.nn.dropout(embedding, self.dropout)

        # lstm输出层
        lstm_outputs = self.biLSTM_layer(lstm_inputs, self.lstm_dim, self.lengths)

        # 投影层
        self.logits = self.project_layer(lstm_outputs)

        # 损失
        self.loss = self.crf_loss_layer(self.logits, self.lengths)

        with tf.variable_scope('optimizer'):
            optimizer = self.config['optimizer']
            if optimizer == "sgd":
                self.opt = tf.train.GradientDescentOptimizer(self.lr)
            elif optimizer == "adam":
                self.opt = tf.train.AdamOptimizer(self.lr)
            elif optimizer == "adgrad":
                self.opt = tf.train.AdagradDAOptimizer(self.lr)
            else:
                raise Exception("优化器错误")

            # compute_gradients:
            # return:A list of (gradient, variable) pairs. Variable is always present, but gradient can be None.
            grad_vars = self.opt.compute_gradients(self.loss)
            capped_grad_vars = [[tf.clip_by_value(g, -self.config['clip'], self.config['clip']), v] for g, v in
                                grad_vars]
            self.train_op = self.opt.apply_gradients(capped_grad_vars, self.global_step)

            # 保存模型
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def embedding_layer(self, word_inputs, seg_inputs, config, name=None):
        """
        :param word_inputs: one-hot编码
        :param seg_inputs: 分词特征
        :param config: 配置
        :param name: 层命名
        :return:
        """
        embedding = []
        with tf.variable_scope("word_embedding" if not name else name), tf.device('/cpu:0'):
            self.word_lookup = tf.get_variable(
                name="word_embedding",
                shape=[self.num_words, self.word_dim],
                initializer=self.initializer
            )
            embedding.append(tf.nn.embedding_lookup(self.word_lookup, word_inputs))

            if config['seg_dim']:
                with tf.variable_scope("seg_embedding"), tf.device('/cpu:0'):
                    self.seg_lookup = tf.get_variable(
                        name="seg_embedding",
                        shape=[self.num_sges, self.seg_dim],
                        initializer=self.initializer
                    )
                    embedding.append(tf.nn.embedding_lookup(self.seg_lookup, seg_inputs))
            embed = tf.concat(embedding, axis=-1)
        return embed