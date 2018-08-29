# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     CNN_model
   Description :   构建CNN模型（没有添加L2正则项），包括嵌入层、卷积层、池化层、softmax层
   Author :       Stephen
   date：          2018/8/28
-------------------------------------------------
   Change Activity:
                   2018/8/28:
-------------------------------------------------
"""
__author__ = 'Stephen'

import tensorflow as tf
import numpy as np

class Text_CNN(object):
    """
    @Parameters
    sequence_length: 句子的长度（此处将所有句子填充为相同的长度59，即最大值）;
    num_classes:输出层中的类别数目；
    vocab_size:词汇量大小。（此参数涉及到嵌入层的大小，嵌入层：[vocab_size, embedding_size]）
    embedding_size:嵌入的维度；
    filter_sizes:滤波器的大小；
    num_filters:滤波器的数目。
    """
    def __init__(self, sequence_length, num_classes, vocab_size,
                 embedding_size, filter_sizes, num_filters):
        """定义网络的输入数据"""
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        #定义嵌入层
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),name='W')
            #在W中查找对应的输入数据input_x的词嵌入矩阵，维度为[None, sequence_length, embedding_size]
            self.embedding_matrix = tf.nn.embedding_lookup(W, self.input_x)
            #将矩阵转变为tensorflow的卷积对应的四维张量：[批次，宽度，高度，通道]->[None, sequence_length, embedding_size, 1]
            self.embedding_matrix_expanded = tf.expand_dims(self.embedding_matrix, -1)

        #定义卷积层和最大池化层
        pooled_outpus = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv-maxpool-%s'%filter_size):
                #Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
                conv = tf.nn.conv2d(
                    self.embedding_matrix_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='conv'
                )
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                #Max-Pooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='pool'
                )
                #此时输出维度为[batch_size, 1, 1, num_filters]
                pooled_outpus.append(pooled)

        #将池化层输出联接为一个向量表示
        num_filters_total = num_filters * len(filter_sizes)
        #按照第四个维度相互聚合
        self.h_pool = tf.concat(pooled_outpus, 3)
        #转置后维度为：[-1, num_filters_total]
        self.h_pool_flat = tf.reshape(self.h_pool, shape=[-1, num_filters_total])

        #Dropout层
        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        #输出层
        with tf.name_scope('output'):
            W = tf.Variable(tf.truncated_normal(shape=[num_filters_total, num_classes], stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name='scores')
            #1：表示按行查找；0：表示按列查找；argmax()返回最大值所在下标
            self.predictions =tf.argmax(self.scores, 1, name='predictions')

        #定义损失loss
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

        #计算准确度
        with tf.name_scope('accuracy'):
            correct_prections = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prections, tf.float32), name='accuracy')
