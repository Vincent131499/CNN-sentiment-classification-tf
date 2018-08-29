# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     process_data
   Description :   数据预处理
   Author :       Stephen
   date：          2018/8/27
-------------------------------------------------
   Change Activity:
                   2018/8/27:
-------------------------------------------------
"""
__author__ = 'Stephen'

import numpy as np
import re
from tensorflow.contrib import learn
import tensorflow as tf
from sklearn.cross_validation import train_test_split

#读取数据参数设置
# tf.flags.DEFINE_float('dev_sample_percentage', .1, 'Percentage of the training data to use for validation')
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string('positive_data_file', './data/rt-polaritydata/rt-polarity.pos', 'Data source for the positive data')
tf.flags.DEFINE_string('negative_data_file', './data/rt-polaritydata/rt-polarity.neg', 'Data source for the negative data')

# FLAGS = tf.flags.FLAGS
FLAGS = tf.flags.FLAGS

"""去除噪音函数"""
def clean_text(line):
    # print('过滤前--------------->', line)
    #替换掉无意义的单个字符
    line = re.sub(r'[^A-Za-z0-9(),!?.\'\`]', ' ', line)
    """使用空格将单词后缀单独分离开来"""
    line = re.sub(r'\'s', ' \'s ', line)
    line = re.sub(r'\'ve', ' \'ve ', line)
    line = re.sub(r'n\'t', ' n\'t ', line)
    line = re.sub(r'\'re', ' \'re ', line)
    line = re.sub(r'\'d', ' \'d ',line)
    line = re.sub(r'\'ll', ' \'ll ',line)
    """使用空格将标点符号、括号等字符单独分离开来"""
    line = re.sub(r',', ' , ', line)
    line = re.sub(r'!', ' ! ', line)
    line = re.sub(r'\?', ' \? ', line)
    line = re.sub(r'\(', ' ( ', line)
    line = re.sub(r'\)', ' ) ', line)
    line = re.sub(r'\s{2,}', ' ', line)
    # line = re.sub(r'')
    # line = re.sub(r',', ' , ', line)
    # print('过滤后--------------->',line)
    return line.strip().lower()

"""从文件中读取数据和标签"""
def load_data_and_label(pos_filename, neg_filename):
    """读取积极类别的数据"""
    positive_texts = open(pos_filename, 'r', encoding='utf-8').readlines()
    # print(positive_texts)
    # positive_texts = open(positive_filename, 'rb').readlines()
    positive_texts = [line.strip() for line in positive_texts]
    print('积极句子数目：', len(positive_texts))
    # print(len(positive_texts))
    """读取消极类别的数据"""
    negative_texts = open(neg_filename, 'r', encoding='utf-8').readlines()
    # negative_texts = open(positive_filename, 'rb').readlines()
    negative_texts = [line.strip() for line in negative_texts]
    print('消极句子数目：', len(negative_texts))

    """去除噪音（英文）"""
    x_text = positive_texts + negative_texts
    print('全部句子数目：', len(x_text))
    x_text = [clean_text(text) for text in x_text]
    print('文本清洗完毕！')

    """生成标签"""
    positive_labels = [[0, 1] for _ in negative_texts]
    negative_labels = [[1, 0] for _ in negative_texts]
    y = np.concatenate([positive_labels, negative_labels], 0)
    print('标签数目：', len(y))
    # print(y)
    # for mat in y:
    #     print(mat)
    return [x_text, y]

def construct_dataset():
    print('加载数据......')
    # positive_filename = './data/rt-polaritydata/rt-polarity.pos'
    # negative_filename = './data/rt-polaritydata/rt-polarity.neg'
    # positive_filename = './data/rt-polarity.pos'
    # negative_filename = './data/rt-polarity.neg'
    x_text, y = load_data_and_label(FLAGS.positive_data_file, FLAGS.negative_data_file)

    """建立词汇表"""
    max_sentence_length = max([len(text.split(' ')) for text in x_text])
    print('最长句子长度：', max_sentence_length)
    #tf.contrib.learn.preprocessing.VocabularyProcessor:生成词汇表，每一个文档/句子的长度<=max_sentnce_length,记录的是单词的位置信息
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_sentence_length)
    #x:每一个句子中的单词对应词汇表的位置
    x = np.array(list(vocab_processor.fit_transform(x_text)))
    print('词汇表建立完毕！')
    # print(len(x))
    # print(x)
    # print(type(x))

    """随机模糊数据，即打乱各个元素的顺序，重新洗牌"""
    np.random.seed(10)
    #np.range()返回的是range object，而np.nrange()返回的是numpy.ndarray()
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    """划分训练集/测试集，此处直接切分"""
    #此处加负号表示是从列表的后面开始查找对应位置
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    # dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    # print('划分索引：', dev_sample_index)
    # x_train, x_dev = x_shuffled[:dev_sample_index], x[dev_sample_index:]
    # y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    """使用sklearn中的cross-validation划分数据集"""
    x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=0.1, random_state=10)

    print('数据集构造完毕，信息如下：')
    print('训练集样本数目：', len(x_train))
    print('开发集样本数目：', len(x_dev))
    print('训练集标签数目：', len(y_train))
    print('开发集标签数目：', len(y_dev))
    # print(type(y_dev))

    del x, y, x_shuffled, y_shuffled

    print('词汇表 Size：', len(vocab_processor.vocabulary_))

    print('x的数据类型：', type(x_train[1][1]))
    print('y的数据类型：', type(y_train[1][1]))

    return x_train, x_dev, y_train, y_dev, vocab_processor

# 创建batch迭代模块
def batch_generater(data, batch_size, num_epochs, shuffle=True): # shuffle=True洗牌
    """
        Generates a batch iterator for a dataset.
        """
    # 每次只输出shuffled_data[start_index:end_index]这么多
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1  # 每一个epoch有多少个batch_size
    print('一个epoch包括%d个batch'%num_batches_per_epoch)
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))  # 洗牌
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size  # 当前batch的索引开始
            end_index = min((batch_num + 1) * batch_size, data_size)  # 判断下一个batch是不是超过最后一个数据了
            yield shuffled_data[start_index:end_index]
# batch_generater()

# if __name__ == '__main__':
#     construct_dataset()