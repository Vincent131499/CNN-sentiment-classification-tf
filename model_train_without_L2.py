# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     model_train
   Description :   训练CNN模型
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
from CNN_model_without_L2 import Text_CNN
# from CNN_model_with_L2 import Text_CNN
from process_data import construct_dataset
from process_data import batch_generater
import time
import os
import datetime

"""模型超参数设置"""
tf.flags.DEFINE_integer('embedding_dim', 128, 'Dimensionality of character embedding (default:128)')
tf.flags.DEFINE_string('filter_sizes', '3,4,5', "Comma-separated filter sizes(default:'3,4,5')")
tf.flags.DEFINE_integer('num_filters', 128, 'Number of filters')
tf.flags.DEFINE_float('dropout_keep_prob', 0.5, 'Dropout keep probability(default:0.5)')
tf.flags.DEFINE_float('l2_reg_lambda', 0.0, 'L2 regularization lambda(default:0.0)')

#训练参数设置
tf.flags.DEFINE_integer('batch_size', 64, 'Batch Size(default: 64)')
tf.flags.DEFINE_integer('num_epochs', 200, 'Number of training epochs(default: 200)') # 总训练次数
tf.flags.DEFINE_integer('evaluate_every', 100, 'Evaluate model on dev set after this many steps(default: 100)') # 每训练100次测试一下
tf.flags.DEFINE_integer('checkpoint_every', 100, 'Save model after this many steps(default: 100)')# 保存一次模型
tf.flags.DEFINE_integer('num_checkpoints', 5, 'Number of checkpoints to store(default: 5)')

#运行设备参数设置
"""
allow_soft_placement设置允许TensorFlow回落的设备上时，优选的设备不存在实现的某些操作。
例如，如果我们的代码在GPU上进行操作，并且我们在没有GPU的机器上运行代码，则不使用
allow_soft_placement会导致错误。
如果设置了log_device_placement，TensorFlow会记录它放置操作的设备（CPU或GPU）。这对调试很有用。
"""
tf.flags.DEFINE_boolean('allow_soft_placement', True, 'Allow device soft device placement') # 加上一个布尔类型的参数，要不要自动分配
tf.flags.DEFINE_boolean('log_device_placement', False, 'Log placement of ops on devices') # 加上一个布尔类型的参数，要不要打印日志

# 打印一下相关初始参数
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

x_train, x_dev, y_train, y_dev, vocab_processor = construct_dataset()
# print(x_train.shape)

with tf.Graph().as_default():
    session_conf  = tf.ConfigProto(
        allow_soft_placement = FLAGS.allow_soft_placement,
        log_device_placement = FLAGS.log_device_placement)
    session = tf.Session(config=session_conf)
    with session.as_default():
        cnn_model = Text_CNN(sequence_length = x_train.shape[1],
                             num_classes = y_train.shape[1],
                             vocab_size = len(vocab_processor.vocabulary_),
                             embedding_size = FLAGS.embedding_dim,
                             filter_sizes = [int(s) for s in FLAGS.filter_sizes.split(',')], #此处应将参数中的'3,4,5'转换为整数列表
                             num_filters = FLAGS.num_filters) # 一共有几个filter
        # cnn_model = Text_CNN(
        #     sequence_length=x_train.shape[1],
        #     num_classes=y_train.shape[1], # 分几类
        #     vocab_size=len(vocab_processor.vocabulary_),
        #     embedding_size=FLAGS.embedding_dim,
        #     filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))), # 上面定义的filter_sizes拿过来，"3,4,5"按","分割
        #     num_filters=FLAGS.num_filters, # 一共有几个filter
        #     l2_reg_lambda=FLAGS.l2_reg_lambda) # l2正则化项

        #定义训练过程training procedure
        global_step = tf.Variable(0, name='global_step', trainable=False)
        #1.定义优化器Adam
        optimizer = tf.train.AdamOptimizer(1e-3)
        #2.通过优化器Adam计算梯度
        grads_and_vars = optimizer.compute_gradients(cnn_model.loss)
        #3.将梯度应用于变量并更新global_step(自增)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        #将模型和记录summary写入本地文件
        timestamp = str(int(time.time()))
        #获取完整路径
        out_dir = os.path.abspath(os.path.join(os.path.curdir, 'runs', timestamp))
        print('Writing to {}\n'.format(out_dir))

        #记录训练过程中的loss和accuracy (Summaries for loss and accuracy)
        loss_summary = tf.summary.scalar('loss', cnn_model.loss)
        accuracy_summary = tf.summary.scalar('accuracy', cnn_model.accuracy)

        """记录训练过程Train Summaries"""
        #tf.summary.merge将指定的summary组合在一起
        train_summary_op = tf.summary.merge([loss_summary, accuracy_summary])
        train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, session.graph_def)

        """记录测试过程Dev Summaries"""
        dev_summary_op = tf.summary.merge([loss_summary, accuracy_summary])
        dev_summary_dir = os.path.join(out_dir, 'summaries', 'dev')
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, session.graph_def)

        """检查点Checkpointing"""
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
        checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
        #在Tensorflow中假定路径已经存在，故我们需要判断该路径，不存在需要创建
        if os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        #创建一个Saver对象用来保存模型
        saver = tf.train.Saver(tf.all_variables())

        #词汇表信息写入本地
        vocab_processor.save(os.path.join(out_dir, 'vocab'))

        #初始化变量
        session.run(tf.initialize_all_variables())

        #定义单个训练步骤
        def train_step(x_batch, y_batch):
            feed_dict = {cnn_model.input_x: x_batch,
                         cnn_model.input_y: y_batch,
                         cnn_model.dropout_keep_prob: FLAGS.dropout_keep_prob}
            _, step, summaries, loss, accuracy = session.run(
                [train_op, global_step, train_summary_op, cnn_model.loss, cnn_model.accuracy],
                feed_dict=feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print('{}: step {}, loss {:g}, acc {:g}'.format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        """对dev set进行评估"""
        def dev_step(x_batch, y_batch, writer = None):
            feed_dict = {cnn_model.input_x: x_batch,
                         cnn_model.input_y: y_batch,
                         cnn_model.dropout_keep_prob: 1.0}
            step, summaries, loss, accuracy = session.run(
                [global_step, dev_summary_op, cnn_model.loss, cnn_model.accuracy],
                feed_dict=feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print('{}: step {}, loss {:g}, acc {:g}'.format(time_str, step, loss, accuracy))
            # train_summary_writer.add_summary(summaries, step)
            if writer:
                writer.add_summary(summaries, step)

        """模型训练"""
        #生成批次数据
        batches = batch_generater(list(zip(x_train, y_train)),
                                  batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs)
        #对每一个baych循环训练
        for batch in batches:
            #*zip()函数是zip()函数的逆过程，将zip对象变成原先组合前的数据。
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(session, global_step) # 将Session和global_step值传进来
            if current_step % FLAGS.evaluate_every == 0: # 每FLAGS.evaluate_every次每100执行一次测试
                print('\nEvaluation')
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print('')
            if current_step % FLAGS.checkpoint_every == 0: # 每checkpoint_every次执行一次保存模型
                path = saver.save(session, checkpoint_prefix, global_step=current_step) # 定义模型保存路径
                print('Saved model checkpoint to {}\n'.format(path))

