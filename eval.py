# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     eval
   Description :  给定样本对模型进行测试
   Author :       Stephen
   date：          2018/8/28
-------------------------------------------------
   Change Activity:
                   2018/8/28:
-------------------------------------------------
"""
__author__ = 'Stephen'

import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
import os
import time
import datetime
import csv
from CNN_model_without_L2 import Text_CNN
import process_data

# tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
# tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string('checkpoint_dir', './runs_L2/1535464991/', 'Checkpoint directory from training run')
tf.flags.DEFINE_boolean('eval_train', False, 'Evaluate on all training data')

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement") # 加上一个布尔类型的参数，要不要自动分配
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices") # 加上一个布尔类型的参数，要不要打印日志

#打印参数
FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()
print('Parameters:\n')
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print('')

if FLAGS.eval_train:
    x_raw, y_test = process_data.load_data_and_label(FLAGS.positive_data_file, FLAGS.negative_data_file)
    y_test = np.argmax(y_test, axis=1)
else:
    x_raw = ['a master piece four years in the making', 'everything is off']
    y_test = [1, 0]
print('数据读取完成！')

#将数据放入词汇表以获得词嵌入矩阵
vocab_path = os.path.join(FLAGS.checkpoint_dir, 'vocab')
#加载词汇表信息
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))
print('数据词嵌入完成！')

print('\nEvaluating......\n')

checkpoint_file = tf.train.latest_checkpoint(os.path.join(FLAGS.checkpoint_dir, 'checkpoints'))
graph  = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(allow_soft_placement = FLAGS.allow_soft_placement,
                                  log_device_placement = FLAGS.log_device_placement)
    session = tf.Session(config=session_conf)
    with session.as_default():
        # 读取保存的模型和变量
        saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_file))
        saver.restore(session, checkpoint_file)
        print('模型加载完毕！')

        #通过名字获取图中的占位符
        input_x = graph.get_operation_by_name('input_x').outputs[0]
        dropout_keep_prob = graph.get_operation_by_name('dropout_keep_prob').outputs[0]

        #获取想要评估的tensor变量
        predictions = graph.get_operation_by_name('output/predictions').outputs[0]

        #生成batch
        batches = process_data.batch_generater(list(x_test), FLAGS.batch_size, num_epochs=1, shuffle=False)

        #存储所得到的predictions
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = session.run(predictions, feed_dict={input_x: x_test_batch, dropout_keep_prob: 1.0})
            # all_predictions.append(batch_predictions)
            all_predictions = np.concatenate([all_predictions, batch_predictions])

#如果y_test定义过则打印准确度
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print('测试样本总数为：', len(y_test))
    print('Accuracy：', correct_predictions / len(y_test))

#将评估保存到csv文件
#np.column_stack():矩阵增加行或列,成为（x_raw[0], all_predictions[0]）
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, 'prediction.csv')
print('Saving evaluation to {0}'.format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)
