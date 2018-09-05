# -*- coding: utf-8 -*-

import tensorflow as tf
import csv
import numpy as np


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=1.5)
    return tf.Variable(initial, dtype=tf.float32, name=name)


def bias_variable(shape, name):
    initial = tf.constant(1.5, shape=shape)
    return tf.Variable(initial, dtype=tf.float32, name=name)


csv_dir = 'C:/Users/303/Desktop/test2_task1.csv'
input_file = open(csv_dir, 'r')
#filename_queue = tf.train.string_input_producer(["C:/Users/303/Downloads/KDDdataSets/dataset20/dataSets/dataSets/training/task1/train_104.csv"])
#model_filepath="C:/Users/303/Desktop/spyderworkspace/KDD/100/model_100_50000.ckpt"
#model_filepath="C:/Users/303/Desktop/spyderworkspace/KDD/101/model_101_50000.ckpt"
#model_filepath="C:/Users/303/Desktop/spyderworkspace/KDD/102/model_102_50000.ckpt"
#model_filepath="C:/Users/303/Desktop/spyderworkspace/KDD/103/model_103_50000.ckpt"
#model_filepath="C:/Users/303/Desktop/spyderworkspace/KDD/104/model_104_50000.ckpt"
#model_filepath="C:/Users/303/Desktop/spyderworkspace/KDD/105/model_105_50000.ckpt"
#model_filepath="C:/Users/303/Desktop/spyderworkspace/KDD/106/model_106_50000.ckpt"
#model_filepath="C:/Users/303/Desktop/spyderworkspace/KDD/107/model_107_50000.ckpt"
#model_filepath="C:/Users/303/Desktop/spyderworkspace/KDD/108/model_108_50000.ckpt"
#model_filepath="C:/Users/303/Desktop/spyderworkspace/KDD/109/model_109_50000.ckpt"
#model_filepath="C:/Users/303/Desktop/spyderworkspace/KDD/110/model_110_50000.ckpt"
#model_filepath="C:/Users/303/Desktop/spyderworkspace/KDD/111/model_111_50000.ckpt"
#model_filepath="C:/Users/303/Desktop/spyderworkspace/KDD/112/model_112_50000.ckpt"
#model_filepath="C:/Users/303/Desktop/spyderworkspace/KDD/113/model_113_50000.ckpt"
#model_filepath="C:/Users/303/Desktop/spyderworkspace/KDD/114/model_114_47000.ckpt"
#model_filepath="C:/Users/303/Desktop/spyderworkspace/KDD/115/model_115_50000.ckpt"
#model_filepath="C:/Users/303/Desktop/spyderworkspace/KDD/116/model_116_50000.ckpt"
#model_filepath="C:/Users/303/Desktop/spyderworkspace/KDD/117/model_117_77000.ckpt"
model_filepath="C:/Users/303/Desktop/spyderworkspace/KDD/118/model_118_78000.ckpt"
#model_filepath="C:/Users/303/Desktop/spyderworkspace/KDD/119/model_119_50000.ckpt"
#model_filepath="C:/Users/303/Desktop/spyderworkspace/KDD/120/model_120_50000.ckpt"
#model_filepath="C:/Users/303/Desktop/spyderworkspace/KDD/121/model_121_50000.ckpt"
#model_filepath="C:/Users/303/Desktop/spyderworkspace/KDD/122/model_122_50000.ckpt"
#model_filepath="C:/Users/303/Desktop/spyderworkspace/KDD/123/model_123_50000.ckpt"
iteration=10
#reader = tf.TextLineReader()
#key, value = reader.read(filename_queue)
#record_defaults = [['10'], ['0.5'], ['0.5'], ['0.5']]
#col1, col2, col3, col4= tf.decode_csv(value, record_defaults=record_defaults)
#col1=tf.string_to_number(col1, name='ToFloat')
#col2=tf.string_to_number(col2, name='ToFloat')
#col3=tf.string_to_number(col3, name='ToFloat')
#col4=tf.string_to_number(col4, name='ToFloat')
#features = tf.stack([col2, col3, col4],0)
#reshape_features=tf.reshape(features,[-1,3],name=None)
#reshape_labels=tf.reshape(col1,[-1,1],name=None)

x = tf.placeholder(tf.float32, [None, 3])
y_ = tf.placeholder(tf.float32, [None, 1])

W_fc1 = weight_variable([3, 5], 'W_fc1')
b_fc1 = bias_variable([5], 'b_fc1')
h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

W_fc2 = weight_variable([5, 10], 'W_fc2')
b_fc2 = bias_variable([10], 'b_fc2')
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

W_fc3 = weight_variable([10, 5], 'W_fc3')
b_fc3 = bias_variable([5], 'b_fc3')
h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)

W_fc4 = weight_variable([5, 3], 'W_fc4')
b_fc4 = bias_variable([3], 'b_fc4')
h_fc4 = tf.nn.relu(tf.matmul(h_fc3, W_fc4) + b_fc4)

W_fc5 = weight_variable([3, 1], 'W_fc5')
b_fc5 = bias_variable([1], 'b_fc5')
target_conv = tf.nn.relu(tf.matmul(h_fc4, W_fc5) + b_fc5)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(target_conv)))
accuracy = tf.subtract(y_, target_conv)
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, model_filepath)
    fieldnames = ['date', 'weather_predict', 'weekday', 'time']
    full_data = csv.DictReader(input_file, fieldnames=fieldnames)
    for row in csv.DictReader(input_file):
        col2 = float(row['weather_predict'])
        col3 = float(row['weekday'])
        col4 = float(row['time'])
        predict = target_conv.eval(feed_dict={x: np.reshape([col2, col3, col4], (-1, 3))})
        print("%g" % predict)
#    for i in range(iteration):
#        predict=target_conv.eval(feed_dict={x: np.reshape([0.21,0.14,0.82], (-1, 3))})
#        print ("step %d, predict %g"%(i, predict))
