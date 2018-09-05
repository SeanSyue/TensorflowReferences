# -*- coding: utf-8 -*-
import tensorflow as tf
import csv
import numpy as np


def relu(x, alpha=0.5, max_value=None):
    '''ReLU.

    alpha: slope of negative section.
    '''
    negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    if max_value is not None:
        x = tf.clip_by_value(x, tf.cast(0., dtype=_FLOATX),
                             tf.cast(max_value, dtype=_FLOATX))
    x -= tf.constant(alpha, dtype=_FLOATX) * negative_part
    return x


# The point for using truncated normal is to overcome saturation of tome functions like sigmoid
# (where if the value is too big/small, the neuron stops learning).
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.6)
    return tf.Variable(initial, dtype=tf.float32, name=name)


def bias_variable(shape, name):
    initial = tf.constant(0.6, shape=shape)
    return tf.Variable(initial, dtype=tf.float32, name=name)


filename_queue = tf.train.string_input_producer\
(["C:/Users/303/Downloads/KDDdataSets/dataset20/dataSets/dataSets/training/task1/train_118.csv"])
model_filepath = "C:/Users/303/Desktop/spyderworkspace/KDD/118/"
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
record_defaults = [['10'], ['0.5'], ['0.5'], ['0.5']]  # Declare exception when handling empty column
col1, col2, col3, col4 = tf.decode_csv(value, record_defaults=record_defaults)
col1 = tf.string_to_number(col1, name='ToFloat')
col2 = tf.string_to_number(col2, name='ToFloat')
col3 = tf.string_to_number(col3, name='ToFloat')
col4 = tf.string_to_number(col4, name='ToFloat')
features = tf.stack([col2, col3, col4], 0)
reshape_features = tf.reshape(features, [-1, 3], name=None)
reshape_labels = tf.reshape(col1, [-1, 1], name=None)


x = tf.placeholder(tf.float32, [None, 3])
y_ = tf.placeholder(tf.float32, [None, 1])

#x = tf.placeholder_with_default( tf.convert_to_tensor(np.zeros((batch_size,3)),tf.float32), [None,3])
#y_ = tf.placeholder_with_default(tf.convert_to_tensor(np.zeros((batch_size,1)),tf.float32), [None,1])

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


iteration = 81000
cross_entropy = tf.reduce_mean(abs(target_conv-y_)*10)  # Define cost function
train_step = tf.train.AdamOptimizer(8e-4).minimize(cross_entropy)  # Optimizer
#correct_prediction = tf.equal(tf.argmax(target_conv,1), tf.argmax(y_,1))
accuracy = tf.subtract(y_, target_conv)  # Define accuracy
saver = tf.train.Saver()
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    sess.run(tf.global_variables_initializer())
    for i in range(iteration):
        train_step.run(feed_dict={x: reshape_features.eval(), y_: reshape_labels.eval()})
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: reshape_features.eval(), y_: reshape_labels.eval()})
            predict = target_conv.eval(feed_dict={x: reshape_features.eval(), y_: reshape_labels.eval()})
            print("step %d, error %g,predict %g" % (i, train_accuracy, predict))
        if i % 1000 == 0:
            save_path = saver.save(sess, model_filepath+'model_118_%d.ckpt' % i)
    coord.request_stop()
    coord.join(threads)
