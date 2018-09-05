# -*- coding: utf-8 -*-
import tensorflow as tf


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


'''
A truncated normal distribution is derived from cutting off the tails from a normal distribution.
The point for using truncated normal is to overcome saturation of tome functions like sigmoid
(where if the value is too big/small, the neuron stops learning).
'''
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.6)
    return tf.Variable(initial, dtype=tf.float32, name=name)


def bias_variable(shape, name):
    initial = tf.constant(0.6, shape=shape)
    return tf.Variable(initial, dtype=tf.float32, name=name)


x = tf.placeholder(tf.float32, [None, 8])
y_ = tf.placeholder(tf.float32, [None, 8])
n_hidl = 3
W_fc1 = weight_variable([8, n_hidl], 'W_fc1')
b_fc1 = bias_variable([n_hidl], 'b_fc1')
h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

W_fc2 = weight_variable([n_hidl, 8], 'W_fc2')
b_fc2 = bias_variable([8], 'b_fc2')
target_conv = tf.matmul(h_fc1, W_fc2) + b_fc2


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
