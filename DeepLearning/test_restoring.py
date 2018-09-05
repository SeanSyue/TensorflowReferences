# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

filename_queue = tf.train.string_input_producer(['D:/bank/dataset/bank-additional-full.csv'])
model_filepath = "D:/bank/checkpoint/"
iteration = 3000


def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    # add one more layer and return the output of this layer
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            initial_W = tf.truncated_normal([in_size, out_size], stddev=0.6)
            Weights = tf.Variable(initial_W, dtype=tf.float32, name='W')
            tf.summary.histogram(layer_name + '/weights', Weights)
        with tf.name_scope('biases'):
            initial_b = tf.zeros([1, out_size])+0.6
            biases = tf.Variable(initial_b, dtype=tf.float32, name='b')
            tf.summary.histogram(layer_name + '/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
        tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs


# Make up some real data
x_np = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_np.shape)
y_np = np.square(x_np) - 0.5 + noise
x_data = tf.convert_to_tensor(x_np, dtype=tf.float32)
y_data = tf.convert_to_tensor(y_np, dtype=tf.float32)


with tf.name_scope('inputs'):
    x = tf.placeholder(tf.float32, [None, 1], name='x_input')
    y_ = tf.placeholder(tf.float32, [None, 1], name='y_input')

# add hidden layer
l1 = add_layer(x, 1, 10, n_layer=1, activation_function=tf.nn.relu)
# add output layer
target_conv = add_layer(l1, 10, 1, n_layer=2, activation_function=None)


# Define cost function
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(abs(target_conv - y_) * 10)
    tf.summary.scalar('cross_entropy:', cross_entropy)

# Train step
with tf.name_scope('train_step'):
    train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(target_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # tf.summary.scalar('accuracy:', accuracy)

# Define error
with tf.name_scope('error'):
    error = tf.subtract(y_, target_conv)
    # tf.summary.scalar('error:', error)


saver = tf.train.Saver()
merged = tf.summary.merge_all()
with tf.Session() as sess:
    saver.restore(sess, model_filepath+'train_3000.ckpt')
    writer = tf.summary.FileWriter('C:/Users/Sean/Desktop/bbb', sess.graph)
    for i in range(iteration+1):
        sess.run(train_step, feed_dict={x: x_data.eval(), y_: y_data.eval()})
        if i % 100 == 0:
            [train_error, predict, result] = sess.run([error, target_conv, merged],
                                                      feed_dict={x: x_data.eval(), y_: y_data.eval()})
            writer.add_summary(result, i)
            # print("step %d, error %g, predict %g" % (i, train_error, predict))
            print("step %d" % i)
