# https://github.com/MorvanZhou/tutorials/blob/master/tensorflowTUT/tf11_build_network/full_code.py
from __future__ import print_function
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt


def one_hot_generator(length, digits=8):
    # How many digits in each output list
    a = []
    for n in range(0, length):
        n = random.randint(0, digits-1)
        a.append(n)
    output = np.eye(digits)[a]
    return output


def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]), dtype=tf.float32)
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, dtype=tf.float32)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


n_sample = 500
train = one_hot_generator(n_sample)
test = one_hot_generator(50)


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32)
ys = tf.placeholder(tf.float32)
# number of nodes in hidden layer
n_node = 8
# add hidden layer
hid = add_layer(xs, 8, n_node, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(hid, n_node, 8, activation_function=tf.nn.relu)

# the error between prediction and real data
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))

step = tf.train.AdamOptimizer(8e-4).minimize(loss)

# important step
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

train_list = []
test_list = []
iter_list = []
for i in range(8001):
    # training
    iter_list.append(i)

    sess.run(step, feed_dict={xs: train, ys: train})
    sess.run(step, feed_dict={xs: test, ys: test})

    train_est = sess.run(loss, feed_dict={xs: train, ys: train})
    test_est = sess.run(loss, feed_dict={xs: test, ys: test})

    train_list.append(train_est)
    test_list.append(test_est)

    if i % 100 == 0:
        print("iteration:", format(i))
        print("train lost:", train_est)
        print("test lost", test_est)
        # a = tf.Print(prediction, [prediction], message="This is a: ")
        # with tf.Session() as sess:
        #     b = tf.add(a, a).eval()
        # print(b)


plt.plot(iter_list, train_list, label='Train')
plt.plot(iter_list, test_list, label='Test')
# plt.xlim([1000, 3000])
# plt.ylim([0, 3])
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title("{} nodes, {} interations".format(n_node, i))
plt.legend()
plt.show()
