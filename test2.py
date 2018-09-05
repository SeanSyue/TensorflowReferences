import tensorflow as tf
import random
import numpy as np


def one_hot_generator(length):
    digits = 8  # How many digits in each output list
    a = []
    for i in range(0, length):
        i = random.randint(0, digits-1)
        a.append(i)
    output = np.eye(digits)[a]
    return output


# def set_weight_scalar(layer_name, n_nodes):
#     for i in range(0, n_nodes):
#         return tf.summary.scalar(layer_name+'_weight[{}]'.format(i), layer_name[i])
#
#
# def set_bias_scalar(layer_name, n_nodes):
#     for i in range(0, n_nodes):
#         return tf.summary.scalar(layer_name+'_bias[{}]'.format(i), layer_name[i])


def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    # add one more layer and return the output of this layer
    # layer_name = 'layer%s' % n_layer
    layer_name = "{}_layer".format(n_layer)
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram(layer_name + '/weights', Weights)
            tf.summary.scalar(layer_name+'/weights', tf.reduce_mean(Weights))
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            tf.summary.histogram(layer_name + '/biases', biases)
            tf.summary.scalar(layer_name + '/biases', tf.reduce_mean(biases))
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
        tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs


train = one_hot_generator(1257)
test = one_hot_generator(540)
n_hid_nodes = 1
filename = "test222"

# define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 8])
    ys = tf.placeholder(tf.float32, [None, 8])

# add output layer
l1 = add_layer(xs*5, 8, n_hid_nodes, 'hidden_layer', activation_function=tf.nn.softsign)
prediction = add_layer(l1, n_hid_nodes, 8, 'output_layer', activation_function=tf.nn.softmax)
tf.summary.scalar('out0', prediction[0][0])
tf.summary.scalar('out1', prediction[0][1])
tf.summary.scalar('out2', prediction[0][2])
tf.summary.scalar('out3', prediction[0][3])
tf.summary.scalar('out4', prediction[0][4])
tf.summary.scalar('out5', prediction[0][5])
tf.summary.scalar('out6', prediction[0][6])
tf.summary.scalar('out7', prediction[0][7])

# the loss between prediction and real data
with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))  # loss
    tf.summary.scalar('loss', cross_entropy)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
merged = tf.summary.merge_all()
# summary writer goes in here
# train_writer = tf.summary.FileWriter("C:\\Users\Sean\Desktop\Th_1_mul5_g5/train", sess.graph)
# test_writer = tf.summary.FileWriter("C:\\Users\Sean\Desktop\Th_1_mul5_g5/test", sess.graph)

train_writer = tf.summary.FileWriter("C:\\Users\Sean\Desktop\{}/train".format(filename), sess.graph)
test_writer = tf.summary.FileWriter("C:\\Users\Sean\Desktop\{}/test".format(filename), sess.graph)

init = tf.global_variables_initializer()
sess.run(init)
for i in range(200):
    # here to determine the keeping probability
    sess.run(train_step, feed_dict={xs: train, ys: train})
    if i % 50 == 0:
        # record loss
        train_result = sess.run(merged, feed_dict={xs: train, ys: train})
        test_result = sess.run(merged, feed_dict={xs: test, ys: test})
        train_writer.add_summary(train_result, i)
        test_writer.add_summary(test_result, i)
print("Done!!!")
