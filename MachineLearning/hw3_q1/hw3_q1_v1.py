import tensorflow as tf
import random
import numpy as np


def one_hot_generator(length, digits=8):
    # How many digits in each output list
    a = []
    for n in range(0, length):
        n = random.randint(0, digits-1)
        a.append(n)
    output = np.eye(digits)[a]
    return output


train = one_hot_generator(1257)
test = one_hot_generator(540)

INOUT_LAYER_SIZE = 8
HIDDEN_LAYER_SIZE = 1


with tf.name_scope("hidden_layer"):
    with tf.name_scope("hidden_layer/weights"):
        h_weights = tf.variable(tf.random_normal([INOUT_LAYER_SIZE, HIDDEN_LAYER_SIZE]), name="h_W")
        tf.summary.histogram("hidden_layer/weights", h_weights)
    with tf.name_scope("hidden_layer/biases"):
        h_biases = tf.variable(tf.zeros[1, HIDDEN_LAYER_SIZE], name='h_b')
        tf.summary.histogram("hidden_layer/biases", h_biases)
    with tf.name_scope("hid_Wx_plus_b"):
        hid_Wx_plus_b = tf.add(tf.matmul(inputs, h_weights), h_biases)
    h_outputs = h_actv_fn(hid_Wx_plus_b, )
    tf.summary.histogram("hidden_layer/outputs", h_outputs)

with tf.name_scope("output_layer"):
    with tf.name_scope("output_layer/weights"):
        o_weights = tf.Variable(tf.random_normal([HIDDEN_LAYER_SIZE, INOUT_LAYER_SIZE]), name='o_w')
        tf.summary.histogram("output_layer/weights", o_weights)
    with tf.name_scope("output_layer/biased"):
        o_biases = tf.Variable(tf.zeros[1, INOUT_LAYER_SIZE], name='o_b')
        tf.summary.histogram("output_layer/biases", o_biases)
    with tf.name_scope("out_Wx_plus_b"):
        out_Wx_plus_b = tf.add(tf.matmul(h_outputs, o_weights), o_biases)
    o_outputs = o_actv_fn(out_Wx_plus_b, )
    tf.summary.histogram("output_layer/outputs", o_outputs)







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
for i in range(30000):
    # here to determine the keeping probability
    sess.run(train_step, feed_dict={xs: train, ys: train})
    if i % 50 == 0:
        # record loss
        train_result = sess.run(merged, feed_dict={xs: train, ys: train})
        test_result = sess.run(merged, feed_dict={xs: test, ys: test})
        train_writer.add_summary(train_result, i)
        test_writer.add_summary(test_result, i)
print("Done!!!")
