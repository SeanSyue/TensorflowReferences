"""A deep MNIST classifier using convolutional layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import time

FLAGS = None


def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Build the graph for the deep net
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b

    n_batch = 50
    # model = "A"
    model = "G"
    if model == "G":
        n_iter = 6000
    elif model == "A":
        n_iter = 6000

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
        cross_entropy = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('loss', cross_entropy)

    with tf.name_scope('adam_optimizer'):
        if model == "A":
            train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
            print("Adam")
        elif model == "G":
            train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)
            print("Gradient")

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
        tf.summary.scalar("accuracy", accuracy)

    graph_location = "C:/Users/Sean/Desktop/hw3_q2(mnist)_perceptron/{}_batch{}_iter{}"\
        .format(model, n_batch, n_iter)
    sess = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(graph_location, sess.graph)
    sess.run(tf.global_variables_initializer())
    start_time = time.time()
    for i in range(n_iter):
        begin_step = time.time()
        batch = mnist.train.next_batch(n_batch)  # load 50 training examples in each training iteration.
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})
        duration = time.time() - begin_step
        if i % 100 == 0:
            [train_accuracy, s] = sess.run([accuracy, merged], feed_dict={x: batch[0], y_: batch[1]})
            writer.add_summary(s, i)
            print('step %d, duration %.3f, training accuracy %g' % (i, duration, train_accuracy))

    total_time = time.time() - start_time
    avg_total_time = total_time / n_iter
    final_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    print('totaltime: %.3fs. average time: %.3fs. test accuracy: %g' % (total_time, avg_total_time, final_accuracy))
    print("tensorboard --logdir=\"{}\"".format(graph_location))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
