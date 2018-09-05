# Source:
# https://github.com/Kulbear/tensorflow-for-deep-learning-research/issues/2

import tensorflow as tf

# if you have a Tensor a, and a Session sess like me in the code,
# then a.eval() and sess.run(a) are exactly one thing.
a = tf.Variable(5, name="scalar")
with tf.Session() as sess:
    sess.run(a.initializer)
    assert a.eval() == sess.run(a)

# the most important difference is that you can use sess.run() to fetch the values of many tensors
b = tf.Variable(4, name="scalar")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    assert a.eval() == sess.run(a)
    print(a.eval())  # 5
    print(b.eval())  # 4
    print(sess.run([a, b]))  # [5, 4]
