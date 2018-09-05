# Initialize session
import tensorflow as tf
sess = tf.Session()

# Some tensor we want to print the value of
a = tf.constant([1.0, 3.0])

# Add print operation
with tf.Session() as sess:
    print(a.eval())

