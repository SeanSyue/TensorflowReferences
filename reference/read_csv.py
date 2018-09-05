# Source:
# https://www.tensorflow.org/api_guides/python/reading_data
import tensorflow as tf

# creates a FIFO queue for holding the filenames until the reader needs them.
# The following line is equivalent to :
# filename_queue = tf.train.string_input_producer(["file0.csv", "file1.csv"])
filename_queue = tf.train.string_input_producer([("file%d" % i) for i in range(2)])

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the decoded result.
# Try a simpler expression:
# col1, col2, col3, col4, col5 = tf.decode_csv(value, record_defaults=[[1]]*5)
record_defaults = [[1], [1], [1], [1], [1]]
col1, col2, col3, col4, col5 = tf.decode_csv(
    value, record_defaults=record_defaults)
features = tf.stack([col1, col2, col3, col4])

with tf.Session() as sess:
    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(1200):
        # Retrieve a single instance:
        example, label = sess.run([features, col5])

    coord.request_stop()
    coord.join(threads)
