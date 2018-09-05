import tensorflow as tf
from DeepLearning.ProjectMain import DeepNN


def train_and_analyze(y_, y):
    # Define cost function
    with tf.name_scope('loss'):
        # loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_) * DIFF_FACTOR, reduction_indices=[1]))
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y)
        loss = tf.reduce_mean(cross_entropy)

    # Train step
    with tf.name_scope('train_step'):
        train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

    # Define accuracy
    with tf.name_scope('accuracy'):
        correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        tf.summary.scalar('accuracy:', accuracy)
        print(tf.shape(accuracy))
    return loss, train_step, accuracy


FILE = 'C:/bank/data_set/example/example5_train.csv'

queue = tf.train.string_input_producer([FILE])
reader = tf.TextLineReader(skip_header_lines=1, name='reader')
_, value = reader.read(queue)
dec = tf.decode_csv(value, record_defaults=[[0.0]]*5)

batch = tf.train.batch(dec, batch_size=4)
features = batch[:-2]
label = batch[-2:]

x_batch = tf.reshape(features, [-1, 3])
y_batch = tf.reshape(label, [-1, 2])

x, y, y_ = DeepNN.deep_1hidden((3, 20, 2))
# loss, train_step, accuracy = train_and_analyze(y_, y)
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y-y_), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    sess.run(tf.global_variables_initializer())

    train_feed = {x: x_batch.eval(), y: y_batch.eval()}
    # test_feed = {x: [2, 11.1, 0.1544849], y: [0, 1]}

    for i in range(1, 501):
        if coord.should_stop():
            break
        print("loop:", i)
        print("train :", sess.run([loss, x_batch, y_batch], feed_dict=train_feed))

        # print("batch:\n", sess.run(batch))

        # print("features and labels:\n", sess.run([features, label]))

        # print("x_y_batch:\n", sess.run([x_batch, y_batch]))

        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    coord.request_stop()
    coord.join(threads)
