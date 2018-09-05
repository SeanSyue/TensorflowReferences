import tensorflow as tf


class DeepNN:
    @staticmethod
    def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
        layer_name = 'layer%s' % n_layer
        with tf.name_scope(layer_name):

            with tf.name_scope('weights'):
                Weights = tf.Variable(tf.truncated_normal([in_size, out_size]), name='W')
                tf.summary.histogram(layer_name + '/weights', Weights)

            with tf.name_scope('biases'):
                initial_b = tf.zeros([1, out_size]) + 0.6
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

    @staticmethod
    def produce_placeholder(node_list):
        with tf.name_scope('inputs'):
            x = tf.placeholder(tf.float32, [None, node_list[0]], name='x_input')
            y = tf.placeholder(tf.float32, [None, 1], name='y_input')
        return x, y

    @staticmethod
    def deep_1hidden(node_list):
        x, y = DeepNN.produce_placeholder(node_list)

        # add hidden layer
        l1 = DeepNN.add_layer(x, node_list[0], node_list[1], n_layer=1, activation_function=tf.nn.sigmoid)
        # add output layer
        y_ = DeepNN.add_layer(l1, node_list[1], node_list[2], n_layer='_out')

        return x, y, y_

    @staticmethod
    def deep_2hidden(node_list):
        x, y = DeepNN.produce_placeholder(node_list)

        # add hidden layer
        l1 = DeepNN.add_layer(x, node_list[0], node_list[1], n_layer=1, activation_function=tf.nn.relu)
        l2 = DeepNN.add_layer(l1, node_list[1], node_list[2], n_layer=2, activation_function=tf.nn.relu)
        # add output layer
        y_ = DeepNN.add_layer(l2, node_list[2], node_list[3], n_layer='_out')

        return x, y, y_

    @staticmethod
    def deep_4hidden(node_list):
        x, y = DeepNN.produce_placeholder(node_list)

        # add hidden layer
        l1 = DeepNN.add_layer(x, node_list[0], node_list[1], n_layer=1, activation_function=tf.nn.relu)
        l2 = DeepNN.add_layer(l1, node_list[1], node_list[2], n_layer=2, activation_function=tf.nn.relu)
        l3 = DeepNN.add_layer(l2, node_list[2], node_list[3], n_layer=3, activation_function=tf.nn.relu)
        l4 = DeepNN.add_layer(l3, node_list[3], node_list[4], n_layer=4, activation_function=tf.nn.relu)
        # add output layer
        y_ = DeepNN.add_layer(l4, node_list[4], node_list[5], n_layer='_out')

        return x, y, y_

    @staticmethod
    def deep_6hidden(node_list):
        x, y = DeepNN.produce_placeholder(node_list)

        # add hidden layer
        l1 = DeepNN.add_layer(x, node_list[0], node_list[1], n_layer=1, activation_function=tf.nn.relu)
        l2 = DeepNN.add_layer(l1, node_list[1], node_list[2], n_layer=2, activation_function=tf.nn.relu)
        l3 = DeepNN.add_layer(l2, node_list[2], node_list[3], n_layer=3, activation_function=tf.nn.relu)
        l4 = DeepNN.add_layer(l3, node_list[3], node_list[4], n_layer=4, activation_function=tf.nn.relu)
        l5 = DeepNN.add_layer(l4, node_list[4], node_list[5], n_layer=5, activation_function=tf.nn.relu)
        l6 = DeepNN.add_layer(l5, node_list[5], node_list[6], n_layer=6, activation_function=tf.nn.relu)
        # add output layer
        y_ = DeepNN.add_layer(l6, node_list[6], node_list[7], n_layer='_out')

        return x, y, y_


def train_and_analyze(y_, y):
    # Define cost function
    with tf.name_scope('loss'):
        # loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_) * DIFF_FACTOR, reduction_indices=[1]))
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y)
        loss = tf.reduce_mean(cross_entropy)

    # Train step
    with tf.name_scope('train_step'):
        train_step = tf.train.AdamOptimizer(0.6).minimize(loss)

    # Define accuracy
    with tf.name_scope('accuracy'):
        correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        tf.summary.scalar('accuracy:', accuracy)
        print(tf.shape(accuracy))
    return loss, train_step, accuracy


FILE = 'C:/bank/data_set/example/example_train.csv'

queue = tf.train.string_input_producer([FILE, FILE])
reader = tf.TextLineReader(skip_header_lines=1, name='reader')
_, value = reader.read(queue)
dec = tf.decode_csv(value, record_defaults=[[0]] * 5)


# features = batch[:-2]
# label = batch[-2:-1]

with tf.Session() as sess:
    batch = tf.train.batch(dec, batch_size=5)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(1, 22):
        print("loop:", i)
        print("dec:", sess.run(dec))
        print("batch:", sess.run(batch))
        # print("features:", sess.run(features))
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    coord.request_stop()
    coord.join(threads)
