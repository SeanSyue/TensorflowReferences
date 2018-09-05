import tensorflow as tf

# TRAIN_FILE = 'C:/bank/data_set/bank_dummy_train.csv'
# TEST_FILE = 'C:/bank/data_set/bank_dummy_test.csv'
# FEATURE_NUMBER = 56

# TRAIN_FILE = 'C:/bank/data_set/bank_train_duration_dropped.csv'
# TEST_FILE = 'C:/bank/data_set/bank_test_duration_dropped.csv'
# FEATURE_NUMBER = 64

TRAIN_FILE = 'C:/bank/data_set/bank_train_up.csv'
TEST_FILE = 'C:/bank/data_set/bank_test.csv'
FEATURE_NUMBER = 65

MODEL_FILEPATH = "C:/bank/checkpoint/"
ITERATION = 10000
LEARNING_RATE = 1e-7
LABEL_FACTOR = 30
DIFF_FACTOR = 70
# SUMMARY_DIRECTORY = 'C:/Users/Sean/Desktop/test2/'
SUMMARY_DIRECTORY = 'C:/bank/summary/'


def pre_processing(trn_file, tst_file):
    # Loading data.
    trn_que = tf.train.string_input_producer([trn_file])
    tst_que = tf.train.string_input_producer([tst_file])
    reader = tf.TextLineReader(skip_header_lines=1)

    # Decode csv files
    _, trn_value = reader.read(trn_que)
    trn_dec = tf.decode_csv(trn_value, record_defaults=[[0.0]]*(FEATURE_NUMBER+1), name='decode_train')
    _, tst_value = reader.read(tst_que)
    tst_dec = tf.decode_csv(tst_value, record_defaults=[[0.0]] * (FEATURE_NUMBER + 1), name='decode_test')

    # Read in training and testing set.
    trn_ftr = trn_dec[:-1]
    trn_lbl = trn_dec[-1]
    tst_ftr = tst_dec[:-1]
    tst_lbl = tst_dec[-1]

    # Reshape tensors
    x_train = tf.reshape(trn_ftr, [-1, FEATURE_NUMBER], name='x_train')
    y_train = tf.reshape(trn_lbl*LABEL_FACTOR, [-1, 1], name='y_train')
    x_test = tf.reshape(tst_ftr, [-1, FEATURE_NUMBER], name='x_test')
    y_test = tf.reshape(tst_lbl*LABEL_FACTOR, [-1, 1], name='y_test')

    return x_train, y_train, x_test, y_test


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


def assign_measurement(y_, y):
    # Define cost function
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_)*DIFF_FACTOR, reduction_indices=[1]))
        # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y)
        # loss = tf.reduce_mean(cross_entropy)

    # Train step
    with tf.name_scope('train_step'):
        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    # Define accuracy
    with tf.name_scope('accuracy'):
        correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        tf.summary.scalar('accuracy:', accuracy)
        print(tf.shape(accuracy))
    return loss, train_step, accuracy


class SummaryManager:
    def __init__(self, session, summary):
        self.sess = session
        self.summ = summary
        self.train_writer = None
        self.test_writer = None

    def assign_writers(self, location_list):
        self.train_writer = tf.summary.FileWriter(location_list[0], self.sess.graph)
        self.test_writer = tf.summary.FileWriter(location_list[1], self.sess.graph)

    def write_summary(self, train_feed, test_feed, i):
        train_result = self.sess.run(self.summ, feed_dict=train_feed)
        test_result = self.sess.run(self.summ, feed_dict=test_feed)
        self.train_writer.add_summary(train_result, i)
        self.test_writer.add_summary(test_result, i)


class ThreadManager:
    def __init__(self):
        self.coord = None
        self.threads = None

    def assign_threads(self):
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(coord=self.coord)

    def join_threads(self):
        self.coord.request_stop()
        self.coord.join(self.threads)


def main():
    # Prepare features and labels
    x_train, y_train, x_test, y_test = pre_processing(TRAIN_FILE, TEST_FILE)

    x, y, y_ = DeepNN.deep_1hidden((FEATURE_NUMBER, 32, 1))
    # x, y, y_ = DeepNN.deep_2hidden((FEATURE_NUMBER, FEATURE_NUMBER, 32, 1))
    # x, y, y_ = DeepNN.deep_4hidden((FEATURE_NUMBER, 84, 54, 27, 10, 1))
    # x, y, y_ = DeepNN.deep_6hidden((FEATURE_NUMBER, FEATURE_NUMBER, FEATURE_NUMBER, 132, 84, 48, 24, 1))

    # Assign evaluation methods
    loss, train_step, accuracy = assign_measurement(y_, y)

    saver = tf.train.Saver()
    summ = tf.summary.merge_all()
    with tf.Session() as sess:
        summ_mgr = SummaryManager(sess, summ)
        thrd_mgr = ThreadManager()
        summ_mgr.assign_writers((SUMMARY_DIRECTORY+'train/', SUMMARY_DIRECTORY+'predict/'))
        thrd_mgr.assign_threads()

        # Assign instances for feeding.
        train_feed = {x: x_train.eval(), y: y_train.eval()}
        test_feed = {x: x_test.eval(), y: y_test.eval()}

        # Initialize all variables.
        sess.run(tf.global_variables_initializer())

        # Train loop
        for i in range(ITERATION + 1):
            sess.run(train_step, feed_dict=train_feed)

            if i % 100 == 0:
                # Write summary and show loss.
                summ_mgr.write_summary(train_feed, test_feed, i)
                # print(f"step {i}\n  accuracy: ", sess.run(accuracy, feed_dict=test_feed))
                print(f"step {i}")
                print("  accuracy:", sess.run(accuracy, feed_dict=train_feed))
                # print("features:", x_train.eval())
                # print("labels:", y_train.eval())
                # print("type:", y_train.dtype)
                # print("type(y)", y.dtype.as_numpy_dtype)
                # print("type(y_)", y_.dtype.as_numpy_dtype)

            if i % 1000 == 0:
                saver.save(sess, MODEL_FILEPATH+'test2')
        thrd_mgr.join_threads()

    print("---------------------------\ntensorboard --logdir=", SUMMARY_DIRECTORY)


if __name__ == '__main__':
    main()
