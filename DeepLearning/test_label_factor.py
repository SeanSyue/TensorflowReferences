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
LABEL_FACTOR = 50
COST_FACTOR = 70
# SUMMARY_DIRECTORY = 'C:/Users/Sean/Desktop/test2/'
SUMMARY_DIRECTORY = 'C:/bank/summary/'


def pre_processing(trn_file, tst_file):
    # Loading data.
    trn_que = tf.train.string_input_producer([trn_file])
    tst_que = tf.train.string_input_producer([tst_file])
    reader = tf.TextLineReader(skip_header_lines=1)

    # Read in training set.
    _, trn_value = reader.read(trn_que)
    trn_dec = tf.decode_csv(trn_value, record_defaults=[[0.0]]*(FEATURE_NUMBER+1),
                            name='decode_train')  # Data type: Int32 or Float32
    trn_ftr = trn_dec[:-1]
    trn_lbl = trn_dec[-1]

    # Read in testing set.
    _, tst_value = reader.read(tst_que)
    tst_dec = tf.decode_csv(tst_value, record_defaults=[[0.0]]*(FEATURE_NUMBER+1),
                            name='decode_test')  # Data type: Int32 or Float32
    tst_ftr = tst_dec[:-1]
    tst_lbl = tst_dec[-1]

    # Reshape tensors
    rs_train_features = tf.reshape(trn_ftr, [-1, FEATURE_NUMBER], name='rs_train_features')
    rs_train_labels = tf.reshape(trn_lbl*LABEL_FACTOR, [-1, 1], name='rs_train_labels')
    rs_test_features = tf.reshape(tst_ftr, [-1, FEATURE_NUMBER], name='rs_test_features')
    rs_test_labels = tf.reshape(tst_lbl*LABEL_FACTOR, [-1, 1], name='rs_test_labels')

    return rs_train_features, rs_train_labels, rs_test_features, rs_test_labels


class DeepNN:

    @staticmethod
    def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
        # add one more layer and return the output of this layer
        layer_name = 'layer%s' % n_layer
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                Weights = tf.Variable(tf.truncated_normal([in_size, out_size]), name='W')
                tf.summary.histogram(layer_name + '/weights', Weights)
            with tf.name_scope('biases'):
                initial_b = tf.zeros([1, out_size]) + 6.0
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
            y_ = tf.placeholder(tf.float32, [None, 1], name='y_input')
        return x, y_

    @staticmethod
    def deep_1hidden(node_list):
        x, y_ = DeepNN.produce_placeholder(node_list)

        # add hidden layer
        l1 = DeepNN.add_layer(x, node_list[0], node_list[1], n_layer=1, activation_function=tf.nn.sigmoid)
        # add output layer
        target_conv = DeepNN.add_layer(l1, node_list[1], node_list[2], n_layer='_out', activation_function=tf.nn.sigmoid)

        return x, y_, target_conv

    @staticmethod
    def deep_2hidden(node_list):
        x, y_ = DeepNN.produce_placeholder(node_list)

        # add hidden layer
        l1 = DeepNN.add_layer(x, node_list[0], node_list[1], n_layer=1, activation_function=tf.nn.relu)
        l2 = DeepNN.add_layer(l1, node_list[1], node_list[2], n_layer=2, activation_function=tf.nn.relu)
        # add output layer
        target_conv = DeepNN.add_layer(l2, node_list[2], node_list[3], n_layer='_out', activation_function=tf.nn.relu)

        return x, y_, target_conv

    @staticmethod
    def deep_4hidden(node_list):
        x, y_ = DeepNN.produce_placeholder(node_list)

        # add hidden layer
        l1 = DeepNN.add_layer(x, node_list[0], node_list[1], n_layer=1, activation_function=tf.nn.relu)
        l2 = DeepNN.add_layer(l1, node_list[1], node_list[2], n_layer=2, activation_function=tf.nn.relu)
        l3 = DeepNN.add_layer(l2, node_list[2], node_list[3], n_layer=3, activation_function=tf.nn.relu)
        l4 = DeepNN.add_layer(l3, node_list[3], node_list[4], n_layer=4, activation_function=tf.nn.relu)
        # add output layer
        target_conv = DeepNN.add_layer(l4, node_list[4], node_list[5], n_layer='_out', activation_function=tf.nn.relu)

        return x, y_, target_conv

    @staticmethod
    def deep_6hidden(node_list):
        x, y_ = DeepNN.produce_placeholder(node_list)

        # add hidden layer
        l1 = DeepNN.add_layer(x, node_list[0], node_list[1], n_layer=1, activation_function=tf.nn.relu)
        l2 = DeepNN.add_layer(l1, node_list[1], node_list[2], n_layer=2, activation_function=tf.nn.relu)
        l3 = DeepNN.add_layer(l2, node_list[2], node_list[3], n_layer=3, activation_function=tf.nn.relu)
        l4 = DeepNN.add_layer(l3, node_list[3], node_list[4], n_layer=4, activation_function=tf.nn.relu)
        l5 = DeepNN.add_layer(l4, node_list[4], node_list[5], n_layer=5, activation_function=tf.nn.relu)
        l6 = DeepNN.add_layer(l5, node_list[5], node_list[6], n_layer=6, activation_function=tf.nn.relu)
        # add output layer
        target_conv = DeepNN.add_layer(l6, node_list[6], node_list[7],
                                       n_layer='_out', activation_function=tf.nn.softmax)

        return x, y_, target_conv


def assign_measurement(target_conv, y_):
    # Define cost function
    with tf.name_scope('cost_fn'):
        cost_fn = tf.reduce_mean(tf.reduce_sum(tf.square(y_ - target_conv)*COST_FACTOR, reduction_indices=[1]))

    # Train step
    with tf.name_scope('train_step'):
        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost_fn)

    with tf.name_scope('accuracy'):
        correct_predictions = tf.equal(target_conv, y_)
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        tf.summary.scalar('accuracy:', accuracy)
        print(tf.shape(accuracy))
    return cost_fn, train_step, accuracy


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
    rs_train_features, rs_train_labels, rs_test_features, rs_test_labels = pre_processing(TRAIN_FILE, TEST_FILE)

    # x, y_, target_conv = DeepNN.deep_1hidden((FEATURE_NUMBER, 32, 1))
    # x, y_, target_conv = DeepNN.deep_2hidden((FEATURE_NUMBER, FEATURE_NUMBER, 32, 1))
    x, y_, target_conv = DeepNN.deep_4hidden((FEATURE_NUMBER, 84, 54, 27, 10, 1))
    # x, y_, target_conv = DeepNN.deep_6hidden((FEATURE_NUMBER, FEATURE_NUMBER, FEATURE_NUMBER, 132, 84, 48, 24, 1))

    # Assign evaluation methods
    cost_fn, train_step, accuracy = assign_measurement(target_conv, y_)

    saver = tf.train.Saver()
    summ = tf.summary.merge_all()
    with tf.Session() as sess:
        summ_mgr = SummaryManager(sess, summ)
        thrd_mgr = ThreadManager()
        summ_mgr.assign_writers((SUMMARY_DIRECTORY+'train/', SUMMARY_DIRECTORY+'predict/'))
        thrd_mgr.assign_threads()

        # Assign instances for feeding.
        train_feed = {x: rs_train_features.eval(), y_: rs_train_labels.eval()}
        test_feed = {x: rs_test_features.eval(), y_: rs_test_labels.eval()}

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
                print("  cost_fn:", sess.run(cost_fn, feed_dict=train_feed))
                # print("features:", rs_train_features.eval())
                # print("labels:", rs_train_labels.eval())
                # print("type:", rs_train_labels.dtype)
                # print("type(y)", y_.dtype.as_numpy_dtype)
                # print("type(target_conv)", target_conv.dtype.as_numpy_dtype)

            if i % 1000 == 0:
                saver.save(sess, MODEL_FILEPATH+'test2')
        thrd_mgr.join_threads()

    print("---------------------------\ntensorboard --logdir=", SUMMARY_DIRECTORY)


if __name__ == '__main__':
    main()
