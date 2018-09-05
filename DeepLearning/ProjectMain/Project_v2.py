import tensorflow as tf
from DeepLearning.ProjectMain import DeepNN

# TRAIN_FILE = 'C:/bank/data_set/bank_dummy_train_up.csv'
# TEST_FILE = 'C:/bank/data_set/bank_dummy_test.csv'
# FEATURE_COUNT = 56

# TRAIN_FILE = 'C:/bank/data_set/bank_train_duration_dropped.csv'
# TEST_FILE = 'C:/bank/data_set/bank_test_duration_dropped.csv'
# FEATURE_COUNT = 64

# TRAIN_FILE = 'C:/bank/data_set/bank_train_y_one_hot_up.csv'
# FEATURE_COUNT = 65
# LABEL_COUNT = 2

# TRAIN_FILE = 'C:/bank/data_set/benchmark/bank_benchmark_tf.csv'
# FEATURE_COUNT = 64
# LABEL_COUNT = 2

TRAIN_FILE = 'C:/bank/data_set/tf/bank_social_tf_minus1.csv'
FEATURE_COUNT = 54
LABEL_COUNT = 2

ITERATION = 200000
LEARNING_RATE = 0.01
BATCH_SIZE = 1000

# TOTAL_LAYER = 4
NODE_LIST = (FEATURE_COUNT, FEATURE_COUNT+20, 32, LABEL_COUNT)
BIAS_OFFSET = 0.6

SUMMARY_PATH = 'C:\\bank\\summary\\tst3\\'
CHECK_POINT_PATH = 'C:/bank/checkpoint/test2'


def data_spliter(trn_file, tst_file, n_feature, n_label):
    # Loading data.
    trn_que = tf.train.string_input_producer([trn_file])
    tst_que = tf.train.string_input_producer([tst_file])
    reader = tf.TextLineReader(skip_header_lines=1)

    # Decode csv files
    _, trn_value = reader.read(trn_que)
    trn_dec = tf.decode_csv(trn_value, record_defaults=[[0.0]] * (n_feature + n_label), name='decode_train')
    _, tst_value = reader.read(tst_que)
    tst_dec = tf.decode_csv(tst_value, record_defaults=[[0.0]] * (n_feature + n_label), name='decode_test')

    # Read in training and testing set.
    trn_ftr = trn_dec[:-n_label]
    trn_lbl = trn_dec[-n_label:]
    tst_ftr = tst_dec[:-n_label]
    tst_lbl = tst_dec[-n_label:]

    # Reshape tensors
    x_train = tf.reshape(trn_ftr, [-1, n_feature], name='x_train')
    y_train = tf.reshape(trn_lbl, [-1, n_label], name='y_train')
    x_test = tf.reshape(tst_ftr, [-1, n_feature], name='x_test')
    y_test = tf.reshape(tst_lbl, [-1, n_label], name='y_test')

    return x_train, y_train, x_test, y_test


def data_spliter_batched_trn_only(trn_file, n_feature, n_label, batch_size_=1):
    # Loading data.
    trn_que = tf.train.string_input_producer([trn_file])
    reader = tf.TextLineReader(skip_header_lines=1)

    # Decode csv files
    _, trn_value = reader.read(trn_que)
    trn_dec = tf.decode_csv(trn_value, record_defaults=[[0.0]] * (n_feature + n_label), name='decode_train')

    batch = tf.train.batch(trn_dec, batch_size=batch_size_)

    # Read in training and testing set.
    trn_ftr = batch[:-n_label]
    trn_lbl = batch[-n_label:]

    # Reshape tensors
    x_train = tf.reshape(trn_ftr, [-1, n_feature], name='x_train')
    y_train = tf.reshape(trn_lbl, [-1, n_label], name='y_train')

    return x_train, y_train


def train_and_analyze(y_, y):
    # Define cost function
    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y)
        loss = tf.reduce_mean(cross_entropy)

    # Train step
    with tf.name_scope('train_step'):
        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    # Define accuracy
    with tf.name_scope('accuracy'):
        correct_predictions = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_, axis=1))
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
    x_train, y_train = data_spliter_batched_trn_only(TRAIN_FILE, FEATURE_COUNT, LABEL_COUNT, BATCH_SIZE)

    # Assign model
    x, y, y_ = DeepNN.model_selector(NODE_LIST, BIAS_OFFSET)

    # Assign evaluation methods
    loss, train_step, accuracy = train_and_analyze(y_, y)

    # Assign check point saver
    saver = tf.train.Saver()

    # Merge all summary.
    summ = tf.summary.merge_all()
    with tf.Session() as sess:

        # Assign graph writer and initialize thread.
        writer = tf.summary.FileWriter(SUMMARY_PATH, sess.graph)

        # Initialize thread
        thrd_mgr = ThreadManager()
        thrd_mgr.assign_threads()

        # Assign instances for feeding.
        train_feed = {x: x_train.eval(), y: y_train.eval()}

        # Initialize all variables.
        sess.run(tf.global_variables_initializer())

        # Train loop
        for i in range(ITERATION + 1):
            loss_, _, accuracy_, summ_ = sess.run([loss, train_step, accuracy, summ], feed_dict=train_feed)

            # print loss and accuracy, write summary.
            if i % 100 == 0:
                print("---------------------------\n"
                      f"loop: {i}\n"
                      f"loss: {loss_}\n"
                      f"train accuracy: {accuracy_}")
                writer.add_summary(summ_, i)

            # Save check point file.
            if i % 1000 == 0:
                saver.save(sess, CHECK_POINT_PATH)

        # Ask for closing thread
        thrd_mgr.join_threads()

    print("==========================================\ntensorboard --logdir=", SUMMARY_PATH, sep='')


if __name__ == '__main__':
    main()
