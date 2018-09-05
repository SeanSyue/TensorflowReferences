import tensorflow as tf

# FILE_TRAIN = 'C:/bank/data_set/example2_train.csv'
# FILE_TEST = 'C:/bank/data_set/example2_test.csv'
FILE = 'C:/bank/data_set/example_train.csv'

# queue = tf.train.string_input_producer([FILE_TEST, FILE_TRAIN])
queue = tf.train.string_input_producer([FILE])
reader = tf.TextLineReader(skip_header_lines=1, name='reader')
_, value = reader.read(queue)
dec = tf.decode_csv(value, record_defaults=[[0.0]] * 5)
features = dec[:-2]
label = dec[-2:]

# batch = tf.train.batch(dec, batch_size=5)
# features = batch[:-2]
# label = batch[-3:-1]

with tf.Session() as sess:

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(1, 30):
        print("loop:", i)
        print("run:", sess.run([features, label]))
        # print("dec:", sess.run(dec))
        # print("features:", sess.run(features))
        # print("label:", sess.run(label))
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    coord.request_stop()
    coord.join(threads)
