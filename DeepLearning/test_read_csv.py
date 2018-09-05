import tensorflow as tf

TRAIN_FILE = 'C:/bank/data_set/bank_train.csv'

with tf.Session() as sess:
    queue = tf.train.string_input_producer([TRAIN_FILE], num_epochs=None)
    reader = tf.TextLineReader(skip_header_lines=1, name='reader')
    _, value = reader.read(queue)
    dec = tf.decode_csv(value, record_defaults=[[0.0]]*66)

    features = tf.reshape(dec[:-1], [-1, 65])
    label = tf.reshape(dec[-1], [-1, 1])

    batch = tf.train.batch([features, label], batch_size=10)

    print("type", tf.shape(features))
    print("type", tf.shape(label))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(2):
        print("loop:", i)

        # print("features:\n", sess.run(features))
        # print("label:\n", sess.run(label))

        print("batch:\n", sess.run(batch))

        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    coord.request_stop()
    coord.join(threads)
