import tensorflow as tf
from DeepLearning.ProjectMain.Project_v3_trn import neural_network, NODE_LIST, FEATURE_COUNT, LABEL_COUNT, data_splitter


model_filepath = 'C:/bank/checkpoint/v3_up7/d5_h1_m9_s18/v3_2000.ckpt'
TEST_FILE = 'C:/bank/data_set/tf/bank_social_tf_up7.csv'


reshape_features, _, key = data_splitter(TEST_FILE, FEATURE_COUNT, LABEL_COUNT)
x = tf.placeholder(tf.float32, [None, FEATURE_COUNT])
y_ = tf.placeholder(tf.float32, [None, LABEL_COUNT])
target_conv = neural_network(x, NODE_LIST)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=target_conv, labels=y_)
correct_predictions = tf.equal(tf.argmax(target_conv, axis=1), tf.argmax(y_, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


saver = tf.train.Saver()
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    saver.restore(sess, model_filepath)

    #  測試資料共8000筆
    for _ in range(8000):
        predict = target_conv.eval(feed_dict={x: reshape_features.eval()})
        print("%g" % predict)

    coord.request_stop()
    coord.join(threads)


#    for i in range(iteration):
#        predict=target_conv.eval(feed_dict={x: np.reshape([0.21,0.14,0.82], (-1, 3))})
#        print ("step %d, predict %g"%(i, predict))


