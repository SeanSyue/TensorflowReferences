import tensorflow as tf

TRAIN_FILE = 'C:/bank/double_up7_train.csv'
model_filepath = 'C:/bank/checkpoint/'
SUMMARY_PATH = 'C:\\bank\\summary\\'

FEATURE_COUNT = 15
LABEL_COUNT = 2
iteration = 300000
NODE_LIST = (FEATURE_COUNT, FEATURE_COUNT, 40, 15, 8, LABEL_COUNT)  # 從左到右依次是輸入、各隱含層和輸出的node數
LEARNING_RATE = 8e-4



def data_splitter(input_file, feature_count, label_count):
    filename_queue = tf.train.string_input_producer([input_file])
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)
    decoded = tf.decode_csv(value, record_defaults=[[0.0]] * (feature_count + label_count))

    reshape_features = tf.reshape(decoded[:-label_count], [-1, feature_count], name=None)
    reshape_labels = tf.reshape(decoded[-label_count:], [-1, label_count], name=None)
    return reshape_features, reshape_labels, key


def neural_network(x, node_list):
    def weight_variable(shape, name):
        initial = tf.truncated_normal(shape, stddev=0.6)
        return tf.Variable(initial, dtype=tf.float32, name=name)

    def bias_variable(shape, name):
        initial = tf.constant(0.6, shape=shape)
        return tf.Variable(initial, dtype=tf.float32, name=name)

    W_fc1 = weight_variable([node_list[0], node_list[1]], 'W_fc1')
    b_fc1 = bias_variable([node_list[1]], 'b_fc1')
    h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

    W_fc2 = weight_variable([node_list[1], node_list[2]], 'W_fc2')
    b_fc2 = bias_variable([node_list[2]], 'b_fc2')
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    W_fc3 = weight_variable([node_list[2], node_list[3]], 'W_fc3')
    b_fc3 = bias_variable([node_list[3]], 'b_fc3')
    h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)

    W_fc4 = weight_variable([node_list[3], node_list[4]], 'W_fc4')
    b_fc4 = bias_variable([node_list[4]], 'b_fc4')
    h_fc4 = tf.nn.relu(tf.matmul(h_fc3, W_fc4) + b_fc4)

    W_fc5 = weight_variable([node_list[4], node_list[5]], 'W_fc5')
    b_fc5 = bias_variable([node_list[5]], 'b_fc5')
    target_conv = tf.matmul(h_fc4, W_fc5) + b_fc5

    tf.summary.histogram('W_fc1', W_fc1)
    tf.summary.histogram('b_fc1', b_fc1)
    tf.summary.histogram('h_fc1', h_fc1)

    tf.summary.histogram('W_fc2', W_fc2)
    tf.summary.histogram('b_fc2', b_fc2)
    tf.summary.histogram('h_fc2', h_fc2)

    tf.summary.histogram('W_fc3', W_fc3)
    tf.summary.histogram('b_fc3', b_fc3)
    tf.summary.histogram('h_fc3', h_fc3)

    tf.summary.histogram('W_fc4', W_fc4)
    tf.summary.histogram('b_fc4', b_fc4)
    tf.summary.histogram('h_fc4', h_fc4)

    tf.summary.histogram('W_fc5', W_fc5)
    tf.summary.histogram('b_fc5', b_fc5)
    tf.summary.histogram('target_conv', target_conv)

    return target_conv


def main():
    #  提取訓練資料、宣告placeholder
    reshape_features, reshape_labels, _ = data_splitter(TRAIN_FILE, FEATURE_COUNT, LABEL_COUNT)
    x = tf.placeholder(tf.float32, [None, FEATURE_COUNT])
    y_ = tf.placeholder(tf.float32, [None, LABEL_COUNT])
    #  建構與NODE_LIST中的各層神經元的數目相符的神經網路模型
    target_conv = neural_network(x, NODE_LIST)

    #  定義訓練和評估方法
    # cross_entropy = tf.reduce_mean(abs(target_conv - y_) * 10)
    # accuracy = tf.subtract(y_, target_conv)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=target_conv, labels=y_)
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
    correct_predictions = tf.equal(tf.argmax(target_conv, axis=1), tf.argmax(y_, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    summ = tf.summary.merge_all()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        writer = tf.summary.FileWriter(SUMMARY_PATH + dt_str, sess.graph)
        sess.run(tf.global_variables_initializer())

        for i in range(iteration):
            # train_step.run(feed_dict={x: reshape_features.eval(), y_: reshape_labels.eval()})
            _, predict, train_accuracy, summary, = sess.run([train_step, target_conv, accuracy, summ],
                                                            feed_dict={x: reshape_features.eval(),
                                                                       y_: reshape_labels.eval()})

            if i % 100 == 0:
                # train_accuracy = accuracy.eval(feed_dict={x: reshape_features.eval(), y_: reshape_labels.eval()})
                # predict = target_conv.eval(feed_dict={x: reshape_features.eval(), y_: reshape_labels.eval()})
                # summary = summ.eval(feed_dict={x: reshape_features.eval(), y_: reshape_labels.eval()})
                print(f"step {i}, error {train_accuracy}, predict {predict}")
                # print(f"step {i}, error {train_accuracy},predict {predict}")
                writer.add_summary(summary, i)

            if i % 1000 == 0:
                saver.save(sess, f'{model_filepath}/{dt_str}/v3_{i}.ckpt')

        coord.request_stop()
        coord.join(threads)

    print(f"pred_counter:{pred_counter}\n"
          f"Accuracy: {pred_counter/(iteration/100)}\n"
          f"==========================================\n"
          f"tensorboard --logdir={SUMMARY_PATH}\\{dt_str}\\ \n"
          f"{model_filepath}/{dt_str}")


if __name__ == '__main__':
    main()
