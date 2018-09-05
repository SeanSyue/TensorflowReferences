import tensorflow as tf
import shutil

'''

自製骨架tf recoder 參考資料
http://blog.csdn.net/u012759136/article/details/52232266

找了一整天 終於找到同時使用eval與train同時進入的方法，
https://github.com/tensorflow/tensorflow/issues/2514#issuecomment-246026683
 LSTM single direction basic single stream  
'''


def build_input_SK(logfilePath, n_inputs, data_path, batch_size, num_classes, max_seq_len):
    '''
    :param skeleton_maxlength:
    :param data_path: train , eval
    :param batch_size: train
    :param num_classes:
    :return:
    '''
    tf.logging.set_verbosity('INFO')
    configlogWriter = open(logfilePath + '\\ConfigLog.txt', mode='a', encoding='utf-8')

    # 複製檔案作為紀錄 > 當前檔案, 目標路徑
    # shutil.copyfile(__file__, logfilePath + '\\' + __file__.split('\\')[-1])

    # === 訓練資料(Training)讀取 ===
    with tf.name_scope('TR_Data_Loader'):
        with tf.name_scope('Skeleton_Data'):
            data_files = tf.gfile.Glob(data_path['TR_skeleton'])
            file_queue = tf.train.string_input_producer(data_files, seed=9453)
            # assert len(data_files) == 20, 'The number of training data files must be 20!!'
            info = 'Training skeleton data files number: %d \n  %s\n\n' % (len(data_files), data_files)
            tf.logging.info(info)
            configlogWriter.write(info)
            reader = tf.TFRecordReader()
            _, value = reader.read(file_queue)

            features = tf.parse_single_example(value,
                                               features={
                                                   "subject": tf.FixedLenFeature([], dtype=tf.string),
                                                   "label": tf.FixedLenFeature([], dtype=tf.int64),
                                                   "seq_Diff": tf.FixedLenFeature([], dtype=tf.int64),
                                                   "skeleton_Diff": tf.FixedLenFeature(shape=[], dtype=tf.string)
                                               })
            subject = tf.reshape(tf.decode_raw(features['subject'], tf.uint8),[24])

            label = tf.cast(features['label'], tf.int32)
            label = tf.reshape(label, [1])

            seq = tf.cast(features['seq_Diff'], tf.int32)
            seq = tf.reshape(seq, [1])

            skeleton = tf.decode_raw(features['skeleton_Diff'], tf.float32)
            skeleton = tf.reshape(skeleton, [-1, n_inputs])

        with tf.name_scope('Data_Queue'):
            # 準備開始讀取資料
            RS_queue = tf.RandomShuffleQueue(
                capacity=16 * batch_size,
                min_after_dequeue=8 * batch_size,
                dtypes=[tf.uint8,tf.int32, tf.int32, tf.float32],
                shapes=[[24],[1], [1], [max_seq_len['Skeleton'], n_inputs]])
            # 必須要是1，多線呈會造成資料錯亂
            num_threads = 1

            RS_enqueue_op = RS_queue.enqueue([subject,label, seq, skeleton])

            tf.train.add_queue_runner(
                tf.train.queue_runner.QueueRunner(
                    RS_queue, [RS_enqueue_op] * num_threads)
            )

    # ===測試資料(Eval)讀取===
    with tf.name_scope('EV_Data_Loader'):
        with tf.name_scope('Skeleton_Data'):
            data_files_EV = tf.gfile.Glob(data_path['EV_skeleton'])
            info = 'Test skeleton files number: %d\n %s\n\n' % (len(data_files_EV), data_files_EV)
            tf.logging.info(info)
            configlogWriter.write(info)

            file_queue_EV = tf.train.string_input_producer(data_files_EV, seed=9453)
            reader_EV = tf.TFRecordReader()
            _, value_EV = reader_EV.read(file_queue_EV)

            features_EV = tf.parse_single_example(value_EV,
                                                  features={
                                                      "subject": tf.FixedLenFeature([], dtype=tf.string),
                                                      "label": tf.FixedLenFeature([], dtype=tf.int64),
                                                      "seq_Diff": tf.FixedLenFeature([], dtype=tf.int64),
                                                      "skeleton_Diff": tf.FixedLenFeature([], dtype=tf.string)
                                                  })

            subject_EV = tf.reshape(tf.decode_raw(features_EV['subject'], tf.uint8),[24])

            label_EV = tf.cast(features_EV['label'], tf.int32)
            label_EV = tf.reshape(label_EV, [1])

            seq_EV = tf.cast(features_EV['seq_Diff'], tf.int32)
            seq_EV = tf.reshape(seq_EV, [1])

            skeleton_EV = tf.decode_raw(features_EV['skeleton_Diff'], tf.float32)
            skeleton_EV = tf.reshape(skeleton_EV, [-1, n_inputs])

        with tf.name_scope('Data_Queue'):
            FIFOQueue_queue_EV = tf.FIFOQueue(
                3 * batch_size,
                dtypes=[tf.uint8,tf.int32, tf.int32, tf.float32],
                shapes=[[24],[1], [1], [max_seq_len['Skeleton'], n_inputs]])
            num_threads = 1

            FIFO_enqueue_op_EV = FIFOQueue_queue_EV.enqueue([subject_EV,label_EV, seq_EV, skeleton_EV])

            tf.train.add_queue_runner(
                tf.train.queue_runner.QueueRunner(
                    FIFOQueue_queue_EV, [FIFO_enqueue_op_EV] * num_threads)
            )

    # Queue 資料來源切換器
    with tf.name_scope('Data_Switch'):
        select_q = tf.placeholder_with_default(tf.constant(0), [], name='Data_Selector')
        q = tf.QueueBase.from_list(select_q, [RS_queue, FIFOQueue_queue_EV])
        # q = tf.QueueBase.from_list(0, [RS_queue, ])

        subject, label, seq, skeleton = q.dequeue_many(batch_size, 'Dequeue_Many')

        assert subject.get_shape()[0] == batch_size
        assert label.get_shape()[0] == batch_size
        assert seq.get_shape()[0] == batch_size
        assert skeleton.get_shape()[0] == batch_size

        indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])

        label = tf.reshape(label, [batch_size, 1], 'label_out')

        seq = tf.reshape(seq, [-1], 'seq_out')

        label_onehot = tf.sparse_to_dense(
            tf.concat(values=[indices, label], axis=1),
            [batch_size, num_classes], 1.0, 0.0, name='label_RGB_onehot_out')

    configlogWriter.write(
        'Input Skeleton shapes : \n Label:{},Skeleton:{} '.format(label.get_shape(),
                                                                  skeleton.get_shape()))
    configlogWriter.flush()
    configlogWriter.close()

    return select_q, \
        {"subject": subject, "label_onehot": label_onehot, "label": label, "seq": seq, "skeleton": skeleton}
