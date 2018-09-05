import tensorflow as tf


def add_layer(input_, in_size, out_size, n_layer, b_offset=0.6, act_fn=None):

    layer_name = f'layer{n_layer}'
    with tf.name_scope(layer_name):

        with tf.name_scope('weights'):
            weights = tf.Variable(tf.truncated_normal([in_size, out_size]), name='W')
            # weights = tf.Variable(tf.contrib.layers.xavier_initializer)
            tf.summary.histogram(layer_name + '/weights', weights)

        with tf.name_scope('biases'):
            initial_b = tf.zeros([1, out_size]) + b_offset
            biases = tf.Variable(initial_b, dtype=tf.float32, name='b')
            tf.summary.histogram(layer_name + '/biases', biases)

        with tf.name_scope('wx_plus_b'):
            wx_plus_b = tf.add(tf.matmul(input_, weights), biases)

        if act_fn is None:
            outputs = wx_plus_b
        else:
            outputs = act_fn(wx_plus_b, )
        tf.summary.histogram(layer_name + '/outputs', outputs)

    return outputs


def produce_placeholder(node_list):
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, node_list[0]], name='x_input')
        y = tf.placeholder(tf.float32, [None, node_list[-1]], name='y_input')
    return x, y


def deep_1hidden(node_list, b_offset=0.6):
    x, y = produce_placeholder(node_list)

    # add hidden layer
    l1 = add_layer(x, node_list[0], node_list[1], n_layer=1, b_offset=b_offset, act_fn=tf.nn.sigmoid)
    # add output layer
    y_ = add_layer(l1, node_list[1], node_list[2], n_layer='_out', b_offset=b_offset)

    return x, y, y_


def deep_2hidden(node_list, b_offset=0.6):
    x, y = produce_placeholder(node_list)

    # add hidden layer
    l1 = add_layer(x, node_list[0], node_list[1], n_layer=1, b_offset=b_offset, act_fn=tf.nn.relu)
    l2 = add_layer(l1, node_list[1], node_list[2], n_layer=2, b_offset=b_offset, act_fn=tf.nn.relu)
    # add output layer
    y_ = add_layer(l2, node_list[2], node_list[3], n_layer='_out', b_offset=b_offset)

    return x, y, y_


def deep_3hidden(node_list, b_offset=0.6):
    x, y = produce_placeholder(node_list)

    # add hidden layer
    l1 = add_layer(x, node_list[0], node_list[1], n_layer=1, b_offset=b_offset, act_fn=tf.nn.relu)
    l2 = add_layer(l1, node_list[1], node_list[2], n_layer=2, b_offset=b_offset, act_fn=tf.nn.relu)
    l3 = add_layer(l2, node_list[2], node_list[3], n_layer=3, b_offset=b_offset, act_fn=tf.nn.relu)
    # add output layer
    y_ = add_layer(l3, node_list[3], node_list[4], n_layer='_out', b_offset=b_offset)

    return x, y, y_


def deep_4hidden(node_list, b_offset=0.6):
    x, y = produce_placeholder(node_list)

    # add hidden layer
    l1 = add_layer(x, node_list[0], node_list[1], n_layer=1, b_offset=b_offset, act_fn=tf.nn.relu)
    l2 = add_layer(l1, node_list[1], node_list[2], n_layer=2, b_offset=b_offset, act_fn=tf.nn.relu)
    l3 = add_layer(l2, node_list[2], node_list[3], n_layer=3, b_offset=b_offset, act_fn=tf.nn.relu)
    l4 = add_layer(l3, node_list[3], node_list[4], n_layer=4, b_offset=b_offset, act_fn=tf.nn.relu)
    # add output layer
    y_ = add_layer(l4, node_list[4], node_list[5], n_layer='_out', b_offset=b_offset)

    return x, y, y_


def deep_5hidden(node_list, b_offset=0.6):
    x, y = produce_placeholder(node_list)

    # add hidden layer
    l1 = add_layer(x, node_list[0], node_list[1], n_layer=1, b_offset=b_offset, act_fn=tf.nn.relu)
    l2 = add_layer(l1, node_list[1], node_list[2], n_layer=2, b_offset=b_offset, act_fn=tf.nn.relu)
    l3 = add_layer(l2, node_list[2], node_list[3], n_layer=3, b_offset=b_offset, act_fn=tf.nn.relu)
    l4 = add_layer(l3, node_list[3], node_list[4], n_layer=4, b_offset=b_offset, act_fn=tf.nn.relu)
    l5 = add_layer(l4, node_list[4], node_list[5], n_layer=5, b_offset=b_offset, act_fn=tf.nn.relu)
    # add output layer
    y_ = add_layer(l5, node_list[5], node_list[6], n_layer='_out', b_offset=b_offset)

    return x, y, y_


def deep_6hidden(node_list, b_offset=0.6):
    x, y = produce_placeholder(node_list)

    # add hidden layer
    l1 = add_layer(x, node_list[0], node_list[1], n_layer=1, b_offset=b_offset, act_fn=tf.nn.relu)
    l2 = add_layer(l1, node_list[1], node_list[2], n_layer=2, b_offset=b_offset, act_fn=tf.nn.relu)
    l3 = add_layer(l2, node_list[2], node_list[3], n_layer=3, b_offset=b_offset, act_fn=tf.nn.relu)
    l4 = add_layer(l3, node_list[3], node_list[4], n_layer=4, b_offset=b_offset, act_fn=tf.nn.relu)
    l5 = add_layer(l4, node_list[4], node_list[5], n_layer=5, b_offset=b_offset, act_fn=tf.nn.relu)
    l6 = add_layer(l5, node_list[5], node_list[6], n_layer=6, b_offset=b_offset, act_fn=tf.nn.relu)
    # add output layer
    y_ = add_layer(l6, node_list[6], node_list[7], n_layer='_out', b_offset=b_offset)

    return x, y, y_


def model_selector(node_list, b_offset=0.6):
    options = {1: deep_1hidden(node_list, b_offset),
               2: deep_2hidden(node_list, b_offset),
               3: deep_3hidden(node_list, b_offset),
               4: deep_4hidden(node_list, b_offset),
               5: deep_5hidden(node_list, b_offset),
               6: deep_6hidden(node_list, b_offset)
               }
    return options[len(node_list) - 2]
