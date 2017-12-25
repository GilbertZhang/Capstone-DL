import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

from GoogleNet import *
import tempfile


def CNN(x, num_classes=5):
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 256, 256, 1])
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 3, 8])
        b_conv1 = bias_variable([8])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 8, 16])
        b_conv2 = bias_variable([16])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('conv3'):
        W_conv3 = weight_variable([5, 5, 16, 32])
        b_conv3 = bias_variable([32])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

    with tf.name_scope('pool3'):
        h_pool3 = max_pool_2x2(h_conv3)

    with tf.name_scope('conv4'):
        W_conv4 = weight_variable([5, 5, 32, 48])
        b_conv4 = bias_variable([48])
        h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)

    with tf.name_scope('pool4'):
        h_pool4 = max_pool_2x2(h_conv4)

    with tf.name_scope('conv4'):
        W_conv5 = weight_variable([5, 5, 48, 64])
        b_conv5 = bias_variable([64])
        h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)

    with tf.name_scope('pool5'):
        h_pool5 = max_pool_2x2(h_conv5)

    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([8 * 8 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool5, [-1, 8 * 8 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, num_classes])
        b_fc2 = bias_variable([num_classes])

        y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
    return y_conv


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

FLAGS = None
mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

"""Parameter"""
input_size = 784
output_class = 5

"""placeholder"""
x = tf.placeholder(tf.float32, [None, input_size])
# y_conv = CNN(x, num_classes=output_class)
y_conv, dict, keep_prob = googlenet(x, num_classes=output_class)
y_ = tf.placeholder(tf.float32, [None, output_class])

""" loss """
with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

graph_location = tempfile.mkdtemp()
print('Saving graph to: %s' % graph_location)
train_writer = tf.summary.FileWriter(graph_location)
train_writer.add_graph(tf.get_default_graph())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
      batch = mnist.train.next_batch(50)
      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob:1.0}))
