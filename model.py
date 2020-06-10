'''
Convolutional Neural Network for Fashion Landmarks Detection.
Daniel E 201190945 for COMP592 Project due 20th of April
Download tfrecords files from https://1drv.ms/u/s!Au5dCZ4YubPGg4giMoXai4H2lLnBRA?e=fHdS9u
'''

import tensorflow.compat.v1 as tf

class LandmarkModel(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, input_tensor):
        # |== Layer 0: input layer ==|
        # Input feature x should be of shape (batch_size, image_width, image_height, color_channels). 
        inputs = tf.cast(input_tensor, tf.float32)

        # |== Layer 1 ==|
        with tf.variable_scope('layer1'):
            # Convolutional layer
            # Computes 32 features using a 3x3 filter with ReLU activation.
            conv1 = tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=[3, 3], strides=(1, 1), padding='SAME', activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2],  strides=(2, 2), padding='SAME')

        # |== Layer 2 ==|
        with tf.variable_scope('layer2'):
            conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3], strides=(1, 1), padding='SAME', activation=tf.nn.relu)
            conv3 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=[3, 3], strides=(1, 1), padding='SAME', activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=(2, 2), padding='SAME')

        # |== Layer 3 ==|
        with tf.variable_scope('layer3'):
            conv4 = tf.layers.conv2d(inputs=pool2, filters=64, kernel_size=[3, 3], strides=(1, 1), padding='SAME', activation=tf.nn.relu)
            conv5 = tf.layers.conv2d(inputs=conv4, filters=64, kernel_size=[3, 3], strides=(1, 1), padding='SAME', activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=(2, 2), padding='SAME')

        # |== Layer 4 ==|
        with tf.variable_scope('layer4'):
            conv6 = tf.layers.conv2d(inputs=pool3, filters=128, kernel_size=[3, 3], strides=(1, 1), padding='SAME', activation=tf.nn.relu)
            conv7 = tf.layers.conv2d(inputs=conv6, filters=128, kernel_size=[3, 3], strides=(1, 1), padding='SAME', activation=tf.nn.relu)
            pool4 = tf.layers.max_pooling2d(inputs=conv7, pool_size=[2, 2], strides=(1, 1), padding='SAME')

        # |== Layer 5 ==|
        with tf.variable_scope('layer5'):
            conv8 = tf.layers.conv2d(inputs=pool4, filters=256, kernel_size=[3, 3], strides=(1, 1), padding='SAME', activation=tf.nn.relu)

        # |== Layer 6 ==|
        with tf.variable_scope('layer6'):
            # Flatten tensor into a batch of vectors
            flatten = tf.layers.flatten(inputs=conv8)
            dense1 = tf.layers.dense(inputs=flatten, units=1024, activation=tf.nn.relu, use_bias=True)
            logits = tf.layers.dense(inputs=dense1, units=self.output_size, activation=None, use_bias=True, name="logits")
            logits = tf.identity(logits, 'final_dense')

        return logits
