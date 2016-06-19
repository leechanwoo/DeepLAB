__author__ = 'chanwoo'

from input_data import LoadImageData
from input_data import LoadLabels
from input_data import Reshape

import tensorflow as tf
import cv2 as cv
import numpy as np
import os




def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def training_model(Image, Label):

    W_conv1 = weight_variable([5, 5, 3, 32])
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(ImageBatch, W_conv1) + b_conv1)
    h_conv1 = max_pool_2x2(h_conv1)

    W_fc1 = weight_variable([14 * 14 * 32, 36])
    b_fc1 = bias_variable([36])

    h_conv1_flat = tf.reshape(h_conv1, [-1, 14*14*32])
    y = tf.nn.softmax(tf.matmul(h_conv1_flat, W_fc1) + b_fc1)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(LabelBatch * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(LabelBatch, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return train_step, accuracy

DATA_NUM = 500
IMAGE_SIZE = 28


Images, ImageShape = LoadImageData(DATA_NUM, IMAGE_SIZE)
Labels, LabelShape = LoadLabels(DATA_NUM)

Batch = tf.train.shuffle_batch([Images, Labels], 50, 3*50*100, 100, enqueue_many=True)
ImageBatch = tf.cast(Batch[0], tf.float32)
LabelBatch = tf.cast(Batch[1], tf.float32)

Training = training_model(ImageBatch, LabelBatch)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess, coord)
    print 'start queue runner'
    i = 0
    try:
        while not coord.should_stop():
            [train_step, accuracy] = sess.run(Training)

            if i % 100 == 0:
                print "step %d, accuracy %f" % (i, accuracy)
            i += 1

            if accuracy > 0.5:
                break

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')

    finally:
        coord.request_stop()
        coord.join(thread)

    coord.request_stop()
    coord.join(thread)




