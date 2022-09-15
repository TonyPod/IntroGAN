# -*- coding:utf-8 -*-  

""" 
@time: 10/27/18 1:37 PM 
@author: Chen He 
@site:  
@file: fashion_mnist.py
@description:  
"""

import gzip
import numpy as np
import os
import pickle

import tensorflow as tf
import tensorflow.contrib.slim as slim

from vgg_preprocessing import preprocess_image
from utils.resnet_v1_mod import resnet_arg_scope
from utils.resnet_v1_mod import resnet_v1_50

NUM_CLASSES = 10


def load_feat(path='datasets/fashion-mnist', kind='train', order_idx=1):
    images, labels, one_hot_labels = load_data(path, kind, order_idx)

    pickle_file = os.path.join(path, '%s_feat_order_%d.pkl' % (kind, order_idx))
    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as fin:
            feats = pickle.load(fin)
    else:
        with tf.Session() as sess:
            real_data = tf.placeholder(tf.float32, shape=[None, 784])
            inputs = tf.map_fn(
                lambda img: preprocess_image(tf.tile(tf.reshape(img, (28, 28, 1)), [1, 1, 3]), 224, 224,
                                             is_training=False), real_data)
            with slim.arg_scope(resnet_arg_scope()):
                _, end_points = resnet_v1_50(inputs, 1000, is_training=False)
                feat_tensor = slim.flatten(end_points['pool5'])

            sess.run(tf.global_variables_initializer())
            var_list = tf.trainable_variables()
            saver = tf.train.Saver(var_list=var_list)
            saver.restore(sess, 'utils/resnet_v1_50.ckpt')

            feats = []
            for i in range(0, len(images), 100):
                feat_batch = sess.run(feat_tensor, feed_dict={real_data: images[i:i+100]})
                feats.extend(feat_batch)
            feats = np.array(feats)

            with open(pickle_file, 'wb') as fout:
                pickle.dump(feats, fout)

    return feats, labels, one_hot_labels


def load_data(path='datasets/fashion-mnist', kind='train', order_idx=1):

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    order = []
    with open(os.path.join(path, 'order_%d.txt' % order_idx)) as file_in:
        for line in file_in.readlines():
            order.append(int(line))
    order = np.array(order)

    labels = change_order(labels, order=order)

    one_hot_labels = np.eye(NUM_CLASSES, dtype=float)[labels]

    return images, labels, one_hot_labels, order


def change_order(cls, order):
    order_dict = dict()
    for i in range(len(order)):
        order_dict[order[i]] = i

    reordered_cls = np.array([order_dict[cls[i]] for i in range(len(cls))])
    return reordered_cls