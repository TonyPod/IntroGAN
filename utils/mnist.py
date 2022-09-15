# -*- coding:utf-8 -*-  

""" 
@time: 10/27/18 1:37 PM 
@author: Chen He 
@site:  
@file: mnist.py
@description:  
"""
import gzip
import os

import numpy as np

NUM_CLASSES = 10


def load_data(path='datasets/mnist', kind='train', order_idx=1):
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
