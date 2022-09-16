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
from scipy.io import loadmat
import pickle

NUM_CLASSES = 10


def rgb2gray(rgb):
    r, g, b = rgb[0, :, :], rgb[1, :, :], rgb[2, :, :]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return np.repeat(np.expand_dims(np.uint8(gray), axis=0), 3, axis=0)


def load_data(path='datasets/svhn', kind='train', order_idx=1, rgb=True, num_samples_per_class=-1):
    """Load SVHN data from `path`"""

    file_path = os.path.join(path, '%s_32x32.mat' % kind)
    file_data = loadmat(file_path)

    labels = np.squeeze(file_data['y']) - 1
    images = np.transpose(file_data['X'], (3, 2, 0, 1))
    if not rgb:
        images = np.array(map(rgb2gray, images))
    images = np.reshape(images, (-1, 32 * 32 * 3))

    order = []
    with open(os.path.join(path, 'order_%d.txt' % order_idx)) as file_in:
        for line in file_in.readlines():
            order.append(int(line))
    order = np.array(order)

    labels = change_order(labels, order=order)

    one_hot_labels = np.eye(NUM_CLASSES, dtype=float)[labels]

    if kind == 'train' and num_samples_per_class > 0:
        indices_chosen = []
        for class_idx in order:
            indices_chosen_cur_cls = np.random.choice(np.where(labels == class_idx)[0], num_samples_per_class,
                                                      replace=False)
            indices_chosen.extend(indices_chosen_cur_cls)
        images, labels, one_hot_labels = images[indices_chosen], labels[indices_chosen], one_hot_labels[indices_chosen]

    return images, labels, one_hot_labels, order


def change_order(cls, order):
    order_dict = dict()
    for i in range(len(order)):
        order_dict[order[i]] = i

    reordered_cls = np.array([order_dict[cls[i]] for i in range(len(cls))])
    return reordered_cls
