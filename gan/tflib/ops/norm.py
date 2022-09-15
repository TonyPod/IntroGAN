# -*- coding:utf-8 -*-  

""" 
@time: 1/10/19 10:29 AM 
@author: Chen He 
@site:  
@file: norm.py
@description:  
"""

import tensorflow as tf


# spectral_norm
def l2_norm(input_x, epsilon=1e-12):
    input_x_norm = input_x / (tf.reduce_sum(input_x**2)**0.5 + epsilon)
    return input_x_norm


def weights_spectral_norm(name, weights, u=None, iteration=1, update_collection=None, reuse=False):

    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        w_shape = weights.get_shape().as_list()
        w_mat = tf.reshape(weights, [-1, w_shape[-1]])
        if u is None:
            u = tf.get_variable('u', shape=[1, w_shape[-1]], initializer=tf.truncated_normal_initializer(),
                                trainable=False)

        def power_iteration(u, ite):
            v_ = tf.matmul(u, tf.transpose(w_mat))
            v_hat = l2_norm(v_)
            u_ = tf.matmul(v_hat, w_mat)
            u_hat = l2_norm(u_)
            return u_hat, v_hat, ite + 1

        u_hat, v_hat, _ = power_iteration(u, iteration)

        sigma = tf.matmul(tf.matmul(v_hat, w_mat), tf.transpose(u_hat))

        w_mat = w_mat / sigma

        if update_collection is None:
            with tf.control_dependencies([u.assign(u_hat)]):
                w_norm = tf.reshape(w_mat, w_shape)
        else:
            if not (update_collection == 'NO_OPS'):
                print(update_collection)
                tf.add_to_collection(update_collection, u.assign(u_hat))

            w_norm = tf.reshape(w_mat, w_shape)
        return w_norm