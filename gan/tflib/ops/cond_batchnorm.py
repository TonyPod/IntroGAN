import numpy as np
import tensorflow as tf

from gan import tflib as lib


def Cond_Batchnorm(name, axes, inputs, is_training=None, stats_iter=None, update_moving_stats=True, fused=True, labels=None, n_labels=None):
    """conditional batchnorm (dumoulin et al 2016) for BCHW conv filtermaps"""
    if axes != [0,2,3]:
        raise Exception('unsupported')
    mean, var = tf.nn.moments(inputs, axes, keep_dims=True)
    shape = mean.get_shape().as_list() # shape is [1,n,1,1]
    offset_m = lib.param(name+'.offset', np.zeros([n_labels, shape[1]], dtype='float32'))
    scale_m = lib.param(name+'.scale', np.ones([n_labels, shape[1]], dtype='float32'))

    # check if labels is one hot
    if len(labels.shape) == 2:
        labels = tf.argmax(labels, axis=1)

    offset = tf.nn.embedding_lookup(offset_m, labels)   # embedding_lookup is a indexing operation
    scale = tf.nn.embedding_lookup(scale_m, labels)
    result = tf.nn.batch_normalization(inputs, mean, var, offset[:,:,None,None], scale[:,:,None,None], 1e-5)
    return result