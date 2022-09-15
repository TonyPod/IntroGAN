import collections
import os
import pickle
import shutil
import time
from collections import Counter

import matplotlib as mpl
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm

import gan.tflib as lib
import gan.tflib.inception_score
import gan.tflib.ops
import gan.tflib.ops.batchnorm
import gan.tflib.ops.conv2d
import gan.tflib.ops.deconv2d
import gan.tflib.ops.linear
import gan.tflib.plot
import gan.tflib.save_images
from utils.fid import calculate_fid_given_paths_with_sess

mpl.use('Agg')
import matplotlib.pyplot as plt

from utils.visualize_embedding_protos_and_samples import colors
from sklearn.manifold import TSNE
from umap import UMAP


def leaky_relu(x, alpha=0.2):
    return tf.maximum(alpha * x, x)


def conv_cond_concat(x, y):
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat([x, y * tf.ones([x_shapes[0], y_shapes[1], x_shapes[2], x_shapes[3]])], 1)


class Memory(object):

    def __init__(self):
        self.protos = collections.OrderedDict()

    def add_protos(self, category, protos):
        self.protos[category] = protos

    def get_protos(self, category):
        return self.protos[category]

    def has_protos(self, category):
        return self.protos.has_key(category)

    def as_ndarray(self):
        return np.array(self.protos.values())


class IntroGAN(object):

    def __init__(self, sess, graph, sess_fid, dataset, mode, batch_size, output_dim,
                 lambda_param, critic_iters, iters, result_dir, checkpoint_interval,
                 adam_lr, adam_beta1, adam_beta2, finetune, improved_finetune, nb_cl, nb_output, protogan_scale,
                 protogan_scale_g,
                 classification_only, proto_weight_real, proto_weight_fake, dist_func, gamma, proto_num, margin,
                 order_idx, order, test_interval, proto_select_criterion, proto_importance,
                 improved_finetune_type, improved_finetune_noise,
                 improved_finetune_noise_level, center_type, fixed_center_idx, num_samples_per_class,
                 exemplars_dual_use, anti_imbalance, train_rel_center, rigorous):

        self.sess = sess
        self.graph = graph
        self.sess_fid = sess_fid

        self.dataset = dataset
        self.mode = mode
        self.batch_size = batch_size
        self.output_dim = output_dim

        self.lambda_param = lambda_param
        self.critic_iters = critic_iters

        self.iters = iters
        self.result_dir = result_dir
        self.save_interval = checkpoint_interval

        self.adam_lr = adam_lr
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2

        self.protogan_scale = protogan_scale
        self.protogan_scale_g = protogan_scale_g

        self.finetune = finetune
        self.improved_finetune = improved_finetune
        self.improved_finetune_type = improved_finetune_type
        self.improved_finetune_noise = improved_finetune_noise
        self.improved_finetune_noise_level = improved_finetune_noise_level

        self.nb_cl = nb_cl
        self.nb_output = nb_output

        self.order_idx = order_idx
        self.order = order

        self.margin = margin

        self.classification_only = classification_only

        self.proto_weight_real = proto_weight_real
        self.proto_weight_fake = proto_weight_fake

        self.dist_func = dist_func
        self.gamma = gamma

        self.proto_num = proto_num
        self.proto_select_criterion = proto_select_criterion
        self.proto_importance = proto_importance
        self.center_type = center_type
        self.fixed_center_idx = fixed_center_idx
        self.train_rel_center = train_rel_center

        self.is_first_session = (self.nb_output == self.nb_cl)

        self.num_samples_per_class = num_samples_per_class
        self.exemplars_dual_use = exemplars_dual_use

        self.anti_imbalance = anti_imbalance
        self.rigorous = rigorous

        if self.is_first_session:
            self.memory = Memory()
        else:
            self.memory = self.load_memory(self.nb_output - self.nb_cl - 1)

        self.test_interval = test_interval

        self.build_model()

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def build_model(self):

        lib.delete_all_params()

        with tf.variable_scope("gan") as scope, self.graph.as_default():
            # placeholder for MNIST samples
            self.real_data_int = tf.placeholder(tf.int32, shape=[self.batch_size, self.output_dim])
            self.real_y = tf.placeholder(tf.float32, shape=[self.batch_size, self.nb_output])
            self.gen_y = tf.placeholder(tf.float32, shape=[None, self.nb_output])
            self.sample_y = tf.placeholder(tf.float32, shape=[None, self.nb_output])
            self.real_weight = tf.placeholder(tf.float32, shape=[self.batch_size])

            self.protos = tf.Variable(tf.random_normal([self.nb_output, self.proto_num, self.output_dim]),
                                      name='Prototype', trainable=False)

            real_data = 2 * ((tf.cast(self.real_data_int, tf.float32) / 255.) - .5)
            fake_data, fake_labels = self.generator(self.batch_size, self.gen_y)

            # set_shape to facilitate concatenation of label and image
            fake_data.set_shape([self.batch_size, fake_data.shape[1].value])
            fake_labels.set_shape([self.batch_size, fake_labels.shape[1].value])

            disc_real, embedding_real_class = self.discriminator(real_data, reuse=False, is_training=True)
            disc_fake, embedding_fake_class = self.discriminator(fake_data, reuse=True, is_training=True)
            self.embedding_protos = tf.map_fn(lambda x: (self.discriminator(x, reuse=True, is_training=True)[1]),
                                              self.protos)
            if self.center_type == 'rel_center':
                self.embedding_protos_center = tf.reduce_mean(self.embedding_protos, axis=1)
            elif self.center_type == 'fixed_center':
                self.embedding_protos_center = self.embedding_protos[:, self.fixed_center_idx, :]
            elif self.center_type == 'multi_center':
                self.embedding_protos_center = self.embedding_protos  # NOTICE: implemented this way but has no meaning!!

            # Get output label
            self.inputs_int = tf.placeholder(tf.int32, shape=[None, self.output_dim])
            inputs = 2 * ((tf.cast(self.inputs_int, tf.float32) / 255.) - .5)
            _, self.embedding_inputs = self.discriminator(inputs, reuse=True, is_training=False)

            random_int = tf.random_uniform([1], maxval=self.proto_num, dtype=tf.int32)

            def random_select(data, axis):

                return tf.squeeze(tf.gather(data, random_int, axis=axis), axis=axis)

            if self.rigorous == 'max':
                sel_func = tf.reduce_max
            elif self.rigorous == 'min':
                sel_func = tf.reduce_min
            elif self.rigorous == 'mean':
                sel_func = tf.reduce_mean
            elif self.rigorous == 'random':
                sel_func = random_select

            def get_logits(inputs, protos, center_type):
                if center_type == 'multi_center':
                    if self.dist_func == 'cosine':
                        classOutput = sel_func(tf.einsum('ab,dcb->acd',
                                                         tf.nn.l2_normalize(inputs, dim=-1),
                                                         tf.nn.l2_normalize(protos, dim=-1)), axis=1)
                    elif self.dist_func == 'squared_l2':
                        classOutput = \
                            -tf.transpose(sel_func(tf.reduce_sum(tf.square(
                                tf.tile(tf.expand_dims(tf.expand_dims(inputs, 0), 0),
                                        [self.nb_output, self.proto_num, 1, 1]) - tf.tile(tf.expand_dims(protos, 2),
                                                                                          [1, 1, tf.shape(inputs)[0],
                                                                                           1])),
                                axis=3), axis=1), [1, 0])
                else:
                    if self.dist_func == 'cosine':
                        classOutput = tf.einsum('ab,cb->ac',
                                                tf.nn.l2_normalize(inputs, dim=-1),
                                                tf.nn.l2_normalize(protos, dim=-1))
                    elif self.dist_func == 'squared_l2':
                        classOutput = \
                            -tf.transpose(tf.reduce_sum(tf.square(
                                tf.tile(tf.expand_dims(inputs, 0),
                                        [self.nb_output, 1, 1]) - tf.tile(tf.expand_dims(protos, 1),
                                                                          [1, tf.shape(inputs)[0], 1])),
                                axis=-1), [1, 0])
                classOutput *= self.gamma
                return classOutput

            self.input_logits = get_logits(self.embedding_inputs, self.embedding_protos_center,
                                           center_type=self.center_type)
            if self.train_rel_center:
                self.real_logits = get_logits(embedding_real_class, self.embedding_protos_center,
                                              center_type=self.center_type)
                fake_logits = get_logits(embedding_fake_class, self.embedding_protos_center,
                                         center_type=self.center_type)
            else:
                self.real_logits = get_logits(embedding_real_class, self.embedding_protos,
                                              center_type='multi_center')
                fake_logits = get_logits(embedding_fake_class, self.embedding_protos, center_type='multi_center')

            self.pred_y = tf.argmax(self.input_logits, axis=1)

            self.real_accu = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(self.real_logits, axis=1), tf.argmax(self.real_y, 1)), dtype=tf.float32))
            self.fake_accu = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(fake_logits, axis=1), tf.argmax(fake_labels, 1)), dtype=tf.float32))

            self.real_class_cost = tf.reduce_mean(self.real_weight *
                                                  tf.nn.softmax_cross_entropy_with_logits(
                                                      logits=self.real_logits - self.real_y * self.margin,
                                                      labels=self.real_y))
            self.gen_class_cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=fake_logits - fake_labels * self.margin,
                                                        labels=fake_labels))

            gen_params = [var for var in tf.trainable_variables() if 'Generator' in var.name]
            disc_params = [var for var in tf.trainable_variables() if 'Discriminator' in var.name]

            if self.mode == 'wgan-gp':
                # Standard WGAN loss
                self.gen_cost = -tf.reduce_mean(disc_fake)
                self.disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real * self.real_weight)

                # Gradient penalty
                alpha = tf.random_uniform(
                    shape=[self.batch_size, 1],
                    minval=0.,
                    maxval=1.
                )
                differences = fake_data - real_data
                interpolates = real_data + (alpha * differences)
                gradients = tf.gradients(self.discriminator(interpolates, reuse=True, is_training=True),
                                         [interpolates])[0]
                slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
                gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
                self.disc_cost += self.lambda_param * gradient_penalty
            elif self.mode == 'dcgan':
                # Vanilla / Non-saturating loss
                self.gen_cost = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.ones_like(disc_fake)))
                self.disc_cost = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.zeros_like(disc_fake)))
                self.disc_cost += tf.reduce_mean(self.real_weight *
                                                 tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real,
                                                                                         labels=tf.ones_like(
                                                                                             disc_real)))
                self.disc_cost /= 2.

            if self.classification_only:
                self.disc_cost = self.protogan_scale * self.real_class_cost
            else:
                self.disc_cost += self.protogan_scale * self.real_class_cost
            self.gen_cost += self.protogan_scale_g * self.gen_class_cost

            # fake proto loss
            if self.center_type == 'multi_center':
                self.proto_loss_real = tf.zeros(1)
                self.proto_loss_fake = tf.zeros(1)
            else:
                if self.dist_func == 'cosine':
                    self.proto_loss_fake = -tf.reduce_mean(tf.multiply(
                        tf.nn.l2_normalize(tf.gather(self.embedding_protos_center, tf.argmax(fake_labels, axis=1)),
                                           dim=-1), tf.nn.l2_normalize(embedding_fake_class, dim=-1)))

                elif self.dist_func == 'squared_l2':
                    self.proto_loss_fake = tf.reduce_mean(tf.reduce_sum(tf.square(
                        tf.gather(self.embedding_protos_center, tf.argmax(fake_labels, axis=1)) - embedding_fake_class),
                        axis=1))

                # real proto loss
                if self.dist_func == 'cosine':
                    self.proto_loss_real = -tf.reduce_mean(tf.multiply(
                        tf.nn.l2_normalize(tf.gather(self.embedding_protos_center, tf.argmax(self.real_y, axis=1)),
                                           dim=-1), tf.nn.l2_normalize(embedding_real_class, dim=-1)))

                elif self.dist_func == 'squared_l2':
                    self.proto_loss_real = tf.reduce_mean(tf.reduce_sum(tf.square(
                        tf.gather(self.embedding_protos_center, tf.argmax(self.real_y, axis=1)) - embedding_real_class),
                        axis=1))

            self.gen_cost += self.proto_weight_fake * self.proto_loss_fake
            self.disc_cost += self.proto_weight_real * self.proto_loss_real

            if self.classification_only:
                self.gen_train_op = \
                    tf.train.AdamOptimizer(learning_rate=0,
                                           beta1=self.adam_beta1, beta2=self.adam_beta2) \
                        .minimize(self.gen_cost, var_list=gen_params)
                self.disc_train_op = \
                    tf.train.AdamOptimizer(learning_rate=self.adam_lr,
                                           beta1=self.adam_beta1, beta2=self.adam_beta2) \
                        .minimize(self.disc_cost, var_list=disc_params)
            else:
                self.gen_train_op = \
                    tf.train.AdamOptimizer(learning_rate=self.adam_lr,
                                           beta1=self.adam_beta1, beta2=self.adam_beta2) \
                        .minimize(self.gen_cost, var_list=gen_params)
                self.disc_train_op = \
                    tf.train.AdamOptimizer(learning_rate=self.adam_lr,
                                           beta1=self.adam_beta1, beta2=self.adam_beta2) \
                        .minimize(self.disc_cost, var_list=disc_params)

            # For generating samples
            fixed_noise_128 = tf.constant(np.random.normal(size=(128, 100)).astype('float32'))
            self.fixed_noise_samples_128 = self.sampler(128, self.sample_y, noise=fixed_noise_128)[0]

            # For calculating inception score
            self.test_noise = tf.random_normal([self.batch_size, 100])
            self.test_samples = self.sampler(self.batch_size, self.sample_y, noise=self.test_noise)[0]

            var_list = [var for var in tf.trainable_variables() if 'Prototype' not in var.name]

            bn_moving_vars = [var for var in tf.global_variables() if 'moving_mean' in var.name]
            bn_moving_vars += [var for var in tf.global_variables() if 'moving_variance' in var.name]
            var_list += bn_moving_vars

            self.saver = tf.train.Saver(var_list=var_list)  # var_list doesn't contain Adam params

            if self.finetune:
                if self.improved_finetune:
                    var_list_for_finetune = [var for var in var_list if 'g_Input.W' not in var.name]
                else:
                    var_list_for_finetune = [var for var in var_list if 'g_Input' not in var.name]
                self.saver_for_finetune = tf.train.Saver(var_list=var_list_for_finetune)

    def gen_labels(self, nb_samples, condition=None):
        labels = np.zeros([nb_samples, self.nb_output], dtype=np.float32)
        for i in range(nb_samples):
            if condition is not None:  # random or not
                label_item = condition
            else:
                label_item = np.random.randint(0, self.nb_output)
            labels[i, label_item] = 1  # one hot label
        return labels

    def generate_image(self, frame, train_log_dir_for_cur_class):
        # different y, same x
        for category_idx in range(self.nb_output):
            y = self.gen_labels(128, category_idx)
            samples = self.sess.run(self.fixed_noise_samples_128, feed_dict={self.sample_y: y})
            samples = ((samples + 1.) * (255. / 2)).astype('int32')

            embedding_samples = self.sess.run(self.embedding_inputs, feed_dict={self.inputs_int: samples})

            samples_folder = os.path.join(train_log_dir_for_cur_class, 'samples', 'class_%d' % (category_idx + 1))
            if not os.path.exists(samples_folder):
                os.makedirs(samples_folder)
            gan.tflib.save_images.save_images(samples.reshape((128, 1, 28, 28)),
                                              os.path.join(samples_folder,
                                                           'samples_{}.jpg'.format(frame)))
            # dump samples for visualization etc.
            with open(os.path.join(samples_folder, 'samples_{}.pkl'.format(frame)), 'wb') as fout:
                pickle.dump(samples, fout)

            with open(os.path.join(samples_folder, 'embedding_samples_{}.pkl'.format(frame)), 'wb') as fout:
                pickle.dump(embedding_samples, fout)

    def show_prototypes(self, frame, train_log_dir_for_cur_class):
        all_protos = self.sess.run(self.protos)
        all_protos = ((all_protos + 1.) * (255. / 2)).astype('int32')

        # embedding
        all_embedding_protos = self.sess.run(self.embedding_protos)

        for category_idx in range(self.nb_output):
            proto_folder = os.path.join(train_log_dir_for_cur_class, 'protos', 'class_%d' % (category_idx + 1))
            if not os.path.exists(proto_folder):
                os.makedirs(proto_folder)

            protos = all_protos[category_idx]
            embedding_protos = all_embedding_protos[category_idx]
            gan.tflib.save_images.save_images(protos.reshape((protos.shape[0], 1, 28, 28)),
                                              os.path.join(proto_folder,
                                                           'protos_{}.jpg'.format(frame)))

            with open(os.path.join(proto_folder, 'protos_{}.pkl'.format(frame)), 'wb') as fout:
                pickle.dump(protos, fout)

            with open(os.path.join(proto_folder, 'embedding_protos_{}.pkl'.format(frame)), 'wb') as fout:
                pickle.dump(embedding_protos, fout)

    def get_fid(self, train_log_dir_for_cur_class):
        print('Calculating fid...')
        time_start = time.time()
        FID_NUM = 10000
        fid_vals = {}

        temp_folder = os.path.join(train_log_dir_for_cur_class, 'fid_temp')

        for category_idx in range(self.nb_output):
            sub_folder = os.path.join(temp_folder, 'class_%d' % (category_idx + 1))
            if not os.path.exists(sub_folder):
                os.makedirs(sub_folder)

            if len(os.listdir(sub_folder)) >= FID_NUM:
                print('Skipping: %d' % (category_idx + 1))
                continue

            x = []
            y = np.zeros([FID_NUM, self.nb_output])
            y[:, category_idx] = np.ones(FID_NUM)
            for i in tqdm(range(0, FID_NUM, self.batch_size), desc='Generation %d' % (category_idx + 1)):
                y_batch = y[i:i + self.batch_size]
                x_batch, _, _ = self.test(len(y_batch), y_batch)
                x.extend(x_batch)
            # print('Generation %d: %.2f seconds' % (category_idx + 1, time.time() - time_start))
            for i, x_single in enumerate(x):
                img = Image.fromarray(
                    np.repeat(x_single.astype('uint8').reshape((1, 28, 28)).transpose((1, 2, 0)), 3, axis=2))
                img.save(os.path.join(sub_folder, '%d.jpg' % (i + 1)))

        for category_idx in range(self.nb_output):
            sub_folder = os.path.join(temp_folder, 'class_%d' % (category_idx + 1))
            fid_val = calculate_fid_given_paths_with_sess(self.sess_fid,
                                                          [sub_folder,
                                                           'precalc_fids/%s/fid_stats_%d.npz' % (
                                                               self.dataset, self.order[category_idx] + 1)])
            fid_vals[category_idx + 1] = fid_val

        # delete temp folders
        shutil.rmtree(temp_folder)

        time_stop = time.time()
        print('Total: %.2f seconds' % (time_stop - time_start))

        return fid_vals

    def test(self, n_samples, y):
        assert n_samples > 0
        if n_samples < self.batch_size:  # padding to make the number the same as self.batch_size
            y = np.concatenate((y, np.zeros([self.batch_size - n_samples, self.nb_output])))

        with self.graph.as_default():
            samples, z = self.sess.run([self.test_samples, self.test_noise], feed_dict={self.sample_y: y})

        if n_samples < self.batch_size:
            samples = samples[:n_samples]
            z = z[:n_samples]

        samples_int = ((samples + 1.) * (255. / 2)).astype('int32')
        return samples_int, samples, z

    def generator(self, n_samples, y, noise=None):

        with tf.variable_scope('Generator') as scope:

            if noise is None:
                noise = tf.random_normal([n_samples, 100])

            # if self.nb_output > 1:
            #     z = tf.concat([noise, y], axis=1)
            # else:
            #     z = noise
            z = tf.concat([noise, y], axis=1)

            output = gan.tflib.ops.linear.Linear('g_Input', z.shape[1].value, 4 * 4 * 4 * 64, z)

            if self.mode == 'wgan':
                output = lib.ops.batchnorm.Batchnorm('g_bn1', [0], output)
            output = tf.nn.relu(output)
            output = tf.reshape(output, [-1, 4 * 64, 4, 4])

            output = lib.ops.deconv2d.Deconv2D('g_2', 4 * 64, 2 * 64, 5, output)
            if self.mode == 'wgan':
                output = lib.ops.batchnorm.Batchnorm('g_bn2', [0, 2, 3], output)
            output = tf.nn.relu(output)

            output = output[:, :, :7, :7]

            output = lib.ops.deconv2d.Deconv2D('g_3', 2 * 64, 64, 5, output)
            if self.mode == 'wgan':
                output = lib.ops.batchnorm.Batchnorm('g_bn3', [0, 2, 3], output)
            output = tf.nn.relu(output)

            output = lib.ops.deconv2d.Deconv2D('g_5', 64, 1, 5, output)
            output = tf.tanh(output)

            return tf.reshape(output, [-1, self.output_dim]), y

    def sampler(self, n_samples, y, noise=None):

        with tf.variable_scope('Generator') as scope:
            scope.reuse_variables()

            if noise is None:
                noise = tf.random_normal([n_samples, 100])

            # if self.nb_output > 1:
            #     z = tf.concat([noise, y], axis=1)
            # else:
            #     z = noise
            z = tf.concat([noise, y], axis=1)

            output = gan.tflib.ops.linear.Linear('g_Input', z.shape[1].value, 4 * 4 * 4 * 64, z)

            if self.mode == 'wgan':
                output = lib.ops.batchnorm.Batchnorm('g_bn1', [0], output)
            output = tf.nn.relu(output)
            output = tf.reshape(output, [-1, 4 * 64, 4, 4])

            output = lib.ops.deconv2d.Deconv2D('g_2', 4 * 64, 2 * 64, 5, output)
            if self.mode == 'wgan':
                output = lib.ops.batchnorm.Batchnorm('g_bn2', [0, 2, 3], output)
            output = tf.nn.relu(output)

            output = output[:, :, :7, :7]

            output = lib.ops.deconv2d.Deconv2D('g_3', 2 * 64, 64, 5, output)
            if self.mode == 'wgan':
                output = lib.ops.batchnorm.Batchnorm('g_bn3', [0, 2, 3], output)
            output = tf.nn.relu(output)

            output = lib.ops.deconv2d.Deconv2D('g_5', 64, 1, 5, output)
            output = tf.tanh(output)

            return tf.reshape(output, [-1, self.output_dim]), y

    def discriminator(self, inputs, reuse, is_training):

        with tf.variable_scope('Discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            output = tf.reshape(inputs, [-1, 1, 28, 28])

            output = lib.ops.conv2d.Conv2D('d_1', 1, 64, 5, output, stride=2)
            output = leaky_relu(output)

            output = lib.ops.conv2d.Conv2D('d_2', 64, 2 * 64, 5, output, stride=2)
            if self.mode == 'wgan':
                output = lib.ops.batchnorm.Batchnorm('d_bn2', [0, 2, 3], output)
            output = leaky_relu(output)

            output = lib.ops.conv2d.Conv2D('d_3', 2 * 64, 4 * 64, 5, output, stride=2)
            if self.mode == 'wgan':
                output = lib.ops.batchnorm.Batchnorm('d_bn3', [0, 2, 3], output)

            output = tf.reshape(output, [-1, 4 * 4 * 4 * 64])
            embeddingOutput = output

            output = leaky_relu(output)

            sourceOutput = gan.tflib.ops.linear.Linear('d_SourceOutput', 4 * 4 * 4 * 64, 1, output)

            return tf.reshape(sourceOutput, shape=[-1]), embeddingOutput

    def classify(self, inputs):

        with self.graph.as_default():
            result = self.sess.run(self.pred_y, feed_dict={self.inputs_int: inputs})

        return result

    def classify_for_logits(self, inputs):

        with self.graph.as_default():
            logits = self.sess.run(self.input_logits, feed_dict={self.inputs_int: inputs})

        return logits

    def train(self, data_X, data_y, test_X, test_y, category_idx, model_exist=False):

        train_log_dir_for_cur_class = self.model_dir_for_class(category_idx)

        if not os.path.exists(train_log_dir_for_cur_class):
            os.makedirs(train_log_dir_for_cur_class)

        if model_exist:
            with self.graph.as_default():
                self.sess.run(tf.initialize_all_variables())
                self.load(category_idx)
        else:
            def get_test_epoch(test_X, test_y):
                reorder = np.array(range(len(test_X)))
                np.random.shuffle(reorder)
                test_X = test_X[reorder]
                test_y = test_y[reorder]

                for j in range(len(test_X) / self.batch_size):
                    yield (test_X[j * self.batch_size:(j + 1) * self.batch_size],
                           test_y[j * self.batch_size:(j + 1) * self.batch_size])

            def get_train_inf(data_X, data_y, weights):
                while True:
                    if self.anti_imbalance == 'oversample':
                        from imblearn.over_sampling import RandomOverSampler
                        _, _, indices = RandomOverSampler(return_indices=True).fit_resample(data_X,
                                                                                            np.argmax(data_y, axis=1))
                        data_X = data_X[indices]
                        data_y = data_y[indices]
                        weights = weights[indices]

                    batch_idxs = len(data_X) // self.batch_size
                    if batch_idxs == 0:
                        idx = np.random.choice(len(data_X), self.batch_size, replace=True)
                        yield (data_X[idx], data_y[idx], weights[idx])
                    else:
                        reorder = np.array(range(len(data_X)))
                        np.random.shuffle(reorder)
                        data_X = data_X[reorder]
                        data_y = data_y[reorder]
                        weights = weights[reorder]
                        for idx in range(0, batch_idxs):
                            _data_X = data_X[idx * self.batch_size:(idx + 1) * self.batch_size]
                            _data_y = data_y[idx * self.batch_size:(idx + 1) * self.batch_size]
                            _weights = weights[idx * self.batch_size:(idx + 1) * self.batch_size]
                            yield (_data_X, _data_y, _weights)

            with self.graph.as_default():
                # Train loop
                self.sess.run(
                    tf.initialize_all_variables())  # saver's var_list contains 28 variables, tf.all_variables() contains more(plus Adam params)
                if self.finetune and not self.is_first_session:
                    _, _, ckpt_path = self.load_finetune(category_idx - self.nb_cl)

                    # special initialized for g_Input
                    if self.improved_finetune:
                        # calc confusion matrix on the training set (only new classes): new classes -> old classes
                        pred_y = []
                        indices_new_classes = np.argmax(data_y, axis=1) >= category_idx - self.nb_cl + 1
                        data_X_new_classes = data_X[indices_new_classes]
                        data_y_new_classes = data_y[indices_new_classes]
                        for sample_idx in range(0, len(data_X_new_classes), self.batch_size):
                            pred_logits_batch = self.classify_for_logits(
                                data_X_new_classes[sample_idx:sample_idx + self.batch_size])
                            pred_y_batch = np.argmax(pred_logits_batch[:, :category_idx - self.nb_cl + 1], axis=1)
                            pred_y.extend(pred_y_batch)
                        pred_y = np.array(pred_y)

                        # get the input tensor
                        ckpt_reader = tf.pywrap_tensorflow.NewCheckpointReader(ckpt_path)
                        input_tensor_names = [key for key in ckpt_reader.get_variable_to_shape_map().keys() if
                                              'g_Input.W' in key]
                        assert len(input_tensor_names) == 1
                        g_Input_offset = 100

                        input_tensor_name = input_tensor_names[0]
                        input_tensor = ckpt_reader.get_tensor(input_tensor_name)

                        for new_category_idx in range(category_idx - self.nb_cl + 1, category_idx + 1):
                            most_confused_with = \
                                Counter(pred_y[np.argmax(data_y_new_classes, axis=1) == new_category_idx]).most_common(
                                    1)[
                                    0][0]
                            tmp_tensor = np.expand_dims(input_tensor[g_Input_offset + most_confused_with], axis=0)
                            if self.improved_finetune_noise:
                                if self.improved_finetune_type == 'v1':
                                    noise_tensor = np.random.normal(0, (np.max(
                                        tmp_tensor) - np.min(tmp_tensor)) / 6 * self.improved_finetune_noise_level,
                                                                    tmp_tensor.shape)  # 3 sigma
                                elif self.improved_finetune_type == 'v2':
                                    noise_tensor = np.random.normal(0, np.std(
                                        tmp_tensor) * self.improved_finetune_noise_level,
                                                                    tmp_tensor.shape)
                                else:
                                    raise Exception()
                                tmp_tensor += noise_tensor
                            input_tensor = np.concatenate((input_tensor, tmp_tensor))

                        # assign value
                        input_var_list = [var for var in tf.trainable_variables() if input_tensor_name in var.name]
                        assert len(input_var_list) == 1
                        self.sess.run(tf.assign(input_var_list[0], input_tensor))

            # reset the cache of the plot
            gan.tflib.plot.reset()

            # use k-means to initialize the prototypes
            dynamic_protos_val = []
            if not self.is_first_session:
                for proto_idx in range(self.nb_output - self.nb_cl):
                    protos = self.memory.get_protos(proto_idx)
                    dynamic_protos_val.append(protos)

            data_X_norm = 2 * ((data_X / 255.) - .5)
            if self.proto_select_criterion == 'random':
                print('Protos random...')
                for class_idx in range(self.nb_output - self.nb_cl, self.nb_output):
                    data_X_float_cur_class = data_X_norm[np.argmax(data_y, axis=1) == class_idx]
                    static_protos_val_cur_class_indices = np.random.choice(range(len(data_X_float_cur_class)),
                                                                           self.proto_num, replace=False)
                    static_protos_val_cur_class = data_X_float_cur_class[static_protos_val_cur_class_indices]
                    dynamic_protos_val.append(static_protos_val_cur_class)
                print('Protos random completed')
            elif self.proto_select_criterion == 'ori_kmeans':
                print('Processing KMeans...')
                for class_idx in range(self.nb_output - self.nb_cl, self.nb_output):
                    data_X_float_cur_class = data_X_norm[np.argmax(data_y, axis=1) == class_idx]
                    kmeans = KMeans(n_clusters=self.proto_num, random_state=0).fit(data_X_float_cur_class)
                    static_protos_val_cur_class = kmeans.cluster_centers_
                    dynamic_protos_val.append(static_protos_val_cur_class)
                print('KMeans completed')
            elif self.proto_select_criterion == 'feat_kmeans':
                print('Processing feature KMeans...')
                for class_idx in range(self.nb_output - self.nb_cl, self.nb_output):
                    data_X_int_cur_class = data_X[np.argmax(data_y, axis=1) == class_idx]
                    data_X_float_cur_class = data_X_norm[np.argmax(data_y, axis=1) == class_idx]
                    embedding_cur_class = self.sess.run(self.embedding_inputs,
                                                        feed_dict={self.inputs_int: data_X_int_cur_class})
                    kmeans = KMeans(n_clusters=self.proto_num, random_state=0).fit(embedding_cur_class)
                    selected_indices = []
                    for cluster_center in kmeans.cluster_centers_:
                        selected_idx = np.argmax(np.sum(np.square(embedding_cur_class - cluster_center), axis=1))
                        while selected_idx in selected_indices:
                            selected_idx = np.random.randint(0, len(embedding_cur_class))
                        selected_indices.append(selected_idx)
                    dynamic_protos_val.append(data_X_float_cur_class[selected_indices])
                print('feature KMeans completed')

            dynamic_protos_val = np.array(dynamic_protos_val)
            self.sess.run(tf.assign(self.protos, dynamic_protos_val))

            # add prototypes
            if not self.is_first_session and self.exemplars_dual_use:
                all_protos = self.sess.run(self.protos)
                all_protos = ((all_protos + 1.) * (255. / 2)).astype('int32')
                all_protos_arr = np.reshape(all_protos[:category_idx + 1 - self.nb_cl], [-1, self.output_dim])

                # if self.use_aug:
                #     images = all_protos_arr.reshape((-1, 1, 28, 28)).transpose((0, 2, 3, 1))
                #     seq = iaa.Sequential([
                #         iaa.Fliplr(0.5),  # horizontally flip 50% of the images
                #         iaa.GaussianBlur(sigma=(0, 0.0))  # blur images with a sigma of 0 to 3.0
                #     ])
                #     images_aug = seq(images=np.repeat(images, self.proto_importance, axis=0))
                #     data_X = images_aug.transpose((0, 3, 1, 2)).reshape((-1, self.output_dim))
                # else:
                data_X = np.concatenate((data_X, np.repeat(all_protos_arr, self.proto_importance, axis=0)))

                all_protos_y = np.eye(self.nb_output, dtype=float)[
                    np.repeat(range(category_idx + 1 - self.nb_cl), self.proto_num)]
                data_y = np.concatenate(
                    (data_y, np.repeat(all_protos_y, self.proto_importance, axis=0)))
                data_X_norm = 2 * ((data_X / 255.) - .5)

            # calc sample weight
            if self.anti_imbalance == 'reweight':
                weights = compute_sample_weight('balanced', np.argmax(data_y, axis=1))
            else:
                weights = np.ones(len(data_y))
            gen = get_train_inf(data_X, data_y, weights)

            pre_iters = 0

            history_conf_mat_dict = dict()
            for task_class_num in range(self.nb_cl, category_idx + 1 + self.nb_cl, self.nb_cl):
                history_conf_mat_dict['class_1-%d' % task_class_num] = dict()

            for iteration in range(self.iters):
                start_time = time.time()
                # Train generator
                if iteration > pre_iters:
                    _, _gen_cost, _gen_class_cost, _fake_accu, _proto_loss_fake = self.sess.run(
                        [self.gen_train_op, self.gen_cost, self.gen_class_cost,
                         self.fake_accu, self.proto_loss_fake],
                        feed_dict={self.gen_y: self.gen_labels(self.batch_size)})

                for _ in range(self.critic_iters):
                    _data_X, _data_y, _weights = gen.next()
                    _disc_cost, _, _real_accu, _real_class_cost, _proto_loss_real = \
                        self.sess.run([self.disc_cost, self.disc_train_op, self.real_accu, self.real_class_cost,
                                       self.proto_loss_real],
                                      feed_dict={self.real_data_int: _data_X,
                                                 self.real_y: _data_y,
                                                 self.gen_y: self.gen_labels(self.batch_size),
                                                 self.real_weight: _weights})

                # for _ in range(self.proto_iters):
                #     _data_X, _data_y = gen.next()
                #     _proto_cost, _ = \
                #         self.sess.run(
                #             [self.proto_cost, self.proto_train_op],
                #             feed_dict={self.real_data_int: _data_X,
                #                        self.real_y: _data_y,
                #                        self.gen_y: self.gen_labels(self.batch_size)})

                lib.plot.plot('time', time.time() - start_time)
                if iteration > pre_iters:
                    lib.plot.plot('train gen cost', _gen_cost)
                    lib.plot.plot('gen class cost', _gen_class_cost)
                    lib.plot.plot('gen accuracy', _fake_accu)
                    lib.plot.plot('proto loss fake', _proto_loss_fake)
                lib.plot.plot('proto loss real', _proto_loss_real)
                lib.plot.plot('train disc cost', _disc_cost)
                lib.plot.plot('real class cost', _real_class_cost)
                lib.plot.plot('real accuracy', _real_accu)

                if self.classification_only:
                    if iteration > pre_iters:
                        print(
                            "iter {}: real class: {}\tfake class: {}\ttime: {}".format(iteration + 1, _real_class_cost,
                                                                                       _gen_class_cost,
                                                                                       time.time() - start_time))
                    else:
                        print("iter {}: real class: {}\ttime: {}".format(iteration + 1, _real_class_cost,
                                                                         time.time() - start_time))

                else:
                    if (iteration + 1) % 500 == 0:
                        if iteration > pre_iters:
                            print(
                                "iter {}: disc: {}\tgen: {}\tgen class: {} (*{})\treal class: {} (*{})\tgen proto: {} (*{})\treal proto: {} (*{})\ttime: {}"
                                    .format(iteration + 1, _disc_cost, _gen_cost,
                                            _gen_class_cost * self.protogan_scale_g,
                                            self.protogan_scale_g, _real_class_cost, self.protogan_scale,
                                            _proto_loss_fake, self.proto_weight_fake, _proto_loss_real,
                                            self.proto_weight_real,
                                            time.time() - start_time))
                        else:
                            print(
                                "iter {}: disc: {}\treal class: {} (*{})\treal proto: {} (*{})\ttime: {}"
                                    .format(iteration + 1, _disc_cost, _real_class_cost, self.protogan_scale,
                                            _proto_loss_real, self.proto_weight_real,
                                            time.time() - start_time))

                if iteration == 0:
                    self.generate_image(iteration, train_log_dir_for_cur_class)
                    self.show_prototypes(iteration, train_log_dir_for_cur_class)

                # if (iteration + 1) < 500 and (iteration + 1) % 20 == 0:
                if (iteration + 1) % 2000 == 0:
                    self.generate_image(iteration + 1, train_log_dir_for_cur_class)
                    self.show_prototypes(iteration + 1, train_log_dir_for_cur_class)

                # Calculate dev loss and generate samples every 100 iters
                if (iteration + 1) % self.test_interval == 0:
                    dev_disc_costs = []
                    for images, labels in get_test_epoch(test_X, test_y):
                        _dev_disc_cost = self.sess.run(self.disc_cost, feed_dict={self.real_data_int: images,
                                                                                  self.real_y: labels,
                                                                                  self.real_weight: np.ones(
                                                                                      self.batch_size),
                                                                                  self.gen_y: self.gen_labels(
                                                                                      self.batch_size)})
                        dev_disc_costs.append(_dev_disc_cost)
                    gan.tflib.plot.plot('dev disc cost', np.mean(dev_disc_costs))

                    if self.dataset == 'fashion-mnist':
                        test_X_num_per_class = 1000
                        assert len(test_X) == (category_idx + 1) * test_X_num_per_class
                        pred_logits = []
                        for old_category_idx in range(category_idx + 1):
                            pred_logits_batch = self.classify_for_logits(test_X[
                                                                         test_X_num_per_class * old_category_idx: test_X_num_per_class * (
                                                                                 old_category_idx + 1)])
                            pred_logits.extend(pred_logits_batch)
                    elif self.dataset == 'mnist':
                        pred_logits = []
                        for pred_y_idx in range(0, len(test_X), 1000):
                            pred_logits_batch = self.classify_for_logits(test_X[pred_y_idx: pred_y_idx + 1000])
                            pred_logits.extend(pred_logits_batch)
                    else:
                        raise Exception()
                    pred_logits = np.array(pred_logits)
                    pred_y = np.argmax(pred_logits, axis=1)

                    # deprecated for MNIST
                    # _test_accu = np.sum(pred_y == np.argmax(test_y, 1)) / float(len(test_X))

                    # confusion matrix and accuracy per class
                    _test_conf_mat = confusion_matrix(pred_y, np.argmax(test_y, 1))
                    _test_accu_per_class = np.diag(_test_conf_mat) * 1. / np.sum(_test_conf_mat, axis=0)

                    _test_accu = np.mean(_test_accu_per_class)
                    lib.plot.plot('test accuracy', _test_accu)

                    print('Test accuracy: {}'.format(_test_accu))
                    print("Test accuracy: " + " | ".join(str(o) for o in _test_accu_per_class))

                    history_conf_mat_dict['class_1-%d' % (category_idx + 1)][iteration + 1] = _test_conf_mat

                    # old task acc & new forgetting rate
                    for old_task_class_num in range(self.nb_cl, category_idx + 1, self.nb_cl):
                        test_indices_old_task = np.argmax(test_y, axis=1) < old_task_class_num
                        pred_y_old_task = np.argmax(pred_logits[test_indices_old_task, :old_task_class_num], axis=1)
                        test_y_old_task = test_y[test_indices_old_task]
                        test_conf_mat_old_task = confusion_matrix(pred_y_old_task, np.argmax(test_y_old_task, 1))
                        history_conf_mat_dict['class_1-%d' % old_task_class_num][iteration + 1] = test_conf_mat_old_task
                        test_acc_old_task = np.sum(pred_y_old_task == np.argmax(test_y_old_task, axis=1)) / float(
                            len(pred_y_old_task))
                        print("Test accuracy (1-{}): {}".format(old_task_class_num, test_acc_old_task))

                # Save checkpoint
                if (iteration + 1) % self.save_interval == 0:
                    self.save(iteration + 1, category_idx)

                # Save logs every 100 iters
                if (iteration + 1) % 1000 == 0:
                    gan.tflib.plot.flush(train_log_dir_for_cur_class)

                # prototype updating if using feat_kmeans
                if (iteration + 1) % 2000 == 0 and self.proto_select_criterion == 'feat_kmeans':
                    dynamic_protos_val = []
                    if not self.is_first_session:
                        for proto_idx in range(self.nb_output - self.nb_cl):
                            protos = self.memory.get_protos(proto_idx)
                            dynamic_protos_val.append(protos)

                    print('Updating protos...')
                    print('Processing feature KMeans...')
                    for class_idx in range(self.nb_output - self.nb_cl, self.nb_output):
                        data_X_int_cur_class = data_X[np.argmax(data_y, axis=1) == class_idx]
                        data_X_float_cur_class = data_X_norm[np.argmax(data_y, axis=1) == class_idx]
                        embedding_cur_class = self.sess.run(self.embedding_inputs,
                                                            feed_dict={self.inputs_int: data_X_int_cur_class})
                        kmeans = KMeans(n_clusters=self.proto_num, random_state=0).fit(embedding_cur_class)
                        selected_indices = []
                        for cluster_center in kmeans.cluster_centers_:
                            selected_idx = np.argmax(np.sum(np.square(embedding_cur_class - cluster_center), axis=1))
                            while selected_idx in selected_indices:
                                selected_idx = np.random.randint(0, len(embedding_cur_class))
                            selected_indices.append(selected_idx)
                        dynamic_protos_val.append(data_X_float_cur_class[selected_indices])
                    print('feature KMeans completed')
                    dynamic_protos_val = np.array(dynamic_protos_val)
                    self.sess.run(tf.assign(self.protos, dynamic_protos_val))

                gan.tflib.plot.tick()

            # final save checkpoint
            self.save(iteration + 1, category_idx, final=True)

            # save history conf mat
            for key in history_conf_mat_dict:
                with open(os.path.join(train_log_dir_for_cur_class, '%s_conf_mat.pkl' % key), 'wb') as fout:
                    pickle.dump(history_conf_mat_dict[key], fout)

        # get fids
        if not self.classification_only:
            cond_fid_file = os.path.join(train_log_dir_for_cur_class, 'cond_fid.pkl')
            if not os.path.exists(cond_fid_file):
                history_cond_fid = dict()
                fid_vals = self.get_fid(train_log_dir_for_cur_class)
                history_cond_fid[self.iters] = fid_vals

                # save history cond fid
                with open(cond_fid_file, 'wb') as fout:
                    pickle.dump(history_cond_fid, fout)

        # get prototypes of the new classes
        protos = self.sess.run(self.protos)
        if not (np.max(protos) <= 1. and np.min(protos) >= -1.):
            protos = []
            for category_idx_in_loop in range(0, self.nb_output):
                protos_file = os.path.join(self.model_dir_for_class(category_idx), 'protos',
                                           'class_%d' % (category_idx_in_loop + 1), 'protos_%d.pkl' % self.iters)
                protos_data = pickle.load(open(protos_file, 'rb'))
                protos_data = 2 * ((protos_data / 255.) - .5)
                protos.append(protos_data)
            protos = np.array(protos)
        assert np.max(protos) <= 1. and np.min(protos) >= -1.
        for idx in range(self.nb_output):
            self.memory.add_protos(idx, protos[idx])

            # save memory
        if not self.check_memory(category_idx):
            self.save_memory(category_idx)

        USE_TSNE = False
        embedding_real_samples_protos_file = os.path.join(train_log_dir_for_cur_class,
                                                          'tsne.pdf' if USE_TSNE else 'umap.pdf')
        if not os.path.exists(embedding_real_samples_protos_file):
            print('Get tsne...')
            start_time = time.time()
            all_embedding_protos = np.zeros([0, 4096], np.float)
            all_embedding_samples = np.zeros([0, 4096], np.float)
            num_sample_per_class_arr = []

            # class
            for category_idx in range(self.nb_output):
                test_X_cur_class = test_X[np.argmax(test_y, axis=1) == category_idx]
                protos_cur_class = np.int32(((protos[category_idx] / 2.) + .5) * 255.)

                embedding_protos = self.sess.run(self.embedding_inputs, feed_dict={self.inputs_int: protos_cur_class})
                embedding_samples = self.sess.run(self.embedding_inputs, feed_dict={self.inputs_int: test_X_cur_class})
                all_embedding_protos = np.concatenate((all_embedding_protos, embedding_protos))
                all_embedding_samples = np.concatenate((all_embedding_samples, embedding_samples))
                num_sample_per_class_arr.append(len(embedding_samples))

            if USE_TSNE:
                tsne = TSNE(n_components=2)
                dim_reduction_result = tsne.fit_transform(np.concatenate((all_embedding_protos, all_embedding_samples)))
            else:
                tsne = UMAP()
                dim_reduction_result = tsne.fit_transform(np.concatenate((all_embedding_protos, all_embedding_samples)))
            plt.figure(figsize=(6, 6), dpi=150)
            for category_idx in range(self.nb_output):
                tsne_protos = dim_reduction_result[
                              category_idx * self.proto_num: (category_idx + 1) * self.proto_num]
                tsne_samples = dim_reduction_result[
                               len(all_embedding_protos) + np.sum(num_sample_per_class_arr[:category_idx],
                                                                  dtype=int): len(
                                   all_embedding_protos) + np.sum(num_sample_per_class_arr[:category_idx + 1],
                                                                  dtype=int)]
                plt.scatter(tsne_protos[:, 0], tsne_protos[:, 1], marker='+', alpha=1., color=colors[category_idx],
                            label='Class %d' % (category_idx + 1))
                plt.scatter(tsne_samples[:, 0], tsne_samples[:, 1], marker='.', alpha=1.,
                            color=colors[category_idx],
                            label='Class %d' % (category_idx + 1))

            plt.legend()
            plt.savefig(embedding_real_samples_protos_file)
            plt.close()

            print('Time used: %.2f' % (time.time() - start_time))

    def save_memory(self, category_idx):
        with open(os.path.join(self.model_dir_for_class(category_idx), 'memory.pkl'), 'wb') as fout:
            pickle.dump(self.memory, fout)

    def load_memory(self, category_idx):
        with open(os.path.join(self.model_dir_for_class(category_idx), 'memory.pkl'), 'rb') as fin:
            return pickle.load(fin)

    def check_memory(self, category_idx):
        return os.path.exists(os.path.join(self.model_dir_for_class(category_idx), 'memory.pkl'))

    @property
    def model_dir(self):
        return IntroGAN.model_dir_static(self)

    @staticmethod
    def model_dir_static(FLAGS):
        finetune_str = (('finetune_improved' + ('_v2' if FLAGS.improved_finetune_type == 'v2' else '') + (
            '_noise_%.1f' % FLAGS.improved_finetune_noise_level if FLAGS.improved_finetune_noise else '')) if FLAGS.improved_finetune else 'finetune') if FLAGS.finetune else 'from_scratch'
        finetune_str += ('_exemplars_dual_use_%d' % FLAGS.proto_importance if FLAGS.exemplars_dual_use else '')
        finetune_str += ('_%s' % FLAGS.anti_imbalance if not FLAGS.anti_imbalance == 'none' else '')
        mode_str = FLAGS.mode + '_critic_%d' % FLAGS.critic_iters
        mode_str += '_ac_%.1f_%.1f' % (FLAGS.protogan_scale, FLAGS.protogan_scale_g)
        proto_str = ('proto_static') + ('_%s' % FLAGS.proto_select_criterion) + ('_%d_weight_%f_%f' % (
            FLAGS.proto_num, FLAGS.proto_weight_real, FLAGS.proto_weight_fake)) + ('_%s' % FLAGS.dist_func) + (
                            '_%f' % FLAGS.gamma) + ('_margin_%.1f' % FLAGS.margin if not FLAGS.margin == 0 else '') + (
                        '' if FLAGS.center_type == 'rel_center' else '_%s' % FLAGS.center_type) + (
                        '_%d' % FLAGS.fixed_center_idx if FLAGS.center_type == 'fixed_center' else '') + (
                        '_train_rel_center' if FLAGS.train_rel_center else '') + (
                        '' if FLAGS.rigorous == 'max' else '_%s_select' % FLAGS.rigorous)
        return os.path.join(FLAGS.result_dir,
                            FLAGS.dataset + ('_order_%d' % FLAGS.order_idx) + (
                                '_subset_%d' % FLAGS.num_samples_per_class if not FLAGS.num_samples_per_class == -1 else ''),
                            'nb_cl_%d' % FLAGS.nb_cl, mode_str,
                            str(FLAGS.adam_lr) + '_' + str(FLAGS.adam_beta1) + '_' + str(FLAGS.adam_beta2),
                            str(FLAGS.iters), proto_str,
                            'classification_only_%s' % finetune_str if FLAGS.classification_only else finetune_str)

    @staticmethod
    def model_dir_for_class_static(FLAGS, category_idx):
        return os.path.join(IntroGAN.model_dir_static(FLAGS), 'class_%d-%d' % (1, category_idx + 1))

    def model_dir_for_class(self, category_idx):
        return os.path.join(self.model_dir, 'class_%d-%d' % (1, category_idx + 1))

    def save(self, step, category_idx, final=False):
        model_name = self.mode + ".model"
        if final:
            checkpoint_dir = os.path.join(self.model_dir_for_class(category_idx), "checkpoints", "final")
        else:
            checkpoint_dir = os.path.join(self.model_dir_for_class(category_idx), "checkpoints", str(step))

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # To make sure that the checkpoints of old classes are no longer recorded
        self.saver.set_last_checkpoints_with_time([])
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load_inception_score(self, category_idx, step=-1):
        log_file = os.path.join(self.model_dir_for_class(category_idx), "log", "log.pkl")
        with open(log_file, 'rb') as file:
            log_data = pickle.load(file)

        if step == -1:
            inception_score_max_idx = max(log_data["inception score"].keys())
            inception_score = log_data["inception score"][inception_score_max_idx]
        else:
            inception_score = log_data["inception score"][step - 1]

        return inception_score

    def load(self, category_idx, step=-1):
        import re
        print(" [*] Reading checkpoints...")
        if step == -1:
            checkpoint_dir = os.path.join(self.model_dir_for_class(category_idx), "checkpoints", "final")
        else:
            checkpoint_dir = os.path.join(self.model_dir_for_class(category_idx), "checkpoints", str(step))

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def load_finetune(self, category_idx, step=-1):
        import re
        print(" [*] Reading checkpoints...")
        if step == -1:
            checkpoint_dir = os.path.join(self.model_dir_for_class(category_idx), "checkpoints", "final")
        else:
            checkpoint_dir = os.path.join(self.model_dir_for_class(category_idx), "checkpoints", str(step))

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver_for_finetune.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))

            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter, os.path.join(checkpoint_dir, ckpt_name)
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0, ''

    @staticmethod
    def check_model(FLAGS, category_idx, step=-1):
        """
        Check whether the old models(which<category_idx) exist
        :param category_idx:
        :return: True or false
        """
        print(" [*] Checking checkpoints for class %d" % (category_idx + 1))
        if step == -1:
            checkpoint_dir = os.path.join(IntroGAN.model_dir_for_class_static(FLAGS, category_idx), "checkpoints",
                                          "final")
        else:
            checkpoint_dir = os.path.join(IntroGAN.model_dir_for_class_static(FLAGS, category_idx), "checkpoints",
                                          str(step))

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            return True
        else:
            return False
