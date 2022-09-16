import os
import pickle
import shutil
import time
from collections import Counter

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from PIL import Image
from sklearn.metrics import confusion_matrix
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


def leaky_relu(x, alpha=0.2):
    return tf.maximum(alpha * x, x)


def conv_cond_concat(x, y):
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat([x, y * tf.ones([x_shapes[0], y_shapes[1], x_shapes[2], x_shapes[3]])], 1)


class MeRGAN(object):

    def __init__(self, sess, graph, sess_fid, dataset, mode, batch_size, output_dim,
                 lambda_param, critic_iters, class_iters, iters, result_dir, checkpoint_interval,
                 adam_lr, adam_beta1, adam_beta2, finetune, improved_finetune, nb_cl, nb_output, acgan_scale,
                 acgan_scale_g,
                 classification_only, order_idx, order, test_interval, use_softmax, use_diversity_promoting,
                 diversity_promoting_weight, improved_finetune_type, improved_finetune_noise,
                 improved_finetune_noise_level,
                 num_samples_per_class,
                 use_protos, protos_path, protos_num, protos_importance):

        self.sess = sess
        self.graph = graph
        self.sess_fid = sess_fid

        self.dataset = dataset
        self.mode = mode
        self.batch_size = batch_size
        self.output_dim = output_dim

        self.lambda_param = lambda_param
        self.critic_iters = critic_iters
        self.class_iters = class_iters

        self.iters = iters
        self.result_dir = result_dir
        self.save_interval = checkpoint_interval

        self.adam_lr = adam_lr
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2

        self.acgan_scale = acgan_scale
        self.acgan_scale_g = acgan_scale_g

        self.finetune = finetune
        self.improved_finetune = improved_finetune
        self.improved_finetune_type = improved_finetune_type
        self.improved_finetune_noise = improved_finetune_noise
        self.improved_finetune_noise_level = improved_finetune_noise_level

        self.nb_cl = nb_cl
        self.nb_output = nb_output

        self.order_idx = order_idx
        self.order = order

        self.classification_only = classification_only

        self.use_diversity_promoting = use_diversity_promoting
        self.diversity_promoting_weight = diversity_promoting_weight

        self.is_first_session = (self.nb_output == self.nb_cl)

        self.test_interval = test_interval

        self.num_samples_per_class = num_samples_per_class

        self.use_protos = use_protos
        self.protos_importance = protos_importance
        self.protos_path = protos_path
        self.protos_num = protos_num

        self.use_softmax = use_softmax

        self.build_model()

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def build_model(self):

        lib.delete_all_params()

        with tf.variable_scope("gan") as scope, self.graph.as_default():
            # placeholder for MNIST samples
            self.real_data_int = tf.placeholder(tf.int32, shape=[self.batch_size, self.output_dim])
            self.real_y = tf.placeholder(tf.float32, shape=[None, self.nb_output])
            self.gen_y = tf.placeholder(tf.float32, shape=[None, self.nb_output])
            self.sample_y = tf.placeholder(tf.float32, shape=[None, self.nb_output])

            real_data = 2 * ((tf.cast(self.real_data_int, tf.float32) / 255.) - .5)
            fake_data, fake_labels, noise = self.generator(self.batch_size, self.gen_y)

            # set_shape to facilitate concatenation of label and image
            fake_data.set_shape([self.batch_size, fake_data.shape[1].value])
            fake_labels.set_shape([self.batch_size, fake_labels.shape[1].value])

            disc_real, disc_real_class = self.discriminator(real_data, reuse=False, is_training=True)
            disc_fake, disc_fake_class = self.discriminator(fake_data, reuse=True, is_training=True)

            # Get output label
            self.inputs_int = tf.placeholder(tf.int32, shape=[None, self.output_dim])
            inputs = 2 * ((tf.cast(self.inputs_int, tf.float32) / 255.) - .5)
            _, self.class_logits = self.discriminator(inputs, reuse=True, is_training=False)
            self.pred_y = tf.argmax(self.class_logits, 1)

            self.embeddings = self.embedding_func(inputs, reuse=True)

            self.real_accu = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(disc_real_class, 1), tf.argmax(self.real_y, 1)), dtype=tf.float32))
            self.fake_accu = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(disc_fake_class, 1), tf.argmax(fake_labels, 1)), dtype=tf.float32))

            if self.use_softmax:
                self.real_class_cost = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(logits=disc_real_class, labels=self.real_y))
                self.gen_class_cost = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(logits=disc_fake_class, labels=fake_labels))
            else:
                self.real_class_cost = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real_class, labels=self.real_y))
                self.gen_class_cost = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_class, labels=fake_labels))

            self.class_cost = self.real_class_cost * self.acgan_scale

            gen_params = [var for var in tf.trainable_variables() if 'Generator' in var.name]
            disc_params = [var for var in tf.trainable_variables() if
                           'Discriminator' in var.name and 'ClassOutput' not in var.name]
            class_params = [var for var in tf.trainable_variables() if
                            'Discriminator' in var.name and 'SourceOutput' not in var.name]

            if self.mode == 'wgan-gp':
                # Standard WGAN loss
                self.gen_cost = -tf.reduce_mean(disc_fake)
                self.disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

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
                # DCGAN loss
                self.gen_cost = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.ones_like(disc_fake)))
                self.disc_cost = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.zeros_like(disc_fake)))
                self.disc_cost += tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real, labels=tf.ones_like(disc_real)))
                self.disc_cost /= 2.

            self.gen_cost += self.acgan_scale_g * self.gen_class_cost

            # diversity promoting
            if self.use_diversity_promoting:
                noise_split = tf.split(noise, 2)
                gen_samples_split = tf.split(fake_data, 2)
                self.diversity_cost = tf.reduce_mean(tf.abs(gen_samples_split[0] - gen_samples_split[1])) / \
                                      tf.reduce_mean(tf.abs(noise_split[0] - noise_split[1]))
                self.gen_cost -= self.diversity_promoting_weight * self.diversity_cost
            else:
                self.diversity_cost = tf.no_op()

            if self.classification_only:
                self.gen_train_op = \
                    tf.train.AdamOptimizer(learning_rate=0,
                                           beta1=self.adam_beta1, beta2=self.adam_beta2) \
                        .minimize(self.gen_cost, var_list=gen_params)
                self.disc_train_op = \
                    tf.train.AdamOptimizer(learning_rate=0,
                                           beta1=self.adam_beta1, beta2=self.adam_beta2) \
                        .minimize(self.disc_cost, var_list=disc_params)
                self.class_train_op = \
                    tf.train.AdamOptimizer(learning_rate=self.adam_lr,
                                           beta1=self.adam_beta1, beta2=self.adam_beta2) \
                        .minimize(self.class_cost, var_list=class_params)
            else:
                self.gen_train_op = \
                    tf.train.AdamOptimizer(learning_rate=self.adam_lr,
                                           beta1=self.adam_beta1, beta2=self.adam_beta2) \
                        .minimize(self.gen_cost, var_list=gen_params)
                self.disc_train_op = \
                    tf.train.AdamOptimizer(learning_rate=self.adam_lr,
                                           beta1=self.adam_beta1, beta2=self.adam_beta2) \
                        .minimize(self.disc_cost, var_list=disc_params)
                self.class_train_op = \
                    tf.train.AdamOptimizer(learning_rate=self.adam_lr,
                                           beta1=self.adam_beta1, beta2=self.adam_beta2) \
                        .minimize(self.class_cost, var_list=class_params)

            # For generating samples
            fixed_noise_128 = tf.constant(np.random.normal(size=(128, 100)).astype('float32'))
            self.fixed_noise_samples_128 = self.sampler(128, self.sample_y, noise=fixed_noise_128)[0]

            # For calculating inception score
            self.test_noise = tf.random_normal([self.batch_size, 100])
            self.test_samples = self.sampler(self.batch_size, self.sample_y, noise=self.test_noise)[0]

            var_list = tf.trainable_variables()

            bn_moving_vars = [var for var in tf.global_variables() if 'moving_mean' in var.name]
            bn_moving_vars += [var for var in tf.global_variables() if 'moving_variance' in var.name]
            var_list += bn_moving_vars

            self.saver = tf.train.Saver(var_list=var_list)  # var_list doesn't contain Adam params

            if self.finetune:
                if self.improved_finetune:
                    var_list_for_finetune = [var for var in var_list if
                                             'g_Input.W' not in var.name and 'd_ClassOutput' not in var.name]
                else:
                    var_list_for_finetune = [var for var in var_list if
                                             'g_Input' not in var.name and 'd_ClassOutput' not in var.name]
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
            samples_folder = os.path.join(train_log_dir_for_cur_class, 'samples', 'class_%d' % (category_idx + 1))
            if not os.path.exists(samples_folder):
                os.makedirs(samples_folder)
            gan.tflib.save_images.save_images(samples.reshape((128, 1, 28, 28)),
                                              os.path.join(samples_folder,
                                                           'samples_{}.jpg'.format(frame)))
            # dump samples for visualization etc.
            with open(os.path.join(samples_folder, 'samples_{}.pkl'.format(frame)), 'wb') as fout:
                pickle.dump(samples, fout)

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

            return tf.reshape(output, [-1, self.output_dim]), y, noise

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

    def embedding_func(self, inputs, reuse):
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
            output = leaky_relu(output)

            output = tf.reshape(output, [-1, 4 * 4 * 4 * 64])

            output = slim.flatten(output)
            return output

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
            output = leaky_relu(output)

            output = tf.reshape(output, [-1, 4 * 4 * 4 * 64])

            output = slim.flatten(output)
            sourceOutput = gan.tflib.ops.linear.Linear('d_SourceOutput', 4 * 4 * 4 * 64, 1, output)
            classOutput = gan.tflib.ops.linear.Linear('d_ClassOutput', 4 * 4 * 4 * 64, self.nb_output, output)

            return tf.reshape(sourceOutput, shape=[-1]), tf.reshape(classOutput, shape=[-1, self.nb_output])

    def classify(self, inputs):

        with self.graph.as_default():
            result = self.sess.run(self.pred_y, feed_dict={self.inputs_int: inputs})

        return result

    def classify_for_logits(self, inputs):

        with self.graph.as_default():
            logits = self.sess.run(self.class_logits, feed_dict={self.inputs_int: inputs})

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

            def get_train_inf(data_X, data_y):
                while True:
                    batch_idxs = len(data_X) // self.batch_size
                    if batch_idxs == 0:
                        idx = np.random.choice(len(data_X), self.batch_size, replace=True)
                        yield (data_X[idx], data_y[idx])
                    else:
                        reorder = np.array(range(len(data_X)))
                        np.random.shuffle(reorder)
                        data_X = data_X[reorder]
                        data_y = data_y[reorder]
                        for idx in range(0, batch_idxs):
                            _data_X = data_X[idx * self.batch_size:(idx + 1) * self.batch_size]
                            _data_y = data_y[idx * self.batch_size:(idx + 1) * self.batch_size]
                            yield (_data_X, _data_y)

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
                                              'g_Input.W' in key or 'd_ClassOutput' in key]
                        assert len(input_tensor_names) == 3
                        g_Input_offset = 100

                        assert 'g_Input.W' in input_tensor_names[0]
                        assert 'd_ClassOutput.W' in input_tensor_names[1]
                        assert 'd_ClassOutput.b' in input_tensor_names[2]

                        g_Input_W_name = input_tensor_names[0]
                        d_ClassOutput_W_name = input_tensor_names[1]
                        d_ClassOutput_b_name = input_tensor_names[2]

                        g_Input_W = ckpt_reader.get_tensor(g_Input_W_name)
                        d_ClassOutput_W = ckpt_reader.get_tensor(d_ClassOutput_W_name)
                        d_ClassOutput_b = ckpt_reader.get_tensor(d_ClassOutput_b_name)

                        for new_category_idx in range(category_idx - self.nb_cl + 1, category_idx + 1):
                            most_confused_with = \
                                Counter(pred_y[np.argmax(data_y_new_classes, axis=1) == new_category_idx]).most_common(
                                    1)[
                                    0][0]

                            tmp_tensor = np.expand_dims(g_Input_W[g_Input_offset + most_confused_with], axis=0)
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
                            g_Input_W = np.concatenate((g_Input_W, tmp_tensor))

                            tmp_tensor = np.expand_dims(d_ClassOutput_W[:, most_confused_with], axis=1)
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
                            d_ClassOutput_W = np.concatenate((d_ClassOutput_W, tmp_tensor), axis=1)

                            tmp_tensor = d_ClassOutput_b[most_confused_with]
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
                            d_ClassOutput_b = np.append(d_ClassOutput_b, tmp_tensor)

                        # assign value
                        input_var_list = [var for var in tf.trainable_variables() if g_Input_W_name in var.name]
                        assert len(input_var_list) == 1
                        self.sess.run(tf.assign(input_var_list[0], g_Input_W))

                        input_var_list = [var for var in tf.trainable_variables() if d_ClassOutput_W_name in var.name]
                        assert len(input_var_list) == 1
                        self.sess.run(tf.assign(input_var_list[0], d_ClassOutput_W))

                        input_var_list = [var for var in tf.trainable_variables() if d_ClassOutput_b_name in var.name]
                        assert len(input_var_list) == 1
                        self.sess.run(tf.assign(input_var_list[0], d_ClassOutput_b))

            # add prototypes
            if not self.is_first_session and self.use_protos:
                real_protos_folder = os.path.join(os.path.dirname(self.result_dir), self.protos_path % (self.dataset,
                                                                                                        self.order_idx,
                                                                                                        self.nb_cl,
                                                                                                        self.protos_num,
                                                                                                        self.protos_importance),
                                                  'class_1-%d' % (category_idx + 1 - self.nb_cl))
                protos_data_file = os.path.join(real_protos_folder, 'memory.pkl')
                with open(protos_data_file, 'rb') as fin:
                    protos_data = pickle.load(fin)

                all_protos = []
                for old_class_idx in range(category_idx + 1 - self.nb_cl):
                    all_protos.extend(((protos_data.protos[old_class_idx] + 1.) * (255. / 2)).astype('uint8'))
                all_protos = np.asarray(all_protos)

                data_X = np.concatenate((data_X, np.repeat(all_protos, self.protos_importance, axis=0)))
                all_protos_y = np.eye(self.nb_output, dtype=float)[
                    np.repeat(range(category_idx + 1 - self.nb_cl), self.protos_num)]
                data_y = np.concatenate(
                    (data_y, np.repeat(all_protos_y, self.protos_importance, axis=0)))

            gen = get_train_inf(data_X, data_y)

            # reset the cache of the plot
            gan.tflib.plot.reset()

            pre_iters = 0

            history_conf_mat_dict = dict()

            for task_class_num in range(self.nb_cl, category_idx + self.nb_cl + (1 if self.nb_cl == 1 else 0),
                                        self.nb_cl):
                history_conf_mat_dict['class_1-%d' % task_class_num] = dict()

            for iteration in range(self.iters):
                start_time = time.time()
                # Train generator
                if iteration > pre_iters:
                    _, _gen_cost, _gen_class_cost, _fake_accu, _diversity_cost = self.sess.run(
                        [self.gen_train_op, self.gen_cost, self.gen_class_cost, self.fake_accu, self.diversity_cost],
                        feed_dict={self.gen_y: self.gen_labels(self.batch_size)})

                for _ in range(self.critic_iters):
                    _data_X, _data_y = gen.next()
                    _disc_cost, _ = \
                        self.sess.run([self.disc_cost, self.disc_train_op],
                                      feed_dict={self.real_data_int: _data_X,
                                                 self.gen_y: self.gen_labels(self.batch_size)})

                for _ in range(self.class_iters):
                    _data_X, _data_y = gen.next()
                    _real_accu, _real_class_cost, _class_cost, _ = \
                        self.sess.run(
                            [self.real_accu,
                             self.real_class_cost, self.class_cost, self.class_train_op],
                            feed_dict={self.real_data_int: _data_X,
                                       self.real_y: _data_y})

                lib.plot.plot('time', time.time() - start_time)
                if iteration > pre_iters:
                    lib.plot.plot('train gen cost', _gen_cost)
                    lib.plot.plot('gen class cost', _gen_class_cost)
                    lib.plot.plot('gen accuracy', _fake_accu)
                lib.plot.plot('train disc cost', _disc_cost)
                lib.plot.plot('train class cost', _class_cost)
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
                                "iter {}: disc: {}\tgen: {}\tgen class: {} (*{})\tclass: {} (*{})\ttime: {}"
                                    .format(iteration + 1, _disc_cost, _gen_cost, _gen_class_cost * self.acgan_scale_g,
                                            self.acgan_scale_g, _class_cost, self.acgan_scale,
                                            time.time() - start_time))
                        else:
                            print(
                                "iter {}: disc: {}\tclass: {} (*{})\ttime: {}"
                                    .format(iteration + 1, _disc_cost, _class_cost, self.acgan_scale,
                                            time.time() - start_time))

                if iteration == 0:
                    self.generate_image(iteration, train_log_dir_for_cur_class)

                # if (iteration + 1) < 500 and (iteration + 1) % 20 == 0:
                if (iteration + 1) % 2000 == 0:
                    self.generate_image(iteration + 1, train_log_dir_for_cur_class)

                # Calculate dev loss and generate samples every 100 iters
                if (iteration + 1) % self.test_interval == 0:
                    dev_disc_costs = []
                    for images, labels in get_test_epoch(test_X, test_y):
                        _dev_disc_cost = self.sess.run(self.disc_cost, feed_dict={self.real_data_int: images,
                                                                                  self.real_y: labels,
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

                gan.tflib.plot.tick()

            # final save checkpoint
            self.save(iteration + 1, category_idx, final=True)

            cond_fid_file = os.path.join(train_log_dir_for_cur_class, 'cond_fid.pkl')
            if not os.path.exists(cond_fid_file):
                history_cond_fid = dict()
                fid_vals = self.get_fid(train_log_dir_for_cur_class)
                history_cond_fid[self.iters] = fid_vals

                # save history cond fid
                with open(cond_fid_file, 'wb') as fout:
                    pickle.dump(history_cond_fid, fout)
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

        self.generate_image(self.iters + 1, train_log_dir_for_cur_class)

    @property
    def model_dir(self):
        return MeRGAN.model_dir_static(self)

    @staticmethod
    def model_dir_static(FLAGS):
        finetune_str = (('finetune_improved' + ('_v2' if FLAGS.improved_finetune_type == 'v2' else '') + (
            '_noise_%.1f' % FLAGS.improved_finetune_noise_level if FLAGS.improved_finetune_noise else '')) if FLAGS.improved_finetune else 'finetune') if FLAGS.finetune else 'from_scratch'
        finetune_str += (
            '_use_%d_protos_weight_%f' % (FLAGS.protos_num, FLAGS.protos_importance) if FLAGS.use_protos else '')
        mode_str = FLAGS.mode + '_critic_%d_class_%d' % (FLAGS.critic_iters, FLAGS.class_iters) + (
            '' if FLAGS.use_softmax else '_sigmoid')
        mode_str += '_ac_%.1f_%.1f' % (FLAGS.acgan_scale, FLAGS.acgan_scale_g)
        mode_str += '_diversity_promoting_%f' % FLAGS.diversity_promoting_weight if FLAGS.use_diversity_promoting else ''
        return os.path.join(FLAGS.result_dir, FLAGS.dataset + '_order_%d' % FLAGS.order_idx + (
            '_subset_%d' % FLAGS.num_samples_per_class if not FLAGS.num_samples_per_class == -1 else ''),
                            'nb_cl_%d' % FLAGS.nb_cl, mode_str,
                            str(FLAGS.adam_lr) + '_' + str(FLAGS.adam_beta1) + '_' + str(FLAGS.adam_beta2),
                            str(FLAGS.iters),
                            'classification_only_%s' % finetune_str if FLAGS.classification_only else finetune_str)

    @staticmethod
    def model_dir_for_class_static(FLAGS, category_idx):
        return os.path.join(MeRGAN.model_dir_static(FLAGS), 'class_%d-%d' % (1, category_idx + 1))

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
            checkpoint_dir = os.path.join(MeRGAN.model_dir_for_class_static(FLAGS, category_idx), "checkpoints", "final")
        else:
            checkpoint_dir = os.path.join(MeRGAN.model_dir_for_class_static(FLAGS, category_idx), "checkpoints",
                                          str(step))

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            return True
        else:
            return False
