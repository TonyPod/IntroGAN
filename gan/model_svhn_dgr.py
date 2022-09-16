import os
import pickle
import shutil
import time
from collections import Counter

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import gan.tflib as lib
import gan.tflib.inception_score
import gan.tflib.ops
import gan.tflib.ops.batchnorm
import gan.tflib.ops.cond_batchnorm
import gan.tflib.ops.conv2d
import gan.tflib.ops.deconv2d
import gan.tflib.ops.layernorm
import gan.tflib.ops.linear
import gan.tflib.plot
import gan.tflib.save_images
from utils.fid import calculate_fid_given_paths_with_sess


class GAN(object):

    def __init__(self, sess, graph, sess_fid, dataset_name, mode, batch_size, output_dim,
                 lambda_param, critic_iters, iters, solver_iters, result_dir, checkpoint_interval,
                 adam_lr, solver_adam_lr, adam_beta1, adam_beta2, finetune, improved_finetune, nb_cl, nb_output,
                 dgr_ratio, order_idx, order, test_interval, dim,
                 improved_finetune_type, improved_finetune_noise, improved_finetune_noise_level):

        self.sess = sess
        self.graph = graph
        self.sess_fid = sess_fid

        self.dataset_name = dataset_name
        self.mode = mode
        self.batch_size = batch_size
        self.output_dim = output_dim

        self.lambda_param = lambda_param
        self.critic_iters = critic_iters
        self.solver_iters = solver_iters

        self.iters = iters
        self.result_dir = result_dir
        self.save_interval = checkpoint_interval

        self.adam_lr = adam_lr
        self.solver_adam_lr = solver_adam_lr
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2

        self.finetune = finetune
        self.improved_finetune = improved_finetune
        self.improved_finetune_type = improved_finetune_type
        self.improved_finetune_noise = improved_finetune_noise
        self.improved_finetune_noise_level = improved_finetune_noise_level

        self.nb_cl = nb_cl
        self.nb_output = nb_output

        self.order_idx = order_idx
        self.order = order

        self.dgr_ratio = dgr_ratio

        self.dim = dim

        self.is_first_session = (self.nb_output == self.nb_cl)

        self.test_interval = test_interval

        self.build_model()

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def build_model(self):

        lib.delete_all_params()

        with tf.variable_scope("gan") as scope, self.graph.as_default():

            '''
            Generator
            '''
            self.real_data_int_G = tf.placeholder(tf.int32, shape=[self.batch_size, self.output_dim])
            real_data = 2 * ((tf.cast(self.real_data_int_G, tf.float32) / 255.) - .5)
            fake_data = self.generator(self.batch_size)

            # set_shape to facilitate concatenation of label and image
            fake_data.set_shape([self.batch_size, fake_data.shape[1].value])

            disc_real = self.discriminator(real_data, reuse=False, is_training=True)
            disc_fake = self.discriminator(fake_data, reuse=True, is_training=True)

            gen_params = [var for var in tf.trainable_variables() if 'Generator' in var.name]
            disc_params = [var for var in tf.trainable_variables() if 'Discriminator' in var.name]

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
                                         [interpolates])
                slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
                gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
                self.disc_cost += self.lambda_param * gradient_penalty
            elif self.mode == 'dcgan':
                # Vanilla / Non-saturating loss
                self.gen_cost = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.ones_like(disc_fake)))
                self.disc_cost = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.zeros_like(disc_fake)))
                self.disc_cost += tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real, labels=tf.ones_like(disc_real)))
                self.disc_cost /= 2.

            self.gen_train_op = \
                tf.train.AdamOptimizer(learning_rate=self.adam_lr,
                                       beta1=self.adam_beta1, beta2=self.adam_beta2) \
                    .minimize(self.gen_cost, var_list=gen_params)
            self.disc_train_op = \
                tf.train.AdamOptimizer(learning_rate=self.adam_lr,
                                       beta1=self.adam_beta1, beta2=self.adam_beta2) \
                    .minimize(self.disc_cost, var_list=disc_params)

            # For generating samples
            fixed_noise_256 = tf.constant(np.random.normal(size=(256, 128)).astype('float32'))
            self.fixed_noise_samples_256 = self.sampler(256, noise=fixed_noise_256)

            '''
            Solver
            '''
            # train
            self.real_data_int_S = tf.placeholder(tf.int32, shape=[self.batch_size, self.output_dim])
            self.real_y_S = tf.placeholder(tf.float32, shape=[self.batch_size, self.nb_output])
            inputs = 2 * ((tf.cast(self.real_data_int_S, tf.float32) / 255.) - .5)
            class_logits = self.solver(inputs, reuse=False, is_training=True)
            self.train_batch_weights = tf.placeholder(tf.float32, shape=[None])

            self.solver_cost = \
                tf.losses.softmax_cross_entropy(logits=class_logits, onehot_labels=self.real_y_S,
                                                weights=self.train_batch_weights)

            # test
            self.real_data_int_S_test = tf.placeholder(tf.int32, shape=[None, self.output_dim])
            inputs_test = 2 * ((tf.cast(self.real_data_int_S_test, tf.float32) / 255.) - .5)
            self.class_logits_test = self.solver(inputs_test, reuse=True, is_training=False)
            self.pred_y = tf.argmax(self.class_logits_test, 1)
            self.pred_prob = tf.nn.softmax(self.class_logits_test)

            solver_params = [var for var in tf.trainable_variables() if 'Solver' in var.name]

            self.solver_train_op = \
                tf.train.AdamOptimizer(learning_rate=self.adam_lr,
                                       beta1=self.adam_beta1, beta2=self.adam_beta2) \
                    .minimize(self.solver_cost, var_list=solver_params)

            # For calculating inception score
            self.test_noise = tf.random_normal([self.batch_size, 128])
            self.test_samples = self.sampler(self.batch_size, noise=self.test_noise)

            '''
            Saver
            '''
            var_list = tf.trainable_variables()

            bn_moving_vars = [var for var in tf.global_variables() if 'moving_mean' in var.name]
            bn_moving_vars += [var for var in tf.global_variables() if 'moving_variance' in var.name]
            var_list += bn_moving_vars

            self.saver = tf.train.Saver(var_list=var_list)  # var_list doesn't contain Adam params

            if self.finetune:
                var_list_for_finetune = [var for var in var_list if 's_ClassOutput' not in var.name]
                self.saver_for_finetune = tf.train.Saver(var_list=var_list_for_finetune)

    def generate_image(self, frame, train_log_dir_for_cur_class):
        samples = self.sess.run(self.fixed_noise_samples_256)
        samples = ((samples + 1.) * (255. / 2)).astype('int32')
        samples_folder = os.path.join(train_log_dir_for_cur_class, 'samples')
        if not os.path.exists(samples_folder):
            os.makedirs(samples_folder)
        gan.tflib.save_images.save_images(samples.reshape((256, 3, 32, 32)),
                                          os.path.join(samples_folder,
                                                       'samples_{}.jpg'.format(frame)))

        # dump samples for visualization etc.
        with open(os.path.join(samples_folder, 'samples_{}.pkl'.format(frame)), 'wb') as fout:
            pickle.dump(samples, fout)

    def get_fid(self, train_log_dir_for_cur_class):
        print('Calculating fid...')
        time_start = time.time()
        FID_NUM = 10000 * self.nb_output
        fid_vals = {}

        temp_folder = os.path.join(train_log_dir_for_cur_class, 'fid_temp')

        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)

        # generate and classify
        x = []
        y = []
        pseudo_y = np.zeros(FID_NUM)
        for i in tqdm(range(0, FID_NUM, self.batch_size)):
            x_batch, _, _ = self.test(len(pseudo_y[i:i + self.batch_size]))
            x.extend(x_batch)

            y_batch = self.get_label(x_batch)
            y.extend(y_batch)
        x = np.array(x)
        y = np.array(y)

        print('Generation: %.2f seconds' % (time.time() - time_start))

        # move to the corresponding folder
        category_idx_no_pics = []
        last_category_idx_with_pics = 0
        for category_idx in range(self.nb_output):
            sub_folder = os.path.join(temp_folder, 'class_%d' % (category_idx + 1))
            if not os.path.exists(sub_folder):
                os.makedirs(sub_folder)

            x_cur_class = x[y == category_idx]
            print('Class %d: %d samples (total: %d)' % (category_idx + 1, len(x_cur_class), FID_NUM))

            if len(x_cur_class) <= 1:
                category_idx_no_pics.append(category_idx)
            else:
                last_category_idx_with_pics = category_idx

            for i, x_single in enumerate(x_cur_class):
                img = Image.fromarray(x_single.astype('uint8').reshape((3, 32, 32)).transpose((1, 2, 0)))
                img.save(os.path.join(sub_folder, '%d.jpg' % (i + 1)))

        for category_idx in range(self.nb_output):
            sub_folder = os.path.join(temp_folder, 'class_%d' % (category_idx + 1))
            if category_idx in category_idx_no_pics:
                sub_folder = os.path.join(temp_folder, 'class_%d' % (last_category_idx_with_pics + 1))
            fid_val = calculate_fid_given_paths_with_sess(self.sess_fid,
                                                          [sub_folder,
                                                           'precalc_fids/%s/fid_stats_%d.npz' % (
                                                               self.dataset_name, self.order[category_idx] + 1)])
            fid_vals[category_idx + 1] = fid_val

        # delete temp folders
        shutil.rmtree(temp_folder)

        time_stop = time.time()
        print('Total: %.2f seconds' % (time_stop - time_start))

        return fid_vals

    def get_label(self, inputs):
        with self.graph.as_default():
            label = self.sess.run(self.pred_y, feed_dict={self.real_data_int_S_test: inputs})
            return label

    def leaky_relu(self, x, alpha=0.2):
        return tf.maximum(alpha * x, x)

    def Normalize(self, name, inputs, labels=None):
        if labels is not None:
            return lib.ops.cond_batchnorm.Cond_Batchnorm(name, [0, 2, 3], inputs, labels=labels,
                                                         n_labels=self.nb_output)
        else:
            return lib.ops.batchnorm.Batchnorm(name, [0, 2, 3], inputs, fused=True)

    def test(self, n_samples):
        assert n_samples > 0
        with self.graph.as_default():
            samples, z = self.sess.run([self.test_samples, self.test_noise])

        if n_samples < self.batch_size:
            samples = samples[:n_samples]
            z = z[:n_samples]

        samples_int = ((samples + 1.) * (255. / 2)).astype('int32')
        return samples_int, samples, z

    def generator(self, n_samples, noise=None):

        with tf.variable_scope('Generator') as scope:
            if noise is None:
                noise = tf.random_normal([n_samples, 128])

            output = lib.ops.linear.Linear('Generator.Input', 128, 4 * 4 * 4 * self.dim, noise)
            output = tf.reshape(output, [-1, 4 * self.dim, 4, 4])
            output = self.Normalize('Generator.BN1', output)
            output = tf.nn.relu(output)

            output = lib.ops.deconv2d.Deconv2D('Generator.2', 4 * self.dim, 2 * self.dim, 5, output)
            output = self.Normalize('Generator.BN2', output)
            output = tf.nn.relu(output)

            output = lib.ops.deconv2d.Deconv2D('Generator.3', 2 * self.dim, self.dim, 5, output)
            output = self.Normalize('Generator.BN3', output)
            output = tf.nn.relu(output)

            output = lib.ops.deconv2d.Deconv2D('Generator.5', self.dim, 3, 5, output)

            output = tf.tanh(output)

            return tf.reshape(output, [-1, self.output_dim])

    def sampler(self, n_samples, noise=None):

        with tf.variable_scope('Generator') as scope:
            scope.reuse_variables()

            if noise is None:
                noise = tf.random_normal([n_samples, 128])

            output = lib.ops.linear.Linear('Generator.Input', 128, 4 * 4 * 4 * self.dim, noise)
            output = tf.reshape(output, [-1, 4 * self.dim, 4, 4])
            output = self.Normalize('Generator.BN1', output)
            output = tf.nn.relu(output)

            output = lib.ops.deconv2d.Deconv2D('Generator.2', 4 * self.dim, 2 * self.dim, 5, output)
            output = self.Normalize('Generator.BN2', output)
            output = tf.nn.relu(output)

            output = lib.ops.deconv2d.Deconv2D('Generator.3', 2 * self.dim, self.dim, 5, output)
            output = self.Normalize('Generator.BN3', output)
            output = tf.nn.relu(output)

            output = lib.ops.deconv2d.Deconv2D('Generator.5', self.dim, 3, 5, output)

            output = tf.tanh(output)

            return tf.reshape(output, [-1, self.output_dim])

    def discriminator(self, inputs, reuse, is_training):

        with tf.variable_scope('Discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            output = tf.reshape(inputs, [-1, 3, 32, 32])

            output = lib.ops.conv2d.Conv2D('Discriminator.1', 3, self.dim, 5, output, stride=2)
            output = self.leaky_relu(output)

            output = lib.ops.conv2d.Conv2D('Discriminator.2', self.dim, 2 * self.dim, 5, output, stride=2)
            output = self.leaky_relu(output)

            output = lib.ops.conv2d.Conv2D('Discriminator.3', 2 * self.dim, 4 * self.dim, 5, output, stride=2)
            output = self.leaky_relu(output)

            output = tf.reshape(output, [-1, 4 * 4 * 4 * self.dim])

            sourceOutput = lib.ops.linear.Linear('d_SourceOutput', 4 * 4 * 4 * self.dim, 1, output)

            return tf.reshape(sourceOutput, shape=[-1])

    def solver(self, inputs, reuse, is_training):

        with tf.variable_scope('Solver') as scope:
            if reuse:
                scope.reuse_variables()

            output = tf.reshape(inputs, [-1, 3, 32, 32])

            output = lib.ops.conv2d.Conv2D('Solver.1', 3, self.dim, 5, output, stride=2)
            output = self.leaky_relu(output)

            output = lib.ops.conv2d.Conv2D('Solver.2', self.dim, 2 * self.dim, 5, output, stride=2)
            output = self.leaky_relu(output)

            output = lib.ops.conv2d.Conv2D('Solver.3', 2 * self.dim, 4 * self.dim, 5, output, stride=2)
            output = self.leaky_relu(output)

            output = tf.reshape(output, [-1, 4 * 4 * 4 * self.dim])

            classOutput = lib.ops.linear.Linear('s_ClassOutput', 4 * 4 * 4 * self.dim, self.nb_output, output)
            return tf.reshape(classOutput, shape=[-1, self.nb_output])

    def classify_for_logits(self, inputs):

        with self.graph.as_default():
            logits = self.sess.run(self.class_logits_test, feed_dict={self.real_data_int_S_test: inputs})

        return logits

    def train(self, data_X, data_y, train_weights, test_X, test_y, category_idx, model_exist=False):

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

            def get_train_inf(data_X, data_y, train_weights):
                while True:
                    batch_idxs = len(data_X) // self.batch_size
                    reorder = np.array(range(len(data_X)))
                    np.random.shuffle(reorder)
                    data_X = data_X[reorder]
                    data_y = data_y[reorder]
                    train_weights = train_weights[reorder]
                    for idx in range(0, batch_idxs):
                        _data_X = data_X[idx * self.batch_size:(idx + 1) * self.batch_size]
                        _data_y = data_y[idx * self.batch_size:(idx + 1) * self.batch_size]
                        _train_weights = train_weights[idx * self.batch_size:(idx + 1) * self.batch_size]
                        yield (_data_X, _data_y, _train_weights)

            with self.graph.as_default():
                # Train loop
                self.sess.run(tf.initialize_all_variables())
                if self.finetune and not self.is_first_session:
                    _, _, ckpt_path = self.load_finetune(category_idx - self.nb_cl)

                    # special initialized for Input
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
                                              's_ClassOutput' in key]
                        assert len(input_tensor_names) == 2

                        assert 's_ClassOutput.b' in input_tensor_names[0]
                        assert 's_ClassOutput.W' in input_tensor_names[1]

                        s_ClassOutput_b_name = input_tensor_names[0]
                        s_ClassOutput_W_name = input_tensor_names[1]

                        s_ClassOutput_b = ckpt_reader.get_tensor(s_ClassOutput_b_name)
                        s_ClassOutput_W = ckpt_reader.get_tensor(s_ClassOutput_W_name)

                        for new_category_idx in range(category_idx - self.nb_cl + 1, category_idx + 1):
                            most_confused_with = \
                                Counter(pred_y[np.argmax(data_y_new_classes, axis=1) == new_category_idx]).most_common(
                                    1)[
                                    0][0]

                            tmp_tensor = s_ClassOutput_b[most_confused_with]
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
                            s_ClassOutput_b = np.append(s_ClassOutput_b, tmp_tensor)

                            tmp_tensor = np.expand_dims(s_ClassOutput_W[:, most_confused_with], axis=1)
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
                            s_ClassOutput_W = np.concatenate((s_ClassOutput_W, tmp_tensor), axis=1)

                        # assign value
                        input_var_list = [var for var in tf.trainable_variables() if s_ClassOutput_W_name in var.name]
                        assert len(input_var_list) == 1
                        self.sess.run(tf.assign(input_var_list[0], s_ClassOutput_W))

                        input_var_list = [var for var in tf.trainable_variables() if s_ClassOutput_b_name in var.name]
                        assert len(input_var_list) == 1
                        self.sess.run(tf.assign(input_var_list[0], s_ClassOutput_b))

            gen = get_train_inf(data_X, data_y, train_weights)

            pre_iters = 0

            history_conf_mat_dict = dict()
            for task_class_num in range(self.nb_cl, category_idx + self.nb_cl, self.nb_cl):
                history_conf_mat_dict['class_1-%d' % task_class_num] = dict()

            '''
            Generator
            '''
            # reset the cache of the plot
            gan.tflib.plot.reset()

            for iteration in range(self.iters):
                start_time = time.time()
                # Train generator
                if iteration > pre_iters:
                    _, _gen_cost = self.sess.run([self.gen_train_op, self.gen_cost])

                for _ in range(self.critic_iters):
                    _data_X, _, _ = gen.next()
                    _disc_cost, _ = \
                        self.sess.run([self.disc_cost, self.disc_train_op],
                                      feed_dict={self.real_data_int_G: _data_X})

                lib.plot.plot('time (generator)', time.time() - start_time)
                if iteration > pre_iters:
                    lib.plot.plot('train gen cost', _gen_cost)
                lib.plot.plot('train disc cost', _disc_cost)

                if iteration > pre_iters:
                    print("iter {}: disc: {}\tgen: {}\ttime: {}"
                          .format(iteration + 1, _disc_cost, _gen_cost, time.time() - start_time))
                else:
                    print("iter {}: disc: {}\ttime: {}"
                          .format(iteration + 1, _disc_cost, time.time() - start_time))

                # Calculate dev loss and generate samples every 100 iters
                if (iteration + 1) % self.test_interval == 0:
                    dev_disc_costs = []
                    for images, _ in get_test_epoch(test_X, test_y):
                        _dev_disc_cost = self.sess.run(self.disc_cost, feed_dict={self.real_data_int_G: images})
                        dev_disc_costs.append(_dev_disc_cost)
                    lib.plot.plot('dev disc cost', np.mean(dev_disc_costs))
                    self.generate_image(iteration + 1, train_log_dir_for_cur_class)

                # Save checkpoint
                if (iteration + 1) % self.save_interval == 0:
                    self.save(iteration + 1, category_idx)

                # Save logs every 100 iters
                if (iteration + 1) % 100 == 0:
                    gan.tflib.plot.flush(train_log_dir_for_cur_class, 'log_generator.pkl')

                gan.tflib.plot.tick()

            '''
            Solver
            '''

            # reset the cache of the plot
            gan.tflib.plot.reset()

            for iteration in range(self.solver_iters):
                start_time = time.time()

                _data_X, _data_y, _weights = gen.next()
                _solver_cost, _ = self.sess.run([self.solver_cost, self.solver_train_op], feed_dict={
                    self.real_data_int_S: _data_X,
                    self.real_y_S: _data_y,
                    self.train_batch_weights: _weights
                })

                lib.plot.plot('time (solver)', time.time() - start_time)
                lib.plot.plot('solver cost', _solver_cost)
                print('iter {}: solver cost: {}\ttime: {}'.format(iteration, _solver_cost, time.time() - start_time))

                if (iteration + 1) % self.test_interval == 0:
                    if self.dataset_name == 'svhn':
                        pred_logits = []
                        for pred_y_idx in range(0, len(test_X), 1000):
                            pred_logits_batch = self.classify_for_logits(test_X[pred_y_idx: pred_y_idx + 1000])
                            pred_logits.extend(pred_logits_batch)
                    else:
                        raise Exception()
                    pred_logits = np.array(pred_logits)
                    pred_y = np.argmax(pred_logits, axis=1)

                    # deprecated for svhn
                    # _test_accu = np.sum(pred_y == np.argmax(test_y, 1)) / float(len(test_X))

                    # confusion matrix and accuracy per class
                    _test_conf_mat = confusion_matrix(pred_y, np.argmax(test_y, 1))
                    _test_accu_per_class = np.diag(_test_conf_mat) * 1. / np.sum(_test_conf_mat, axis=0)

                    _test_accu = np.mean(_test_accu_per_class)
                    lib.plot.plot('test accuracy', _test_accu)

                    print('Test accuracy: {}'.format(_test_accu))
                    print("Test accuracy: " + " | ".join(str(o) for o in _test_accu_per_class))

                    gan.tflib.plot.flush(train_log_dir_for_cur_class, 'log_solver.pkl')

                    history_conf_mat_dict['class_1-%d' % (category_idx + 1)][iteration + 1] = _test_conf_mat

                    # old task acc & new forgetting rate
                    for old_task_class_num in range(self.nb_cl, category_idx, self.nb_cl):
                        test_indices_old_task = np.argmax(test_y, axis=1) < old_task_class_num
                        pred_y_old_task = np.argmax(pred_logits[test_indices_old_task, :old_task_class_num], axis=1)
                        test_y_old_task = test_y[test_indices_old_task]
                        test_conf_mat_old_task = confusion_matrix(pred_y_old_task, np.argmax(test_y_old_task, 1))
                        history_conf_mat_dict['class_1-%d' % old_task_class_num][iteration + 1] = test_conf_mat_old_task
                        test_acc_old_task = np.sum(pred_y_old_task == np.argmax(test_y_old_task, axis=1)) / float(
                            len(pred_y_old_task))
                        print("Test accuracy (1-{}): {}".format(old_task_class_num, test_acc_old_task))

                gan.tflib.plot.tick()

            '''
            Saver
            '''
            # final save checkpoint
            self.save(iteration + 1, category_idx, final=True)

            # save history conf mat
            for key in history_conf_mat_dict:
                with open(os.path.join(train_log_dir_for_cur_class, '%s_conf_mat.pkl' % key), 'wb') as fout:
                    pickle.dump(history_conf_mat_dict[key], fout)

        # get fids
        cond_fid_file = os.path.join(train_log_dir_for_cur_class, 'cond_fid.pkl')
        if not os.path.exists(cond_fid_file):
            history_cond_fid = dict()
            fid_vals = self.get_fid(train_log_dir_for_cur_class)
            history_cond_fid[self.iters] = fid_vals

            # save history cond fid
            with open(cond_fid_file, 'wb') as fout:
                pickle.dump(history_cond_fid, fout)


    @property
    def model_dir(self):
        finetune_str = (('finetune_improved' + ('_v2' if self.improved_finetune_type == 'v2' else '') + (
            '_noise_%.1f' % self.improved_finetune_noise_level if self.improved_finetune_noise else '')) if
                        self.improved_finetune else 'finetune') if self.finetune else 'from_scratch'
        mode_str = self.mode + '_critic_%d' % self.critic_iters
        return os.path.join(self.result_dir, self.dataset_name + '_order_%d' % self.order_idx, 'nb_cl_%d' % self.nb_cl,
                            mode_str, str(self.adam_lr) + '_' + str(self.adam_beta1) + '_' + str(self.adam_beta2),
                            str(self.iters),
                            str(self.solver_iters) + '_' + str(self.solver_adam_lr) + '_' + str(self.dgr_ratio),
                            finetune_str)

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

    def check_model(self, category_idx, step=-1):
        """
        Check whether the old models(which<category_idx) exist
        :param category_idx:
        :return: True or false
        """
        print(" [*] Checking checkpoints for class %d" % (category_idx + 1))
        if step == -1:
            checkpoint_dir = os.path.join(self.model_dir_for_class(category_idx), "checkpoints", "final")
        else:
            checkpoint_dir = os.path.join(self.model_dir_for_class(category_idx), "checkpoints", str(step))

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            return True
        else:
            return False
