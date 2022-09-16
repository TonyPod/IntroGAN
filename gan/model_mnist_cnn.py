import os
import pickle
import time
from collections import Counter

import matplotlib as mpl
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

import gan.tflib as lib
import gan.tflib.inception_score
import gan.tflib.ops
import gan.tflib.ops.batchnorm
import gan.tflib.ops.conv2d
import gan.tflib.ops.deconv2d
import gan.tflib.ops.linear
import gan.tflib.plot
import gan.tflib.save_images

mpl.use('Agg')
import matplotlib.pyplot as plt

from utils.visualize_embedding_protos_and_samples import colors
from sklearn.manifold import TSNE
from umap import UMAP


def leaky_relu(x, alpha=0.2):
    return tf.maximum(alpha * x, x)


class ClsNet(object):

    def __init__(self, sess, graph, dataset_name, batch_size, output_dim, iters, result_dir,
                 adam_lr, adam_beta1, adam_beta2, nb_cl, nb_output, order_idx, order, test_interval,
                 finetune, improved_finetune, improved_finetune_type, improved_finetune_noise,
                 improved_finetune_noise_level):

        self.sess = sess
        self.graph = graph

        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.output_dim = output_dim

        self.iters = iters
        self.result_dir = result_dir

        self.adam_lr = adam_lr
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2

        self.nb_cl = nb_cl
        self.nb_output = nb_output
        self.is_first_session = (self.nb_output == self.nb_cl)

        self.order_idx = order_idx
        self.order = order

        self.test_interval = test_interval

        self.finetune = finetune
        self.improved_finetune = improved_finetune
        self.improved_finetune_noise = improved_finetune_noise
        self.improved_finetune_type = improved_finetune_type
        self.improved_finetune_noise_level = improved_finetune_noise_level

        self.build_model()

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def build_model(self):

        lib.delete_all_params()

        with tf.variable_scope("gan") as scope, self.graph.as_default():
            # placeholder for MNIST samples
            self.real_data_int = tf.placeholder(tf.int32, shape=[self.batch_size, self.output_dim])
            self.real_y = tf.placeholder(tf.float32, shape=[None, self.nb_output])

            real_data = 2 * ((tf.cast(self.real_data_int, tf.float32) / 255.) - .5)

            disc_real_class = self.classifier(real_data, reuse=False, is_training=True)[0]

            # Get output label
            self.inputs_int = tf.placeholder(tf.int32, shape=[None, self.output_dim])
            inputs = 2 * ((tf.cast(self.inputs_int, tf.float32) / 255.) - .5)
            self.class_logits, self.embedding_inputs = self.classifier(inputs, reuse=True, is_training=False)
            self.pred_y = tf.argmax(self.class_logits, 1)

            self.real_accu = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(disc_real_class, 1), tf.argmax(self.real_y, 1)), dtype=tf.float32))
            self.class_cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=disc_real_class, labels=self.real_y))

            var_list = tf.trainable_variables()

            self.class_train_op = \
                tf.train.AdamOptimizer(learning_rate=self.adam_lr,
                                       beta1=self.adam_beta1, beta2=self.adam_beta2) \
                    .minimize(self.class_cost, var_list=var_list)

            bn_moving_vars = [var for var in tf.global_variables() if 'moving_mean' in var.name]
            bn_moving_vars += [var for var in tf.global_variables() if 'moving_variance' in var.name]
            var_list += bn_moving_vars

            self.saver = tf.train.Saver(var_list=var_list)  # var_list doesn't contain Adam params

            if self.finetune:
                var_list_for_finetune = [var for var in var_list if 'c_ClassOutput' not in var.name]
                self.saver_for_finetune = tf.train.Saver(var_list=var_list_for_finetune)

    def classifier(self, inputs, reuse, is_training):

        with tf.variable_scope('Classifier') as scope:
            if reuse:
                scope.reuse_variables()

            output = tf.reshape(inputs, [-1, 1, 28, 28])

            output = lib.ops.conv2d.Conv2D('c_1', 1, 64, 5, output, stride=2)
            output = leaky_relu(output)

            output = lib.ops.conv2d.Conv2D('c_2', 64, 2 * 64, 5, output, stride=2)
            output = leaky_relu(output)

            output = lib.ops.conv2d.Conv2D('c_3', 2 * 64, 4 * 64, 5, output, stride=2)

            output = tf.reshape(output, [-1, 4 * 4 * 4 * 64])

            embeddingOutput = output
            output = leaky_relu(output)

            classOutput = gan.tflib.ops.linear.Linear('c_ClassOutput', 4 * 4 * 4 * 64, self.nb_output, output)

            return tf.reshape(classOutput, shape=[-1, self.nb_output]), embeddingOutput

    def classify(self, inputs):

        with self.graph.as_default():
            result = self.sess.run(self.pred_y, feed_dict={self.inputs_int: inputs})

        return result

    def classify_for_logits(self, inputs):

        with self.graph.as_default():
            logits = self.sess.run(self.class_logits, feed_dict={self.inputs_int: inputs})

        return logits

    def get_fid(self, train_log_dir_for_cur_class, train_x, train_y, sess_fid):

        # import
        from PIL import Image
        from utils.fid import calculate_fid_given_paths_with_sess

        print('Calculating fid...')
        time_start = time.time()
        fid_vals = {}

        temp_folder = os.path.join(train_log_dir_for_cur_class, 'fid_temp')

        for category_idx in range(self.nb_output):
            sub_folder = os.path.join(temp_folder, 'class_%d' % (category_idx + 1))
            if not os.path.exists(sub_folder):
                os.makedirs(sub_folder)
            x = train_x[np.argmax(train_y, axis=1) == category_idx]
            for i, x_single in enumerate(x):
                img = Image.fromarray(
                    np.repeat(x_single.astype('uint8').reshape((1, 28, 28)).transpose((1, 2, 0)), 3, axis=2))
                img.save(os.path.join(sub_folder, '%d.jpg' % (i + 1)))
            fid_val = calculate_fid_given_paths_with_sess(sess_fid,
                                                          [sub_folder,
                                                           'precalc_fids/%s/fid_stats_%d.npz' % (
                                                               self.dataset_name, self.order[category_idx] + 1)])
            fid_vals[category_idx + 1] = fid_val

        # delete temp folders
        # shutil.rmtree(temp_folder)

        time_stop = time.time()
        print('Total: %.2f seconds' % (time_stop - time_start))

        return fid_vals

    def train(self, data_X_cumul, data_y_cumul, test_X_cumul, test_y_cumul, category_idx, model_exist=False,
              save_training_set=False, sess_fid=None, calc_fid=False):

        train_log_dir_for_cur_class = self.model_dir_for_class(category_idx)

        if not os.path.exists(train_log_dir_for_cur_class):
            os.makedirs(train_log_dir_for_cur_class)

        # get fids of the training set
        if (save_training_set and sess_fid is not None) and calc_fid:
            cond_fid_file = os.path.join(train_log_dir_for_cur_class, 'cond_fid.pkl')
            if not os.path.exists(cond_fid_file):
                history_cond_fid = dict()
                fid_vals = self.get_fid(train_log_dir_for_cur_class, data_X_cumul, data_y_cumul, sess_fid)
                history_cond_fid[self.iters] = fid_vals

                # save history cond fid
                with open(cond_fid_file, 'wb') as fout:
                    pickle.dump(history_cond_fid, fout)

        if model_exist:
            self.load(category_idx)
        else:
            with self.graph.as_default():
                self.sess.run(tf.initialize_all_variables())
                if self.finetune and not self.is_first_session:
                    _, _, ckpt_path = self.load_finetune(category_idx - self.nb_cl)

                    # special initialized for g_Input
                    if self.improved_finetune:
                        # calc confusion matrix on the training set (only new classes): new classes -> old classes
                        pred_y = []
                        indices_new_classes = np.argmax(data_y_cumul, axis=1) >= category_idx - self.nb_cl + 1
                        data_X_new_classes = data_X_cumul[indices_new_classes]
                        data_y_new_classes = data_y_cumul[indices_new_classes]
                        for sample_idx in range(0, len(data_X_new_classes), self.batch_size):
                            pred_logits_batch = self.classify_for_logits(
                                data_X_new_classes[sample_idx:sample_idx + self.batch_size])
                            pred_y_batch = np.argmax(pred_logits_batch[:, :category_idx - self.nb_cl + 1], axis=1)
                            pred_y.extend(pred_y_batch)
                        pred_y = np.array(pred_y)

                        # get the input tensor
                        ckpt_reader = tf.pywrap_tensorflow.NewCheckpointReader(ckpt_path)
                        input_tensor_names = [key for key in ckpt_reader.get_variable_to_shape_map().keys() if
                                              'c_ClassOutput' in key]
                        assert len(input_tensor_names) == 2
                        g_Input_offset = 0

                        assert 'c_ClassOutput.W' in input_tensor_names[0]
                        assert 'c_ClassOutput.b' in input_tensor_names[1]

                        c_ClassOutput_W_name = input_tensor_names[0]
                        c_ClassOutput_b_name = input_tensor_names[1]

                        c_ClassOutput_W = ckpt_reader.get_tensor(c_ClassOutput_W_name)
                        c_ClassOutput_b = ckpt_reader.get_tensor(c_ClassOutput_b_name)

                        for new_category_idx in range(category_idx - self.nb_cl + 1, category_idx + 1):
                            most_confused_with = \
                                Counter(pred_y[np.argmax(data_y_new_classes, axis=1) == new_category_idx]).most_common(
                                    1)[
                                    0][0]

                            tmp_tensor = np.expand_dims(c_ClassOutput_W[:, most_confused_with], axis=1)
                            if self.improved_finetune_noise:
                                if self.improved_finetune_type == 'v1':
                                    noise_tensor = np.random.normal(0, (np.max(tmp_tensor) - np.min(
                                        tmp_tensor)) / 6 * self.improved_finetune_noise_level,
                                                                    tmp_tensor.shape)  # 3 sigma
                                elif self.improved_finetune_type == 'v2':
                                    noise_tensor = np.random.normal(0, np.std(
                                        tmp_tensor) * self.improved_finetune_noise_level,
                                                                    tmp_tensor.shape)
                                else:
                                    raise Exception()
                                tmp_tensor += noise_tensor
                            c_ClassOutput_W = np.concatenate((c_ClassOutput_W, tmp_tensor), axis=1)

                            tmp_tensor = c_ClassOutput_b[most_confused_with]
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
                            c_ClassOutput_b = np.append(c_ClassOutput_b, tmp_tensor)

                        # assign value
                        input_var_list = [var for var in tf.trainable_variables() if c_ClassOutput_W_name in var.name]
                        assert len(input_var_list) == 1
                        self.sess.run(tf.assign(input_var_list[0], c_ClassOutput_W))

                        input_var_list = [var for var in tf.trainable_variables() if c_ClassOutput_b_name in var.name]
                        assert len(input_var_list) == 1
                        self.sess.run(tf.assign(input_var_list[0], c_ClassOutput_b))

            def get_train_inf(data_X, data_y):
                while True:
                    batch_idxs = len(data_X) // self.batch_size
                    if batch_idxs == 0:
                        reorder = np.random.choice(range(len(data_X)), self.batch_size, replace=True)
                        _data_X = data_X[reorder]
                        _data_y = data_y[reorder]
                        yield (_data_X, _data_y)
                    else:
                        reorder = np.array(range(len(data_X)))
                        np.random.shuffle(reorder)
                        data_X = data_X[reorder]
                        data_y = data_y[reorder]
                        for idx in range(0, batch_idxs):
                            _data_X = data_X[idx * self.batch_size:(idx + 1) * self.batch_size]
                            _data_y = data_y[idx * self.batch_size:(idx + 1) * self.batch_size]
                            yield (_data_X, _data_y)

            gen = get_train_inf(data_X_cumul, data_y_cumul)

            history_conf_mat_dict = dict()
            for task_class_num in range(self.nb_cl, category_idx + self.nb_cl, self.nb_cl):
                history_conf_mat_dict['class_1-%d' % task_class_num] = dict()

            for iteration in range(self.iters):
                start_time = time.time()
                _data_X, _data_y = gen.next()
                _class_cost, _ = \
                    self.sess.run([self.class_cost, self.class_train_op],
                                  feed_dict={self.real_data_int: _data_X,
                                             self.real_y: _data_y})

                if (iteration + 1) % 100 == 0:
                    print("iter {}: class: {}\ttime: {}".format(iteration + 1, _class_cost, time.time() - start_time))

                # Calculate dev loss and generate samples every 100 iters
                if (iteration + 1) % self.test_interval == 0:
                    if self.dataset_name == 'fashion-mnist':
                        test_X_num_per_class = 1000
                        assert len(test_X_cumul) == (category_idx + 1) * test_X_num_per_class
                        pred_logits = []
                        for old_category_idx in range(category_idx + 1):
                            pred_logits_batch = self.classify_for_logits(test_X_cumul[
                                                                         test_X_num_per_class * old_category_idx: test_X_num_per_class * (
                                                                                 old_category_idx + 1)])
                            pred_logits.extend(pred_logits_batch)
                    elif self.dataset_name == 'mnist':
                        pred_logits = []
                        for pred_y_idx in range(0, len(test_X_cumul), 1000):
                            pred_logits_batch = self.classify_for_logits(test_X_cumul[pred_y_idx: pred_y_idx + 1000])
                            pred_logits.extend(pred_logits_batch)
                    else:
                        raise Exception()
                    pred_logits = np.array(pred_logits)
                    pred_y = np.argmax(pred_logits, axis=1)

                    # deprecated for MNIST
                    # _test_accu = np.sum(pred_y == np.argmax(test_y_cumul, 1)) / float(len(test_X_cumul))
                    # confusion matrix and accuracy per class
                    _test_conf_mat = confusion_matrix(pred_y, np.argmax(test_y_cumul, 1))
                    _test_accu_per_class = np.diag(_test_conf_mat) * 1. / np.sum(_test_conf_mat, axis=0)

                    _test_accu = np.mean(_test_accu_per_class)

                    print('Test accuracy: {}'.format(_test_accu))
                    print("Test accuracy: " + " | ".join(str(o) for o in _test_accu_per_class))

                    history_conf_mat_dict['class_1-%d' % (category_idx + 1)][iteration + 1] = _test_conf_mat

                    # old task acc & new forgetting rate
                    for old_task_class_num in range(self.nb_cl, category_idx, self.nb_cl):
                        test_indices_old_task = np.argmax(test_y_cumul, axis=1) < old_task_class_num
                        pred_y_old_task = np.argmax(pred_logits[test_indices_old_task, :old_task_class_num], axis=1)
                        test_y_old_task = test_y_cumul[test_indices_old_task]
                        test_conf_mat_old_task = confusion_matrix(pred_y_old_task, np.argmax(test_y_old_task, 1))
                        history_conf_mat_dict['class_1-%d' % old_task_class_num][iteration + 1] = test_conf_mat_old_task
                        test_acc_old_task = np.sum(pred_y_old_task == np.argmax(test_y_old_task, axis=1)) / float(
                            len(pred_y_old_task))
                        print("Test accuracy (1-{}): {}".format(old_task_class_num, test_acc_old_task))

            # final save checkpoint
            self.save(iteration + 1, category_idx, final=True)

            # save history conf mat
            for key in history_conf_mat_dict:
                with open(os.path.join(train_log_dir_for_cur_class, '%s_conf_mat.pkl' % key), 'wb') as fout:
                    pickle.dump(history_conf_mat_dict[key], fout)

        USE_TSNE = False
        embedding_real_samples_protos_file = os.path.join(train_log_dir_for_cur_class,
                                                          'tsne.pdf' if USE_TSNE else 'umap.pdf')
        if not os.path.exists(embedding_real_samples_protos_file):
            print('Get tsne...')
            start_time = time.time()
            all_embedding_samples = np.zeros([0, 4096], np.float)
            num_sample_per_class_arr = []

            # class
            for category_idx in range(self.nb_output):
                test_X_cur_class = test_X_cumul[np.argmax(test_y_cumul, axis=1) == category_idx]
                embedding_samples = self.sess.run(self.embedding_inputs,
                                                  feed_dict={self.inputs_int: test_X_cur_class})
                all_embedding_samples = np.concatenate((all_embedding_samples, embedding_samples))
                num_sample_per_class_arr.append(len(embedding_samples))

            if USE_TSNE:
                tsne = TSNE(n_components=2)
                dim_reduction_result = tsne.fit_transform(all_embedding_samples)
            else:
                tsne = UMAP()
                dim_reduction_result = tsne.fit_transform(all_embedding_samples)
            plt.figure(figsize=(6, 6), dpi=150)
            for category_idx in range(self.nb_output):
                tsne_samples = dim_reduction_result[np.sum(num_sample_per_class_arr[:category_idx],
                                                           dtype=int): np.sum(
                    num_sample_per_class_arr[:category_idx + 1],
                    dtype=int)]
                plt.scatter(tsne_samples[:, 0], tsne_samples[:, 1], marker='.', alpha=1.,
                            color=colors[category_idx],
                            label='Class %d' % (category_idx + 1))

            plt.legend()
            plt.savefig(embedding_real_samples_protos_file)
            plt.close()

            print('Time used: %.2f' % (time.time() - start_time))

    @property
    def model_dir(self):
        finetune_str = (('finetune_improved' + ('_v2' if self.improved_finetune_type == 'v2' else '') + (
            '_noise_%.1f' % self.improved_finetune_noise_level if self.improved_finetune_noise else '')) if self.improved_finetune else 'finetune') if self.finetune else 'from_scratch'
        return os.path.join(self.result_dir, self.dataset_name + '_order_%d' % self.order_idx, 'nb_cl_%d' % self.nb_cl,
                            str(self.adam_lr) + '_' + str(self.adam_beta1) + '_' + str(self.adam_beta2),
                            str(self.iters), finetune_str)

    def model_dir_for_class(self, category_idx):
        return os.path.join(self.model_dir, 'class_%d-%d' % (1, category_idx + 1))

    def save(self, step, category_idx, final=False):
        model_name = "cnn.model"
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
