# -*- coding:utf-8 -*-

import tensorflow as tf
tf.set_random_seed(1993)

import numpy as np
np.random.seed(1993)
import os
import pprint

from gan.model_svhn_cnn import ClsNet

from utils.visualize_result_single import vis_acc_and_fid

flags = tf.app.flags

flags.DEFINE_string("dataset", "svhn", "The name of dataset [svhn]")

# Hyperparameters
flags.DEFINE_integer("batch_size", 100, "The size of batch images")
flags.DEFINE_integer("iters", 10000, "How many generator iters to train")
flags.DEFINE_integer("output_dim", 3*32*32, "Number of pixels in MNIST/fashion-MNIST (3*32*32) [3072]")
flags.DEFINE_integer("dim", 64, "GAN dim")
flags.DEFINE_float("adam_lr", 2e-4, 'default: 1e-4, 2e-4, 3e-4')
flags.DEFINE_float("adam_beta1", 0.5, 'default: 0.0')
flags.DEFINE_float("adam_beta2", 0.999, 'default: 0.9')
flags.DEFINE_boolean("finetune", False, '')
flags.DEFINE_boolean("improved_finetune", True, '')
flags.DEFINE_boolean("improved_finetune_noise", True, 'use the same weight or add some variation')
flags.DEFINE_float("improved_finetune_noise_level", 0.5, 'noise level')

# Add how many classes every time
flags.DEFINE_integer('nb_cl', 2, '')

# DEBUG
flags.DEFINE_integer('from_class_idx', 0, 'starting category_idx')
flags.DEFINE_integer('to_class_idx', 9, 'ending category_idx')
flags.DEFINE_integer('order_idx', 1, 'class orders [1~5]')
flags.DEFINE_integer("test_interval", 500, "test interval (since the total training consists of 10,000 iters, we need 20 points, so 10,000 / 20 = 500)")

# Visualize
flags.DEFINE_boolean('vis_result', True, 'visualize accuracy and fid figure')

FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()


def main(_):

    pp.pprint(flags.FLAGS.__flags)

    from utils import svhn
    raw_images_train, train_labels, train_one_hot_labels, order = svhn.load_data(kind='train', order_idx=FLAGS.order_idx)
    raw_images_test, test_labels, test_one_hot_labels, order = svhn.load_data(kind='test', order_idx=FLAGS.order_idx)

    # Total training samples
    NUM_TRAIN_SAMPLES_TOTAL = len(raw_images_train)
    NUM_TEST_SAMPLES_TOTAL = len(raw_images_test)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    method_name = '_'.join(os.path.basename(__file__).split('.')[0].split('_')[2:])
    result_dir = os.path.join('result', method_name)
    print('Result dir: %s' % result_dir)

    '''
    Class Incremental Learning
    '''
    print('Starting from category ' + str(FLAGS.from_class_idx + 1) + ' to ' + str(FLAGS.to_class_idx + 1))
    print('Adding %d categories every time' % FLAGS.nb_cl)
    assert (FLAGS.from_class_idx % FLAGS.nb_cl == 0)
    for category_idx in range(FLAGS.from_class_idx, FLAGS.to_class_idx + 1, FLAGS.nb_cl):

        to_category_idx = category_idx + FLAGS.nb_cl - 1
        if FLAGS.nb_cl == 1:
            print('Adding Category ' + str(category_idx + 1))
        else:
            print('Adding Category %d-%d' % (category_idx + 1, to_category_idx + 1))

        train_indices_reals = [idx for idx in range(NUM_TRAIN_SAMPLES_TOTAL) if
                               train_labels[idx] <= to_category_idx]
        test_indices = [idx for idx in range(NUM_TEST_SAMPLES_TOTAL) if
                        test_labels[idx] <= to_category_idx]

        train_x = raw_images_train[train_indices_reals, :]
        train_y = train_one_hot_labels[train_indices_reals, :(to_category_idx+1)]
        test_x = raw_images_test[test_indices, :]
        test_y = test_one_hot_labels[test_indices, :(to_category_idx+1)]

        graph = tf.Graph()
        sess = tf.Session(config=run_config, graph=graph)
        llgan_obj = ClsNet(sess, graph,
                           dataset_name=FLAGS.dataset,
                           batch_size=FLAGS.batch_size,
                           output_dim=FLAGS.output_dim,
                           iters=FLAGS.iters,
                           result_dir=result_dir,
                           adam_lr=FLAGS.adam_lr,
                           adam_beta1=FLAGS.adam_beta1,
                           adam_beta2=FLAGS.adam_beta2,
                           nb_cl=FLAGS.nb_cl,
                           nb_output=(to_category_idx + 1),
                           order_idx=FLAGS.order_idx,
                           order=order,
                           test_interval=FLAGS.test_interval,
                           finetune=FLAGS.finetune,
                           improved_finetune=FLAGS.improved_finetune,
                           improved_finetune_noise=FLAGS.improved_finetune_noise,
                           improved_finetune_noise_level=FLAGS.improved_finetune_noise_level,
                           dim=FLAGS.dim)

        print(llgan_obj.model_dir)

        '''
        Train generative model(GAN)
        '''
        if llgan_obj.check_model(to_category_idx):
            model_exist = True
            print(" [*] Model of Class %d-%d exists. Skip the training process" % (category_idx + 1, to_category_idx + 1))
        else:
            model_exist = False
            print(" [*] Model of Class %d-%d does not exist. Start the training process" % (category_idx + 1, to_category_idx + 1))
        llgan_obj.train(train_x, train_y, test_x, test_y, to_category_idx, model_exist=model_exist)

        if FLAGS.vis_result:
            vis_acc_and_fid(llgan_obj.model_dir, FLAGS.dataset, FLAGS.nb_cl, test_interval=FLAGS.test_interval,
                            num_iters=FLAGS.iters, vis_fid=False)


if __name__ == '__main__':
    tf.app.run()
