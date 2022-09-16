# -*- coding:utf-8 -*-
'''
On LL-GAN v1.3
move the embedding one layer ahead
'''
import time

import tensorflow as tf

tf.set_random_seed(1993)

import numpy as np
from tqdm import tqdm

np.random.seed(1993)
import os
import pprint

from gan.model_svhn_introgan import ProtoGAN

from utils import fid
from utils.visualize_result_single import vis_acc_and_fid

flags = tf.app.flags

flags.DEFINE_string("dataset", "svhn", "The name of dataset [svhn]")
flags.DEFINE_boolean("rgb", True, "rgb or gray")

# Hyperparameters
flags.DEFINE_integer("lambda_param", 10, "Gradient penalty lambda hyperparameter [10]")
flags.DEFINE_integer("critic_iters", 1,
                     "How many critic iterations per generator iteration [1 for DCGAN and 5 for WGAN-GP]")
flags.DEFINE_integer("batch_size", 100, "The size of batch images")
flags.DEFINE_integer("iters", 10000, "How many generator iters to train")
flags.DEFINE_integer("output_dim", 3 * 32 * 32, "Number of pixels in MNIST/fashion-MNIST (3*32*32) [3072]")
flags.DEFINE_integer("dim", 64, "GAN dim")
flags.DEFINE_string("mode", 'dcgan', "Valid options are dcgan or wgan-gp")
flags.DEFINE_float("protogan_scale", 1.0, "")
flags.DEFINE_float("protogan_scale_g", 0.1, "")
flags.DEFINE_integer("gan_save_interval", 5000, 'interval to save a checkpoint(number of iters)')
flags.DEFINE_float("adam_lr", 2e-4, 'default: 1e-4, 2e-4, 3e-4')
flags.DEFINE_float("adam_beta1", 0.5, 'default: 0.5')
flags.DEFINE_float("adam_beta2", 0.999, 'default: 0.999')
flags.DEFINE_float("proto_weight_real", 0., "")
flags.DEFINE_float("proto_weight_fake", 0., "")
flags.DEFINE_integer("proto_importance", 1, "relative importance of protos versus ordinary samples")
flags.DEFINE_string("dist_func", 'squared_l2', "cosine, squared_l2")
flags.DEFINE_float("gamma", 1e-2, 'smoothness of the output probabilities')
flags.DEFINE_string("center_type", 'rel_center', '[rel_center/fixed_center/multi_center]')
flags.DEFINE_integer("fixed_center_idx", 0, '[0, proto_num - 1]')
flags.DEFINE_integer("proto_num", 20, "")
flags.DEFINE_float("margin", 0., "")
flags.DEFINE_string("proto_select_criterion", 'random', "random or ori_kmeans or feat_kmeans")
flags.DEFINE_boolean("finetune", True, 'if gan finetuned from the previous model')
flags.DEFINE_boolean("improved_finetune", True, 'if gan finetuned from the previous model')
flags.DEFINE_string("improved_finetune_type", 'v2', '')
flags.DEFINE_boolean("improved_finetune_noise", True, 'use the same weight or add some variation')
flags.DEFINE_float("improved_finetune_noise_level", 0.5, 'noise level')
flags.DEFINE_boolean("classification_only", False, "True for classification only")
flags.DEFINE_integer("num_samples_per_class", -1, "number of samples per class in the training set")
flags.DEFINE_boolean("exemplars_dual_use", True, "if the exemplars also be added to the training set")
flags.DEFINE_string("anti_imbalance", 'none', "[reweight|oversample|none]")
flags.DEFINE_string("rigorous", 'min', '[min|max|random]')
flags.DEFINE_boolean("train_rel_center", False, "")

# Add how many classes every time
flags.DEFINE_integer('nb_cl', 2, '')

# DEBUG
flags.DEFINE_integer('from_class_idx', 0, 'starting category_idx')
flags.DEFINE_integer('to_class_idx', 9, 'ending category_idx')
flags.DEFINE_integer('order_idx', 1, 'class orders [1~5]')
flags.DEFINE_integer("test_interval", 500,
                     "test interval (since the total training consists of 10,000 iters, we need 20 points, so 10,000 / 20 = 500)")

# Visualize
flags.DEFINE_boolean('vis_result', True, 'visualize accuracy and fid figure')

FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()


def main(_):
    start_time = time.time()
    assert FLAGS.fixed_center_idx < FLAGS.proto_num

    if 0 < FLAGS.num_samples_per_class < FLAGS.proto_num:
        FLAGS.proto_num = FLAGS.num_samples_per_class

    pp.pprint(flags.FLAGS.__flags)

    NUM_CLASSES = 10

    from utils import svhn
    raw_images_train, train_labels, train_one_hot_labels, order = svhn.load_data(kind='train',
                                                                                 order_idx=FLAGS.order_idx,
                                                                                 num_samples_per_class=FLAGS.num_samples_per_class)
    raw_images_test, test_labels, test_one_hot_labels, order = svhn.load_data(kind='test', order_idx=FLAGS.order_idx)

    # Total training samples
    NUM_TRAIN_SAMPLES_TOTAL = len(raw_images_train)
    NUM_TEST_SAMPLES_TOTAL = len(raw_images_test)

    NUM_TRAIN_SAMPLES_PER_CLASS = NUM_TRAIN_SAMPLES_TOTAL / NUM_CLASSES

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    method_name = '_'.join(os.path.basename(__file__).split('.')[0].split('_')[2:])
    result_dir = os.path.join('result', method_name)
    FLAGS.result_dir = result_dir
    print('Result dir: %s' % result_dir)

    graph_fid = tf.Graph()
    with graph_fid.as_default():
        inception_path = fid.check_or_download_inception('tmp/imagenet')
        fid.create_inception_graph(inception_path)
    sess_fid = tf.Session(config=run_config, graph=graph_fid)

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
                               category_idx <= train_labels[idx] <= to_category_idx]
        test_indices = [idx for idx in range(NUM_TEST_SAMPLES_TOTAL) if
                        test_labels[idx] <= to_category_idx]

        train_x = raw_images_train[train_indices_reals, :]
        train_y = train_one_hot_labels[train_indices_reals, :(to_category_idx + 1)]
        test_x = raw_images_test[test_indices, :]
        test_y = test_one_hot_labels[test_indices, :(to_category_idx + 1)]

        '''
        Check model exist
        '''
        if ProtoGAN.check_model(FLAGS, to_category_idx):
            model_exist = True
            print(" [*] Model of Class %d-%d exists. Skip the training process" % (
                category_idx + 1, to_category_idx + 1))
        else:
            model_exist = False
            print(" [*] Model of Class %d-%d does not exist. Start the training process" % (
                category_idx + 1, to_category_idx + 1))

        '''
        Train generative model(DC-GAN)
        '''
        # Mixed with generated samples of old classes
        num_old_classes = to_category_idx + 1 - FLAGS.nb_cl
        if num_old_classes > 0:  # first session or not
            if not model_exist:
                train_y_gens = np.repeat(range(num_old_classes), [NUM_TRAIN_SAMPLES_PER_CLASS])
                train_y_gens_one_hot = np.eye(num_old_classes)[train_y_gens.reshape(-1)]
                llgan_obj.load(category_idx - 1)

                # train_x_gens, _, _ = llgan_obj.test(NUM_TRAIN_SAMPLES_PER_CLASS * num_old_classes, train_y_gens_one_hot)

                train_x_gens = []
                for start_idx in tqdm(range(0, len(train_y_gens_one_hot), FLAGS.batch_size), desc='Generative replay'):
                    train_y_gens_one_hot_cur_batch = train_y_gens_one_hot[start_idx: start_idx + FLAGS.batch_size]
                    train_x_gens_batch, _, _ = llgan_obj.test(len(train_y_gens_one_hot_cur_batch),
                                                              train_y_gens_one_hot_cur_batch)
                    train_x_gens.extend(train_x_gens_batch)

                train_x = np.concatenate((train_x, np.uint8(train_x_gens)))
                train_y = np.concatenate((train_y, np.eye(to_category_idx + 1)[train_y_gens.reshape(-1)]))

            sess_gan.close()
            del sess_gan

        graph_gen = tf.Graph()
        sess_gan = tf.Session(config=run_config, graph=graph_gen)

        llgan_obj = ProtoGAN(sess_gan, graph_gen, sess_fid,
                             dataset=FLAGS.dataset,
                             mode=FLAGS.mode,
                             batch_size=FLAGS.batch_size,
                             output_dim=FLAGS.output_dim,
                             lambda_param=FLAGS.lambda_param,
                             critic_iters=FLAGS.critic_iters,
                             iters=FLAGS.iters,
                             result_dir=FLAGS.result_dir,
                             checkpoint_interval=FLAGS.gan_save_interval,
                             adam_lr=FLAGS.adam_lr,
                             adam_beta1=FLAGS.adam_beta1,
                             adam_beta2=FLAGS.adam_beta2,
                             protogan_scale=FLAGS.protogan_scale,
                             protogan_scale_g=FLAGS.protogan_scale_g,
                             finetune=FLAGS.finetune,
                             improved_finetune=FLAGS.improved_finetune,
                             nb_cl=FLAGS.nb_cl,
                             nb_output=(to_category_idx + 1),
                             classification_only=FLAGS.classification_only,
                             proto_weight_real=FLAGS.proto_weight_real,
                             proto_weight_fake=FLAGS.proto_weight_fake,
                             dist_func=FLAGS.dist_func,
                             gamma=FLAGS.gamma,
                             proto_num=FLAGS.proto_num,
                             margin=FLAGS.margin,
                             order_idx=FLAGS.order_idx,
                             proto_select_criterion=FLAGS.proto_select_criterion,
                             order=order,
                             test_interval=FLAGS.test_interval,
                             proto_importance=FLAGS.proto_importance,
                             improved_finetune_type=FLAGS.improved_finetune_type,
                             improved_finetune_noise=FLAGS.improved_finetune_noise,
                             improved_finetune_noise_level=FLAGS.improved_finetune_noise_level,
                             dim=FLAGS.dim,
                             rgb=FLAGS.rgb,
                             center_type=FLAGS.center_type,
                             fixed_center_idx=FLAGS.fixed_center_idx,
                             num_samples_per_class=FLAGS.num_samples_per_class,
                             exemplars_dual_use=FLAGS.exemplars_dual_use,
                             anti_imbalance=FLAGS.anti_imbalance,
                             train_rel_center=FLAGS.train_rel_center,
                             rigorous=FLAGS.rigorous)

        print(ProtoGAN.model_dir_static(FLAGS))

        '''
        Train generative model(GAN)
        '''
        llgan_obj.train(train_x, train_y, test_x, test_y, to_category_idx, model_exist=model_exist)

        if FLAGS.vis_result:
            vis_acc_and_fid(llgan_obj.model_dir, FLAGS.dataset, FLAGS.nb_cl, test_interval=FLAGS.test_interval,
                            num_iters=FLAGS.iters, vis_fid=(not FLAGS.classification_only))

    sess_fid.close()
    stop_time = time.time()
    print('TOTAL RUNNING TIME: %f' % (stop_time - start_time))
    with open(os.path.join(ProtoGAN.model_dir_static(FLAGS), 'running_time.txt'), 'w') as fout:
        fout.write('%f' % (stop_time - start_time))


if __name__ == '__main__':
    tf.app.run()
