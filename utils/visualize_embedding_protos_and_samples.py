# -*- coding:utf-8 -*-  

""" 
@time: 3/12/19 4:14 PM 
@author: Chen He 
@site:  
@file: proto_dim_reduction.py
@description:  
"""

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import numpy as np
import os
import pickle

num_classes_dict = {
    'fashion-mnist': 10,
    'svhn': 10,
    'imagenet_64x64_dogs': 30,
    'imagenet_64x64_birds': 30
}

test_intervals_dict = {
    'fashion-mnist': 500,
    'svhn': 500,
    'imagenet_64x64_dogs': 500,
    'imagenet_64x64_birds': 500
}

num_iters_dict = {
    'fashion-mnist': 10000,
    'svhn': 10000,
    'imagenet_64x64_dogs': 10000,
    'imagenet_64x64_birds': 10000
}

embedding_dim_dict = {
    'fashion-mnist': 1024,
    'svhn': 1024,
    'imagenet_64x64_dogs': 1024,
    'imagenet_64x64_birds': 1024
}

USE_KMEANS = False
NUM_REAL = 128

colors = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabebe',
          '#469990', '#e6beff', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9',
          '#ffffff', '#000000']


def visualize(folder_path, dataset_name, nb_cl, vis_pca=False, vis_tsne=True, embedding_dim=None, test_interval=None):
    num_classes = num_classes_dict[dataset_name]
    if test_interval is None:
        test_interval = test_intervals_dict[dataset_name]
    num_iters = num_iters_dict[dataset_name]
    if embedding_dim is None:
        embedding_dim = embedding_dim_dict[dataset_name]

    # session
    for num_seen_classes in range(nb_cl, num_classes + nb_cl, nb_cl):
        print('Task %d' % (num_seen_classes / nb_cl))

        subfolder = os.path.join(folder_path, 'class_1-%d' % num_seen_classes)

        proto_folder = os.path.join(subfolder, 'protos')
        sample_folder = os.path.join(subfolder, 'samples')

        vis_folder = os.path.join(subfolder, 'vis_2d')
        if not os.path.exists(vis_folder):
            os.makedirs(vis_folder)

        # iteration
        for iter_idx in range(test_interval, num_iters + test_interval, test_interval):
            print('Iteration %d' % (iter_idx))

            pca_protos_dict = dict()
            pca_samples_dict = dict()

            all_embedding_protos = np.zeros([0, embedding_dim], np.float)
            all_embedding_samples = np.zeros([0, embedding_dim], np.float)

            # class
            for category_idx in range(num_seen_classes):
                proto_embedding_file = os.path.join(proto_folder, 'class_%d' % (category_idx + 1),
                                                    'embedding_protos_%d.pkl' % iter_idx)
                with open(proto_embedding_file, 'rb') as fin:
                    embedding_protos = pickle.load(fin)

                sample_embedding_file = os.path.join(sample_folder, 'class_%d' % (category_idx + 1),
                                                     'embedding_samples_%d.pkl' % iter_idx)
                with open(sample_embedding_file, 'rb') as fin:
                    embedding_samples = pickle.load(fin)

                pca_protos_dict[category_idx] = embedding_protos
                pca_samples_dict[category_idx] = embedding_samples

                all_embedding_protos = np.concatenate((all_embedding_protos, embedding_protos))
                all_embedding_samples = np.concatenate((all_embedding_samples, embedding_samples))

            if vis_pca:
                pca = PCA(n_components=2)
                pca.fit(np.concatenate((all_embedding_protos, all_embedding_samples)))
                plt.figure(figsize=(6, 6), dpi=150)
                for category_idx in range(num_seen_classes):
                    pca_protos = pca_protos_dict[category_idx]
                    pca_samples = pca_samples_dict[category_idx]
                    plt.scatter(pca_protos[:, 0], pca_protos[:, 1], marker='+', alpha=1., color=colors[category_idx],
                                label='Class %d' % (category_idx + 1))
                    plt.scatter(pca_samples[:, 0], pca_samples[:, 1], marker='.', alpha=1., color=colors[category_idx],
                                label='Class %d' % (category_idx + 1))

                plt.legend()
                plt.savefig(os.path.join(vis_folder, 'pca_%d.pdf' % iter_idx))
                plt.close()

            if vis_tsne:
                tsne = TSNE(n_components=2)
                num_proto_per_class = len(embedding_protos)
                num_sample_per_class = len(embedding_samples)
                tsne_result = tsne.fit_transform(np.concatenate((all_embedding_protos, all_embedding_samples)))
                plt.figure(figsize=(6, 6), dpi=150)
                for category_idx in range(num_seen_classes):
                    tsne_protos = tsne_result[
                                  category_idx * num_proto_per_class: (category_idx + 1) * num_proto_per_class]
                    tsne_samples = tsne_result[len(all_embedding_protos) + category_idx * num_sample_per_class: len(
                        all_embedding_protos) + (category_idx + 1) * num_sample_per_class]
                    plt.scatter(tsne_protos[:, 0], tsne_protos[:, 1], marker='+', alpha=1., color=colors[category_idx],
                                label='Class %d' % (category_idx + 1))
                    plt.scatter(tsne_samples[:, 0], tsne_samples[:, 1], marker='.', alpha=1.,
                                color=colors[category_idx],
                                label='Class %d' % (category_idx + 1))

                plt.legend()
                plt.savefig(os.path.join(vis_folder, 'tsne_%d.pdf' % iter_idx))
                plt.close()


if __name__ == '__main__':
    # visualize('../result/protogan/fashion-mnist_order_1/nb_cl_2/dcgan_critic_1_class_1_ac_1.0_0.1/0.0002_0.5_0.999/10000/proto_static_20_weight_0.000100_squared_l2/dim_1024/from_scratch', dataset_name='fashion-mnist', nb_cl=2)
    # visualize('../result/protogan/fashion-mnist_order_1/nb_cl_2/dcgan_critic_1_class_1_ac_1.0_0.1/0.0002_0.5_0.999/10000/proto_static_20_weight_0.000100_squared_l2/dim_1024/finetune_improved', dataset_name='fashion-mnist', nb_cl=2)
    # visualize('../result/protogan/fashion-mnist_order_1/nb_cl_2/dcgan_critic_1_class_1_ac_1.0_0.1/0.0002_0.5_0.999/10000/proto_static_non-trainable_20_weight_0.000100_squared_l2/dim_1024/finetune_improved', dataset_name='fashion-mnist', nb_cl=2)

    # visualize('../result/protogan/fashion-mnist_order_1/nb_cl_2/dcgan_critic_1_class_1_ac_1.0_0.1/0.0002_0.5_0.999/10000/proto_static_20_weight_0.000100_squared_l2_0.010000/dim_1024/from_scratch', dataset_name='fashion-mnist', nb_cl=2)
    # visualize('../result/protogan/svhn_order_1/nb_cl_2/dcgan_critic_1_class_1_ac_1.0_0.1/0.0002_0.5_0.999/10000/proto_static_20_weight_0.000100_squared_l2_0.010000/embedding_dim_1024_gan_dim_64/finetune_improved', dataset_name='svhn', nb_cl=2)
    # visualize('../result/protogan_v2/fashion-mnist_order_1/nb_cl_2/dcgan_critic_1_ac_1.0_0.1_reconstr_0.000100/0.0002_0.5_0.999/10000/proto_static_20_squared_l2_0.010000_update_random_init/dim_128/from_scratch', dataset_name='fashion-mnist', nb_cl=2, embedding_dim=128)
    # visualize('../result/protogan_v2/fashion-mnist_order_1/nb_cl_2/dcgan_critic_1_ac_1.0_0.1_reconstr_0.000100/0.0002_0.5_0.999/10000/__proto_static_20_squared_l2_0.010000_update/dim_128/from_scratch', dataset_name='fashion-mnist', nb_cl=2, embedding_dim=128)
    # visualize(
    #     '../result/protogan_v1_3_2/fashion-mnist_order_1/nb_cl_2/dcgan_critic_1_ac_1.0_0.1/0.0002_0.5_0.999/10000/proto_static_random_dup_2_20_weight_0.000000_0.000000_squared_l2_0.010000/finetune_improved_noise_0.0',
    #     dataset_name='fashion-mnist', nb_cl=2, embedding_dim=4096)
    # visualize(
    #     '../result/protogan_v1_3_2/svhn_order_1/nb_cl_2/dcgan_critic_1_ac_1.0_0.1/0.0002_0.5_0.999/10000/proto_static_random_dup_2_20_weight_0.000000_0.000000_squared_l2_0.010000/gan_dim_64/finetune_improved_noise_0.5',
    #     dataset_name='svhn', nb_cl=2, embedding_dim=4096)
    # visualize(
    #     '../result/protogan_v1_3_2/svhn_order_1/nb_cl_2/dcgan_critic_1_ac_1.0_0.1/0.0002_0.5_0.999/10000/proto_static_random_dup_2_20_weight_0.000000_0.000000_squared_l2_0.010000_multi_center/gan_dim_64/finetune_improved_noise_0.5',
    #     dataset_name='svhn', nb_cl=2, embedding_dim=4096)

    # visualize(
    #     '../result/protogan_v1_3_8/svhn_order_1/nb_cl_2/dcgan_critic_1_ac_1.0_0.1/0.0002_0.5_0.999/10000/proto_static_random_20_weight_0.000000_0.000000_squared_l2_0.010000/gan_dim_64/finetune_improved_v2_noise_0.5_exemplars_dual_use_1',
    #     dataset_name='svhn', nb_cl=2, embedding_dim=4096, test_interval=2000)
    visualize(
        '../result/protogan_v1_3_8/svhn_order_1/nb_cl_2/dcgan_critic_1_ac_1.0_0.1/0.0002_0.5_0.999/10000/proto_static_random_20_weight_0.000000_0.000000_squared_l2_0.010000_train_rel_center/gan_dim_64/finetune_improved_v2_noise_0.5_exemplars_dual_use_1',
        dataset_name='svhn', nb_cl=2, embedding_dim=4096, test_interval=2000)
