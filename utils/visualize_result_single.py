# -*- coding:utf-8 -*-  

""" 
@time: 1/25/18 9:48 AM 
@author: Chen He 
@site:  
@file: visualize_result.py
@description:  
"""

import matplotlib as mpl

mpl.use('Agg')
from pylab import *
import numpy as np
import os
import pickle

num_classes_dict = {
    'fashion-mnist': 10,
    'mnist': 10,
    'svhn': 10,
    'cifar-10': 10,
    'cifar-100': 100,
    'imagenet_64x64_dogs': 30,
    'imagenet_32x32_mini': 100
}

test_intervals_dict = {
    'fashion-mnist': 500,
    'mnist': 500,
    'svhn': 500,
    'cifar-10': 500,
    'cifar-100': 500,
    'imagenet_64x64_dogs': 500,
    'imagenet_32x32_mini': 500
}

num_iters_dict = {
    'fashion-mnist': 500,
    'mnist': 500,
    'svhn': 500,
    'cifar-10': 10000,
    'cifar-100': 10000,
    'imagenet_64x64_dogs': 10000,
    'imagenet_32x32_mini': 10000,
}


# Draw the accuracy curve of certain method
def vis_acc_and_fid(folder_path, dataset, nb_cl, test_interval=None, num_iters=None, vis_acc=True, vis_fid=True,
                    num_classes=None, fid_num=10000):
    if test_interval is None:
        test_interval = test_intervals_dict[dataset]

    if num_iters is None:
        num_iters = num_iters_dict[dataset]

    if num_classes is None:
        num_classes = num_classes_dict[dataset]
    test_fid_interval = num_iters

    '''
    vis acc
    '''
    if vis_acc:
        history_acc = dict()
        task_num_acc = 0

        for num_seen_classes in range(nb_cl, num_classes + nb_cl, nb_cl):
            subfolder = os.path.join(folder_path, 'class_1-%d' % num_seen_classes)

            # check whether the task has been trained or not
            if not os.path.exists(os.path.join(subfolder, 'class_1-%d_conf_mat.pkl' % nb_cl)):
                break

            task_num_acc += 1

            if vis_acc:
                history_acc['task %d' % (num_seen_classes / nb_cl)] = dict()

                # load acc
                for task_num_seen_classes in range(nb_cl, num_seen_classes + nb_cl, nb_cl):
                    conf_mat_filename = os.path.join(subfolder, 'class_1-%d_conf_mat.pkl' % task_num_seen_classes)
                    with open(conf_mat_filename, 'rb') as fin:
                        conf_mat_over_time = pickle.load(fin)

                    for iter in range(test_interval, num_iters + test_interval, test_interval):
                        conf_mat = conf_mat_over_time[iter]
                        accs = np.diag(conf_mat) * 100.0 / np.sum(conf_mat, axis=0)
                        acc = np.mean(accs)

                        history_acc['task %d' % (task_num_seen_classes / nb_cl)][
                            (task_num_acc - 1) * num_iters + iter] = acc

        if task_num_acc < 10:
            plt.figure(figsize=(18, 9), dpi=150)
            for task_idx in range(task_num_acc):
                ax = plt.subplot('%d%d%d' % (task_num_acc, 1, task_idx + 1))
                if task_idx == 0:
                    ax.set_title(dataset, fontdict={'size': 14, 'weight': 'bold'})

                acc_over_time = history_acc['task %d' % (task_idx + 1)]
                x, z = zip(*sorted(acc_over_time.items(), key=lambda d: d[0]))
                ax.plot(x, z, marker='.')

                # Horizontal reference lines
                for i in range(num_iters, num_iters * task_num_acc, num_iters):
                    plt.vlines(i, 0, 100, colors="lightgray", linestyles="dashed")

                ax.set_xlim(0, task_num_acc * num_iters)
                ax.set_ylim(np.max([np.min(z) - 1., 0]), np.min([np.max(z) + 1., 100]))
                ax.autoscale_view('tight')

                ax.xaxis.set_visible(False)
                ax.set_ylabel('Task %d' % (task_idx + 1), fontdict={'size': 12, 'weight': 'bold'})

                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)

            # set the xaxis of the last subplot ON
            ax.spines['bottom'].set_visible(True)
            ax.xaxis.set_visible(True)
            ax.set_xlabel('iterations', fontdict={'size': 12, 'weight': 'bold'})

            plt.margins(0)

            output_name = os.path.join(folder_path, 'accuracy_curve.pdf')
            plt.savefig(output_name)
            plt.close()

        # txt versions (only record the average acc like in iCaRL)
        average_accs = []
        for task_idx_acc in range(task_num_acc):
            average_acc = history_acc['task %d' % (task_idx_acc + 1)][num_iters * (task_idx_acc + 1)]
            average_accs.append(average_acc)
        with open(os.path.join(folder_path, 'average_accuracy.txt'), 'w') as fout:
            fout.write(os.linesep.join([str(elem) for elem in average_accs]))

        if num_classes / nb_cl == task_num_acc:
            acc_arr = []
            for task_idx in range(task_num_acc):
                acc_arr.append(np.mean(history_acc['task %d' % (task_idx + 1)].values()))
            acc_auc = np.mean(acc_arr)
            with open(os.path.join(folder_path, 'ta_acc.txt'), 'w') as fout:
                fout.write(str(acc_auc))

    '''
    vis fid
    '''
    if vis_fid:
        history_fid = dict()
        task_num_fid = 0

        sub_name = 'fid_%d' % fid_num if not fid_num == 10000 else 'fid'

        for num_seen_classes in range(nb_cl, num_classes + nb_cl, nb_cl):
            subfolder = os.path.join(folder_path, 'class_1-%d' % num_seen_classes)

            # check whether the task has been trained or not
            if not os.path.exists(os.path.join(subfolder, 'cond_%s.pkl' % sub_name)):
                break

            task_num_fid += 1

            # load fid
            if vis_fid:
                fid_filename = os.path.join(subfolder, 'cond_%s.pkl' % sub_name)
                with open(fid_filename, 'rb') as fin:
                    fid = pickle.load(fin)

                history_fid['task %d' % (num_seen_classes / nb_cl)] = dict()

                for iter in range(test_fid_interval, num_iters + test_fid_interval, test_fid_interval):
                    fid_vals = fid[iter]
                    for task_num_seen_classes in range(nb_cl, num_seen_classes + nb_cl, nb_cl):
                        fid_sum = []
                        for i in range(task_num_seen_classes):
                            fid_sum.append(fid_vals[i + 1])
                        history_fid['task %d' % (task_num_seen_classes / nb_cl)][
                            (task_num_fid - 1) * num_iters + iter] = np.mean(fid_sum)

        if task_num_fid < 10:
            plt.figure(figsize=(18, 9), dpi=150)
            for task_idx in range(task_num_fid):
                ax = plt.subplot('%d%d%d' % (task_num_fid, 1, task_idx + 1))
                if task_idx == 0:
                    ax.set_title(dataset, fontdict={'size': 14, 'weight': 'bold'})

                fid_over_time = history_fid['task %d' % (task_idx + 1)]
                x, z = zip(*sorted(fid_over_time.items(), key=lambda d: d[0]))
                ax.plot(x, z, marker='.')

                # Horizontal reference lines
                for i in range(num_iters, num_iters * task_num_fid, num_iters):
                    plt.vlines(i, 0, 100, colors="lightgray", linestyles="dashed")

                ax.set_xlim(0, task_num_fid * num_iters)
                ax.set_ylim(np.min(z) - 1., np.max(z) + 1.)
                ax.autoscale_view('tight')

                ax.xaxis.set_visible(False)
                ax.set_ylabel('Task %d' % (task_idx + 1), fontdict={'size': 12, 'weight': 'bold'})

                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)

            # set the xaxis of the last subplot ON
            ax.spines['bottom'].set_visible(True)
            ax.xaxis.set_visible(True)
            ax.set_xlabel('iterations', fontdict={'size': 12, 'weight': 'bold'})

            plt.margins(0)

            output_name = os.path.join(folder_path, '%s_curve.pdf' % sub_name)
            plt.savefig(output_name)
            plt.close()

        # txt versions (only record the average fid like in iCaRL)
        average_fids = []
        for task_idx_fid in range(task_num_fid):
            average_fid = history_fid['task %d' % (task_idx_fid + 1)][num_iters * (task_idx_fid + 1)]
            average_fids.append(average_fid)
        with open(os.path.join(folder_path, 'average_%s.txt' % sub_name), 'w') as fout:
            fout.write(os.linesep.join([str(elem) for elem in average_fids]))

        if num_classes / nb_cl == task_num_fid:
            fid_arr = []
            for task_idx in range(task_num_fid):
                fid_arr.append(np.mean(history_fid['task %d' % (task_idx + 1)].values()))
            fid_auc = np.mean(fid_arr)
            with open(os.path.join(folder_path, 'ta_%s.txt' % sub_name), 'w') as fout:
                fout.write(str(fid_auc))


def vis_acc_mergan_like(folder_path, nb_cl, num_classes, dataset, num_iters):
    aver_acc_arr = []
    task_num_acc = 0

    for num_seen_classes in range(nb_cl, num_classes + nb_cl, nb_cl):
        subfolder = os.path.join(folder_path, 'class_1-%d' % num_seen_classes)

        # check whether the task has been trained or not
        acc_file = os.path.join(subfolder, 'mergan_like_acc.pkl')
        if not os.path.exists(acc_file):
            break

        task_num_acc += 1

        with open(acc_file, 'rb') as fin:
            accs = pickle.load(fin)
            aver_acc_arr.append(np.mean(accs))

    plt.figure(figsize=(7.5, 6), dpi=150)
    plt.title(dataset)

    x = range(num_iters, num_iters * (task_num_acc + 1), num_iters)

    plt.plot(x, aver_acc_arr, marker='.')

    plt.margins(0)

    output_name = os.path.join(folder_path, 'mergan_like_acc_curve.pdf')
    plt.savefig(output_name)
    plt.close()


if __name__ == '__main__':
    # vis_acc_and_fid('../result/protogan/fashion-mnist_order_1/nb_cl_2/dcgan_critic_1_class_1_ac_1.0_0.1/0.0002_0.5_0.999/10000/proto_static_20_weight_0.000100_squared_l2/dim_1024/finetune_improved', dataset='fashion-mnist', nb_cl=2)
    # vis_acc_and_fid('../result/protogan/fashion-mnist_order_1/nb_cl_2/dcgan_critic_1_class_1_ac_1.0_0.1/0.0002_0.5_0.999/10000/proto_static_20_weight_0.000100_squared_l2/dim_1024/finetune', dataset='fashion-mnist', nb_cl=2)
    # vis_acc_and_fid('../result/protogan/fashion-mnist_order_1/nb_cl_2/dcgan_critic_1_class_1_ac_1.0_0.1/0.0002_0.5_0.999/10000/proto_static_20_weight_0.000100_squared_l2/dim_1024/from_scratch', dataset='fashion-mnist', nb_cl=2)
    # vis_acc_and_fid('../result/protogan/fashion-mnist_order_2/nb_cl_2/dcgan_critic_1_class_1_ac_1.0_0.1/0.0002_0.5_0.999/10000/proto_static_20_weight_0.000100_squared_l2/dim_1024/finetune_improved', dataset='fashion-mnist', nb_cl=2)
    # vis_acc_and_fid('../result/protogan/fashion-mnist_order_2/nb_cl_2/dcgan_critic_1_class_1_ac_1.0_0.1/0.0002_0.5_0.999/10000/proto_static_20_weight_0.000100_squared_l2/dim_1024/finetune', dataset='fashion-mnist', nb_cl=2)
    # vis_acc_and_fid('../result/protogan/fashion-mnist_order_2/nb_cl_2/dcgan_critic_1_class_1_ac_1.0_0.1/0.0002_0.5_0.999/10000/proto_static_20_weight_0.000100_squared_l2/dim_1024/from_scratch', dataset='fashion-mnist', nb_cl=2)
    # vis_acc_and_fid('../result/protogan/fashion-mnist_order_3/nb_cl_2/dcgan_critic_1_class_1_ac_1.0_0.1/0.0002_0.5_0.999/10000/proto_static_20_weight_0.000100_squared_l2/dim_1024/finetune_improved', dataset='fashion-mnist', nb_cl=2)
    # vis_acc_and_fid('../result/protogan/fashion-mnist_order_3/nb_cl_2/dcgan_critic_1_class_1_ac_1.0_0.1/0.0002_0.5_0.999/10000/proto_static_20_weight_0.000100_squared_l2/dim_1024/finetune', dataset='fashion-mnist', nb_cl=2)
    # vis_acc_and_fid('../result/protogan/fashion-mnist_order_3/nb_cl_2/dcgan_critic_1_class_1_ac_1.0_0.1/0.0002_0.5_0.999/10000/proto_static_20_weight_0.000100_squared_l2/dim_1024/from_scratch', dataset='fashion-mnist', nb_cl=2)
    vis_acc_mergan_like(
        '../result/iacgan/svhn_order_1/nb_cl_2/dcgan_critic_1_class_1_ac_1.0_0.1/0.0002_0.5_0.999/10000/finetune_improved',
        2, 10, 'svhn', 10000)
