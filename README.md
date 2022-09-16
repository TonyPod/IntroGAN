# IntroGAN

This is the official implementation of the paper ***Introspective GAN: Learning to Grow a GAN for Incremental Generation and Classification*** which is under review for Pattern Recognition (PR).

## Introduction

Lifelong learning, the ability to continually learn new concepts throughout our life, is a hallmark of human intelligence. Generally, humans learn a new concept by knowing what it looks like and what makes it different from the others, which are correlated. Those two ways can be characterized by generation and classification in machine learning respectively. In this paper, we carefully design a dynamically growing GAN called **Introspective GAN (IntroGAN)** that can perform incremental generation and classification simultaneously with the guidance of prototypes, inspired by their roles of efficient information organization in human visual learning and excellent performance in other fields like zero-shot/few-shot/incremental learning. Specifically, we incorporate prototype-based classification which is robust to feature change in incremental learning and GAN as a generative memory to alleviate forgetting into a unified end-to-end framework. A comprehensive benchmark on the joint incremental generation and classification task is proposed and our method demonstrates promising results. Additionally, we conduct comprehensive analyses over the properties of IntroGAN and verify that generation and classification can be mutually beneficial in incremental scenarios, which is an inspiring area to be further exploited.

## Usage

### 1. Requirements

The code is implemented in Python 2.7.

The CUDA version we use is 8.0 and the cuDNN version is 6.0.

The Tensorflow version is 1.4.

For requirements for the Python modules, simply run:

``pip install -r requirements.txt``

### 2. Dataset Preparation

#### 2.1 MNIST

Download ``train-images-idx3-ubyte.gz``, ``train-labels-idx1-ubyte.gz``, ``t10k-images-idx3-ubyte.gz``, ``t10k-labels-idx1-ubyte.gz`` from the official website of [MNIST](http://yann.lecun.com/exdb/mnist/) and move them to ``datasets/mnist``.

#### 2.2 Fashion-MNIST

Download ``train-images-idx3-ubyte.gz``, ``train-labels-idx1-ubyte.gz``, ``t10k-images-idx3-ubyte.gz``, ``t10k-labels-idx1-ubyte.gz`` from the official website of [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) and move them to ``datasets/fashion-mnist``.

#### 2.3 SVHN

Download ``train_32x32.mat``, ``test_32x32.mat`` from the official website of [SVHN](http://ufldl.stanford.edu/housenumbers/) and move them to ``datasets/svhn``.

### 3. Precomputed statistics for calculating FIDs

Download the files for different datasets below and extracted in the ``precalc_fids/`` folder

**MNIST**: [[Google Drive]](https://drive.google.com/file/d/12UU537Y7sZkltiTMCVU1WApNQ_7PdKVS/view?usp=sharing)

**Fashion-MNIST**: [[Google Drive]](https://drive.google.com/file/d/1ide3Be6ypqt0ymYamQQ_DpfFTJvq70dG/view?usp=sharing)

**SVHN**: [[Google Drive]](https://drive.google.com/file/d/13naH1scHToqwcyvb9InaTeK705ejElID/view?usp=sharing)

### 4. Training

**MNIST**:

_IntroGAN_: `python mnist_train_introgan.py --dataset mnist`

_DGR_: `python mnist_train_dgr.py --dataset mnist`

_MeRGAN_: `python mnist_train_mergan.py --dataset mnist`

**Fashion-MNIST**: 

_IntroGAN_: `python mnist_train_introgan.py`

_DGR_: `python mnist_train_dgr.py`

_MeRGAN_: `python mnist_train_mergan.py`

**SVHN**: 

_IntroGAN_: `python svhn_train_introgan.py`

_DGR_: `python svhn_train_dgr.py`

_MeRGAN_: `python svhn_train_mergan.py`

**After running the code above, the TA-ACC and TA-FID of this particular run can be found in the result folder, e.g. `result/introgan/fashion-mnist_order_1/nb_cl_2/dcgan_critic_1_ac_1.0_0.1/0.0002_0.5_0.999/500/proto_static_random_20_weight_0.000000_0.000000_squared_l2_0.010000_min_select/finetune_improved_v2_noise_0.5_exemplars_dual_use_1`**

## Further 

If you have any question, feel free to contact me. My email is chen.he@vipl.ict.ac.cn