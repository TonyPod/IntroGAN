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



#### 2.2 Fashion-MNIST

Download ``train-images-idx3-ubyte.gz``, ``train-labels-idx1-ubyte.gz``, ``t10k-images-idx3-ubyte.gz``, ``t10k-labels-idx1-ubyte.gz`` from the official website of [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) and move them to ``datasets/fashion-mnist``.

### 3. Precomputed statistics for calculating FIDs

Download the files for different datasets below and extracted in the ``precalc_fids/`` folder

**MNIST**: [[Google Drive]](https://drive.google.com/file/d/12UU537Y7sZkltiTMCVU1WApNQ_7PdKVS/view?usp=sharing)

**Fashion-MNIST**: [[Google Drive]](https://drive.google.com/file/d/1ide3Be6ypqt0ymYamQQ_DpfFTJvq70dG/view?usp=sharing)

**SVHN**: [[Google Drive]](https://drive.google.com/file/d/13naH1scHToqwcyvb9InaTeK705ejElID/view?usp=sharing)

### 3. Training

**MNIST**: ``python mnist_train_introgan.py``

**Fashion-MNIST**: ``python mnist_train_introgan.py --dataset mnist``


**SVHN**: ``python svhn_train_introgan.py``