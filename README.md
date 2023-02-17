# Project 1 - MLP in Tensorflow and PyTorch

Course: Neural Networks @ FIIT STU\
Authors: Viktor Modroczký & Michaela Hanková

## Versions

Python 3.7.9 has been used for this project for compatibility reasons.
We used version 2.11.0 of Tensorflow and version 1.13.1 of PyTorch.

Run `pip install -r requirements.txt` before running the code.

## Dataset

We used the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset for this project. It contains 60,000 training images and 10,000 test images. Each image is a 28x28 grayscale image, associated with a label from 10 classes. The dataset is available in both libraries.

## Model Architecture and Hyperparameters

The model is a Multi-Layer Perceptron. It has 784 inputs which represent the pixels of the image. The output layer has 10 neurons, one for each class. The activation function is ReLU for the hidden layers and Softmax for the output layer. The loss function is Sparse Categorical Crossentropy. The optimizer is Stochastic Gradient Descent with a learning rate of 0.001.

## Tensorflow Implementation

TODO

## PyTorch Implementation

TODO

## Comparison

TODO
