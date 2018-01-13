# Convolutional Neural Networks for Visual Recognition
This repository contains the solutions to assignments for [CS231n](https://cs231n.github.io) offered by Stanford University.

## Assignment 1
Implemented and applied several classifiers such as
* k-nearest-neighbor using Euclidean (or L2) distance
* linear multiclass SVM with hinge-loss (or max-margin loss)
* softmax classifier with cross-entropy loss
* two-layer vanilla neural network

to **CIFAR-10** dataset with *SGD* for optimization and *L2 regularization* in all cases. *Cross validation* and *random search* methods were applied for hyperparameter tuning. Improvements gained by using *higher-level representations* (or "features") instead of raw pixel values are also examined.

## Assignment 2
* Implemented *Dropout* and *Batch Normalization* layers from scratch
* Used above layers in a *Fully Connected Neural Network* and analyzed the improvements in performance on **CIFAR-10**
* Implemented a *Convolutional Neural Network* using NumPy, complete with **convolutional**, **max-pool**, **spatial batch normalization** and **fully-connected** layers and used it for image classification on *CIFAR-10* dataset
* Used **TensorFlow** to experiment on different architectures for *Convolutional Neural Networks* and obtain a decent performance on *CIFAR-10* dataset
## Assignment 3
* Image captioning using vanilla RNN and then LSTMs
* Generative Adversarial Networks
* Neural Style Transfer
* Network Visualization
