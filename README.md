# Autoencoders-Variants

This is a repository about *Pytorch* implementations of different *Autoencoder* variants on *MNIST* or *CIFAR-10* dataset just for studing so training hyperparameters have not been well-tuned. The following models are going to be implemented:

- [x] Fully-connected Autoencoder (Simple Autoencoder)
- Convolutional Autoencoder
- Sparse Autoencoder (L1 regularization / KL divergence)
- Denoising Autoencoder
- Contractive Autoencoder
- Variational Autoencoder
- Sequence-to-Sequence Autoencoder
- Adversarial Autoencoder

Tensorflow version may be updated in the future.

# How To Run

You can just find the autoencoder you want according to file names where the model is defined and simply run it. Data loader and some other methods are written in `data_utils.py`

# Reference Blogs & Posts

- [Introduction to autoencoders](https://www.jeremyjordan.me/autoencoders/])
- [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)