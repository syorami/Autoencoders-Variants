# Autoencoders-Variants

This is a repository about *Pytorch* implementations of different *Autoencoder* variants on *MNIST* or *CIFAR-10* dataset just for studing so training hyperparameters have not been well-tuned. The following models are going to be implemented:

- [x] Fully-connected Autoencoder (Simple Autoencoder)
- [x] Convolutional Autoencoder
- [x] Sparse Autoencoder (L1 regularization)
- Sparse Autoencoder (KL divergence)
- Denoising Autoencoder
- Contractive Autoencoder
- Variational Autoencoder
- Sequence-to-Sequence Autoencoder
- Adversarial Autoencoder

Tensorflow version may be updated in the future.

# Experiment Results & Analysis

Here I will list some results recorded in my experiments or analysis just for simple comparisons between different autoencoder methods after 100 epochs training (I hope this may be helpful).

| Methods | Best MSE Loss (MNIST or CIFAR-10) |
| :------: | :------: |
| Simple Autoencoder | 0.0318 (MNIST) |
| Sparse Autoencoder (L1 reg) | 0.0301 (MNIST) |
| Convolutional Autoencoder | 0.0223 (MNIST) |

# How To Run

You can just find the autoencoder you want according to file names where the model is defined and simply run it. Data loader and some other methods are 
written in `data_utils.py`.    

Pretrained autoencoders are saved in `history` directory and you can simply load them by setting `TRAIN_SCRATCH` flag in python file. Reconstruction results can be find in `reconstruct_images` directory.

# Reference Blogs & Posts

- [Introduction to autoencoders](https://www.jeremyjordan.me/autoencoders/])
- [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)