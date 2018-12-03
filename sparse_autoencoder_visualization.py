import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt

from simple_autoencoder import Autoencoder
from sparse_autoencoder_l1 import SparseAutoencoderL1
from sparse_autoencoder_KL import SparseAutoencoderKL

if __name__ == '__main__':

    autoencoder = Autoencoder()
    sparse_autoencoder_l1 = SparseAutoencoderL1()
    sparse_autoencoder_kl = SparseAutoencoderKL()

    autoencoder.load_state_dict(torch.load('./history/simple_autoencoder.pt'))
    sparse_autoencoder_l1.load_state_dict(torch.load('./history/sparse_autoencoder_l1.pt'))
    sparse_autoencoder_kl.load_state_dict(torch.load('./history/sparse_autoencoder_KL.pt'))

    autoencoder.cpu()
    sparse_autoencoder_l1.cpu()
    sparse_autoencoder_kl.cpu()

    autoencoder_weigths = autoencoder.state_dict()['encoder.0.weight'].view(128, 1, 28, 28)
    sae_l1_weights = sparse_autoencoder_l1.state_dict()['encoder.0.weight'].view(128, 1, 28, 28)
    sae_kl_weights = sparse_autoencoder_kl.state_dict()['encoder.0.weight'].view(128, 1, 28, 28)

    torch_scale = lambda x: (x - torch.min(x)) / (torch.max(x) - torch.min(x))
    ae_weights_norm = torch_scale(autoencoder_weigths)
    sae_l1_weights_norm = torch_scale(sae_l1_weights)
    sae_kl_weights_norm = torch_scale(sae_kl_weights)

    plt.figure(figsize=(12, 6))
    plt.subplot(131)
    plt.title('Autoencoder')
    plt.imshow(np.transpose(torchvision.utils.make_grid(ae_weights_norm).numpy(), (1, 2, 0)))
    plt.subplot(132)
    plt.title('Sparse Autoencoder with L1 Reg')
    plt.imshow(np.transpose(torchvision.utils.make_grid(sae_l1_weights_norm).numpy(), (1, 2, 0)))
    plt.subplot(133)
    plt.title('Sparse Autoencoder with KL divergence')
    plt.imshow(np.transpose(torchvision.utils.make_grid(sae_kl_weights_norm).numpy(), (1, 2, 0)))
    plt.savefig('./images/sparse_autoencoder_visualization.png')