import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt

from simple_autoencoder import Autoencoder
from sparse_autoencoder_l1 import SparseAutoencoder

if __name__ == '__main__':
    autoencoder = Autoencoder()
    sparse_autoencoder = SparseAutoencoder()

    autoencoder.load_state_dict(torch.load('./history/simple_autoencoder.pt'))
    sparse_autoencoder.load_state_dict(torch.load('./history/sparse_autoencoder_l1.pt'))

    autoencoder.cpu()
    sparse_autoencoder.cpu()

    autoencoder_weigths = autoencoder.state_dict()['encoder.0.weight'].view(128, 1, 28, 28)
    sparse_autoencoder_weights = sparse_autoencoder.state_dict()['encoder.0.weight'].view(128, 1, 28, 28)

    torch_scale = lambda x: (x - torch.min(x)) / (torch.max(x) - torch.min(x))
    ae_weights_norm = torch_scale(autoencoder_weigths)
    sae_weights_norm = torch_scale(sparse_autoencoder_weights)

    plt.figure()
    plt.subplot(121)
    plt.title('Autoencoder Visualization')
    plt.imshow(np.transpose(torchvision.utils.make_grid(ae_weights_norm).numpy(), (1, 2, 0)))
    plt.subplot(122)
    plt.title('Sparse Autoencoder Visualization')
    plt.imshow(np.transpose(torchvision.utils.make_grid(sae_weights_norm).numpy(), (1, 2, 0)))
    plt.savefig('./images/sparse_autoencoder_visualization.png')