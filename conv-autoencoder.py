import warnings
warnings.filterwarnings('ignore')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import data_utils

from torch import nn
from torch.autograd import Variable

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.MaxPool2d(2, )
        )