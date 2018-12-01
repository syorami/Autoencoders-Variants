import warnings
warnings.filterwarnings('ignore')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

import torch
import data_utils

from torch import nn
from torch.autograd import Variable

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 784),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def model_training(autoencoder, train_loader, epoch):
    loss_metric = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    autoencoder.train()
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        images, _ = data
        images = Variable(images)
        images = images.view(images.size(0), -1)
        if cuda: images = images.to(device)
        outputs = autoencoder(images)
        loss = loss_metric(images, outputs)
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Iter[{}/{}], MSE loss:{:.4f}'
                  .format(epoch + 1, EPOCHS, i + 1, len(train_loader.dataset) // BATCH_SIZE, loss.item()))

def evaluation(autoencoder, test_loader):
    total_loss = 0
    loss_metric = nn.MSELoss()
    for i, data in enumerate(test_loader):
        images, _ = data
        images = Variable(images)
        if cuda: images = images.to(device)
        outputs = autoencoder(images)
        loss = loss_metric(images, outputs)
        total_loss += loss * len(data)
    print('\nAverage MSE Loss on Test set: {:.4f}\n'.format(total_loss))


if __name__ == '__main__':

    EPOCHS = 200
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5

    train_loader, test_loader = data_utils.load_mnist(BATCH_SIZE)

    autoencoder = Autoencoder()
    if cuda: autoencoder.to(device)

    for epoch in range(EPOCHS):
        model_training(autoencoder, train_loader, epoch)
        evaluation(autoencoder, test_loader)
