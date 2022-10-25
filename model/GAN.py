import torch.nn as nn
from torchsummary import summary

class Discriminator(nn.Module):
    """
    Determines whether the input is MNIST data or not.
    input: 
    """    
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(784, 1024)
        self.LReLU1 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout1 = nn.Dropout(0.3)

        # self.fc2 = nn.Linear(1024, 512)
        # self.LReLU2 = nn.LeakyReLU(0.2, inplace=True)
        # self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(1024, 256)
        self.LReLU3 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout3 = nn.Dropout(0.3)

        self.fc4 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout1(self.LReLU1(self.fc1(x)))
        # x = self.dropout2(self.LReLU2(self.fc2(x)))
        x = self.dropout3(self.LReLU3(self.fc3(x)))        
        out = self.sigmoid(self.fc4(x))
        return out

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100, 256)
        self.LReLU1 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout1 = nn.Dropout(0.3)

        # self.fc2 = nn.Linear(256, 512)
        # self.LReLU2 = nn.LeakyReLU(0.2, inplace=True)
        # self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(256, 1024)
        self.LReLU3 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout3 = nn.Dropout(0.3)

        self.fc4 = nn.Linear(1024, 784)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout1(self.LReLU1(self.fc1(x)))
        # x = self.dropout2(self.LReLU2(self.fc2(x)))
        x = self.dropout3(self.LReLU3(self.fc3(x)))
        out = self.tanh(self.fc4(x))
        return out