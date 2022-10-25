# PyTorch_MNIST_GAN
<img src="https://img.shields.io/badge/license-MIT-green">   

## Summary
This is a Pytorch implementation of the GAN model proposed in "Generative Adversarial Nets".   
The paper is available [here](https://arxiv.org/pdf/1406.2661.pdf).   
**The model architecture followed the brief introduction in the paper, and there was no exact description.**

## Directory Tree
When you run train.py, the MNIST dataset is automatically downloaded and the training log is also automatically saved.

```
PyTorch_MNIST_ResNet
├─data
│  └─MNIST
├─log
├─model
│  └─GAN
└─utils
   └─log_visualization
```

## Requirements
All experiments were performed in CUDA 11.8, cudnn 8.5.0 environment.   
I provide the versions of the python package in 'requirements.txt'.   
The list is below.   
```
matplotlib==3.6.1
numpy==1.23.1
pandas==1.5.0
torch==1.12.1
torchsummary==1.5.1
torchvision==0.13.1
```


## Install
Run the following command in the terminal.
```
pip install -r requirements.txt
```

## Usage
This is made for MNIST only, but you can use it wherever you want by editing the dataset, dataloader and shape of fc layer.

```
# for model training
python train.py
```

The structure of the Generative model is as follows.
```python
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100, 128)
        self.ReLU = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, 784)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.ReLU(self.fc1(x))
        x = self.tanh(self.fc2(x))
        return x
```

The structure of the Discriminative model is as follows.
```python
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.ReLU = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.ReLU(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x
```

and can adjust batch size and epochs by editing line 15~16 of train.py

```python
EPOCH = # num of epoch
BATCH_SIZE = # size of batch
```

In order to understand the performance and loss of each model in the training process, the training process is recorded in the log folder.

If you run the log_visualize.py in the 'utils' directory, you can get the following result.
```
python utils/log_visualize.py 
```

If there is an any problem or question, please contact me or leave it in the Issues tab.    
Welcome!   

## Result for model proposed in "Deep Residual Learning for Image Recognition"
In all experiments, Adam optimizer and BinaryCrossEntropyLoss were used, and lr scheduler was not used.

```
EPOCH = 
BATCH_SIZE = 256
LEARNING_RATE = 2e-4
```

### Model's Result
