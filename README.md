# PyTorch_MNIST_GAN

![](https://img.shields.io/badge/license-MIT-green)

## Summary

This is a Pytorch implementation of the GAN model proposed in "Generative Adversarial Nets".  
The paper is available [here](https://arxiv.org/pdf/1406.2661.pdf).

The full code is available [here](https://arxiv.org/pdf/1406.2661.pdf "here").  
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

```
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

```
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

and can adjust batch size, epochs and learning rate by editing line 15~16 of train.py

```
EPOCH = # set num of epoch
BATCH_SIZE = # set size of batch
LEARNING_RATE = # set learning rate
```

In order to understand the performance and loss of each model in the training process, the training process is recorded in the log folder.

If you run the log\_visualize.py in the 'utils' directory, you can get the following result.

```
python utils/log_visualize.py 
```

If there is an any problem or question, please contact me or leave it in the Issues tab.  
Welcome!

## Result

In all experiments, Adam optimizer and BinaryCrossEntropyLoss were used, and lr scheduler was not used.  
The model was trained on a Tesla V100 GPU environment.

```
EPOCH = 1000
BATCH_SIZE = 256
LEARNING_RATE = 2e-4
```

## After 0 iter
![0 iter result](https://user-images.githubusercontent.com/59161083/198069315-130d45d4-69f0-4e5b-b024-59d0c118f514.png)

## After 1000 iter
![1000 iter result](https://user-images.githubusercontent.com/59161083/198069329-5da8b61a-5202-44c6-b86a-13b55aa8bdfd.png)

## After 2000 iter
![2000 iter result](https://user-images.githubusercontent.com/59161083/198069344-744fc719-6724-4a68-8101-a42905a482df.png)

## After 3000 iter
![3000 iter result](https://user-images.githubusercontent.com/59161083/198069358-936b23ee-ca97-4a10-aa42-b1f411c0526e.png)

## After 4000 iter
![4000 iter result](https://user-images.githubusercontent.com/59161083/198069370-f9c9342c-c2ba-44a5-94e6-d96e6ac027b9.png)

## After 5000 iter
![5000 iter result](https://user-images.githubusercontent.com/59161083/198069381-66e0c2c2-1b5f-401d-9a99-bf997fc63bab.png)

## After 6000 iter
![6000 iter result](https://user-images.githubusercontent.com/59161083/198069392-814c9d61-3160-4729-b4db-691736652e8c.png)

## After 7000 iter
![7000 iter result](https://user-images.githubusercontent.com/59161083/198069400-a537e134-514b-4c13-b175-ede5b95f87c1.png)

## After 8000 iter
![8000 iter result](https://user-images.githubusercontent.com/59161083/198069425-0a6f94a0-3667-45c5-a0f2-96d0fac63e77.png)

## After 9000 iter
![9000 iter result](https://user-images.githubusercontent.com/59161083/198069544-0082008f-d46a-412d-8132-4e9643caa2c4.png)

## After 10000 iter
![10000 iter result](https://user-images.githubusercontent.com/59161083/198069485-d0ccccc3-f4d8-41ed-a310-a3742cd69895.png)
