import os
from sqlite3 import Row
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from model.GAN import Discriminator, Generator
from utils.util import AverageMeter

import matplotlib.pyplot as plt

BATCH_SIZE = 32
EPOCH = 50

images = []
if __name__ == "__main__":
    mnist_train_dataset = datasets.MNIST(
        root="./data/", train=True, transform=transforms.ToTensor(), download=True
    )
    mnist_test_dataset = datasets.MNIST(
        root="./data/", train=False, transform=transforms.ToTensor(), download=True
    )

    mnist_train_dataloader = DataLoader(
        mnist_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, drop_last=True
    )
    mnist_test_dataloader = DataLoader(
        mnist_test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, drop_last=True
    )

    G = Generator().cuda()
    D = Discriminator().cuda()

    optimizer_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    criterion = nn.BCELoss()

    log_dir = "./log"

    os.makedirs(log_dir, exist_ok=True)

    G_loss_log, D_loss_log = AverageMeter(), AverageMeter()

    with open(os.path.join(log_dir, "GAN_train_log.csv"), "w") as log:
        G.train()
        D.train()
        for epoch in range(EPOCH):
            for iter, (x, y) in enumerate(mnist_train_dataloader):
                x = x.view(x.shape[0], -1)
                x = x.float().cuda()

                real = torch.ones(BATCH_SIZE, 1).cuda()
                fake = torch.zeros(BATCH_SIZE, 1).cuda()

                # for Generator
                optimizer_G.zero_grad()
                noise = torch.randn(BATCH_SIZE, 100).cuda()
                out_G = G(noise)
                out_D = D(out_G)

                loss_G = criterion(out_D, real)
                loss_G.backward()
                optimizer_G.step()
                
                # for Discriminator
                optimizer_D.zero_grad()
                out_D_real = D(x) # from real(image) data
                out_D_fake = D(G(noise)) # from fake(noise) data

                loss_D_real = criterion(out_D_real, real)
                loss_D_fake = criterion(out_D_fake, fake)
                
                loss_D = (loss_D_fake + loss_D_real) / 2
                loss_D.backward()
                optimizer_D.step()


                if (epoch * len(mnist_train_dataloader) + iter) % 100 == 0:
                    print(
                        "Iter [%3d/%3d] | Generator Loss %.4f | Discriminator Loss %.4f"
                        % (
                            epoch * len(mnist_train_dataloader) + iter,
                            EPOCH * len(mnist_train_dataloader),
                            loss_G.item(),
                            loss_D.item()
                        )
                    )

                    log.write(
                        "%d,%.4f,%.4f\n"
                        % (
                            epoch * len(mnist_train_dataloader) + iter,
                            loss_G.item(),
                            loss_D.item()
                        )
                    )

                if (epoch * len(mnist_train_dataloader) + iter) % 100 == 0:
                    z = torch.randn(BATCH_SIZE, 100).type(torch.FloatTensor).cuda()
                    G.eval()
                    gen_imgs = G(z)
                    gen_imgs = gen_imgs.view(BATCH_SIZE,1,28,28).cpu().detach().numpy()
                    fig, axes = plt.subplots(4, 8, figsize=(10, 10))
                    for idx, img in enumerate(gen_imgs):
                        plt.subplot(4,8,idx+1)
                        plt.imshow(img[0], cmap='gray') 
                    plt.savefig(str("./images/")+str(epoch * len(mnist_train_dataloader) + iter)+" iter result.png")
                    G.train()