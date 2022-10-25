import os
import pandas as pd
import matplotlib.pyplot as plt

log_dir = "./log/"
current = os.getcwd()

GAN_log = pd.read_csv(
    log_dir + "GAN_train_log.csv", index_col=0, header=None
)

image_path = "./images/"
summary = "PyTorch_MNIST_GAN"

######################################
# Visualiziing training loss and acc #
######################################
""" If you want to compare another model's train acc/loss, activate below comments. """

fig, ax = plt.subplots(1, 1, figsize=(16, 5))

ax.plot(GAN_log.iloc[:, 0], linewidth=1, label="Generator")
ax.plot(GAN_log.iloc[:, 1], linewidth=1, label="Discriminator")

ax.set_title("Training Loss Graph", fontsize=15)
ax.set_xlabel("Iteration", fontsize=15)
ax.set_ylabel("Loss", fontsize=15)

fig.legend(fontsize=15)
plt.savefig(image_path + summary + "_training_accuracy_Graph")
plt.show()
