import models
import matplotlib.pyplot as plt
import matplotlib
import load_data
import torch
import torchvision.utils as vutils
import numpy as np
import argparse
import time
import load_data


def realnfake(image, image_size, epochs):
    dataloader = load_data.main((image_size))
    real = next(iter(dataloader))

    plt.figure(figsize=(64, 64))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images", fontsize=100)
    plt.imshow(np.transpose(vutils.make_grid(real[0].to(device)[
               :128], padding=5, normalize=True).cpu(), (1, 2, 0)))

    img = vutils.make_grid(image, padding=2, normalize=True)
    img = img.detach().numpy()
    plt.subplot(1, 2, 2)
    plt.title("Fake Images", fontsize=100)
    plt.axis("off")
    plt.imshow(np.transpose(img, (1, 2, 0)))

    plt.savefig('generated_images/RealAndFake' +
                str(image_size) + "_" + str(epochs) + '.png')
    plt.close()


# Set matplotlib backend i.e. writing file
# instead of showing to speed up the process
matplotlib.use('Agg')
parser = argparse.ArgumentParser(description='Parameters')
parser.add_argument('-e', type=int, default=50, help="number of epochs")
parser.add_argument('-img', type=int, default=128,
                    help="Image size, i.e. 64 or 128")
parser.add_argument("-g", type=bool, default=True,
                    help="single image or grid? True = grid")
parser.add_argument("-r", type=bool, default=False,
                    help="Generate both real and fake images for comparison")
args = parser.parse_args()


epochs = args.e
image_grid = args.g
image_size = args.img
both = args.r
ndf = image_size
ngf = image_size

nc = 3
nz = 100
img_list = []
filepath = 'trained_models/modelG_' + str(image_size)+'_' + str(epochs)
Generator = models.Generator(0, ngf)

try:
    Generator.load_state_dict(torch.load(
        filepath, map_location='cpu'))
except:
    print("Model does not exist")

device = torch.device('cpu')
noise = torch.randn(128, 100, 1, 1)
image = Generator(noise)


if(image_grid == False):
    for i in range(len(image)):
        img = vutils.make_grid(image[i], padding=2, normalize=True)
        img = img.detach().numpy()
        plt.figure(figsize=(128/92, 128/92))
        plt.axis("off")
        plt.imshow(np.transpose(img, (1, 2, 0)))
        plt.savefig(
            "generated_images/128-55-"+str(i)+".png")
        plt.close()

if(image_grid == True and both == False):
    img = vutils.make_grid(image, padding=2, normalize=True)
    img = img.detach().numpy()
    plt.figure(figsize=(32, 32))
    plt.axis("off")
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.savefig("generated_images/128-55-grid.png")
    plt.close()
if(image_grid == True and both == True):
    realnfake(image, image_size, epochs)
