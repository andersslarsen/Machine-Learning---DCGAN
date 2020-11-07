import models
import matplotlib.pyplot as plt
import matplotlib
import load_data
import torch
import torchvision.utils as vutils
import numpy as np

# Set matplotlib backend i.e. writing file
# instead of showing to speed up the process
matplotlib.use('Agg')

nc = 3
ndf = 128
ngf = 128
nz = 100
image_grid = False
img_list = []

Generator = models.Generator(0, ngf)
Generator.load_state_dict(torch.load(
    '/Users/HenrikGruner/code/trained_models/modelG_128_50_55', map_location='cpu'))

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
            "/Users/HenrikGruner/code/DATA/Gen_images20/128-55-"+str(i)+".png")
        plt.close()
else:
    img = vutils.make_grid(image, padding=2, normalize=True)
    img = img.detach().numpy()
    plt.figure(figsize=(32, 32))
    plt.axis("off")
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.savefig("/Users/HenrikGruner/code/DATA/Gen_images20/128-55-grid.png")
    plt.close()
