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
real = False
Generator = models.Generator(0, ngf)
Generator.load_state_dict(torch.load(
    'modelG_128_55', map_location='cuda'))

#device = torch.device('cuda')
#noise = torch.randn(128, 100, 1, 1)
#image = Generator(noise)
if(real == False):
	for j in range(100, 500):
		device = torch.device('cuda')
		noise = torch.randn(128, 100, 1, 1)
		image = Generator(noise)
		if(image_grid == False):
			for i in range(len(image)):
				img = vutils.make_grid(image[i], padding=2, normalize=True)
				img = img.detach().numpy()
				plt.figure(figsize=(128/96, 128/96), dpi = 96)
				plt.axis("off")
				plt.imshow(np.transpose(img, (1, 2, 0)))
				plt.savefig(
					"../IS/Inception-Score/data/128-55-"+str(i+j*128)+".png")
				plt.close()
if(image_grid == True):
    img = vutils.make_grid(image, padding=2, normalize=True)
    img = img.detach().numpy()
    plt.figure(figsize=(32, 32))
    plt.axis("off")
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.savefig("Training/Gen_images/128-55-grid.png")
    plt.close()
if(real == True):	
	for j in range(100):
		dataloader = load_data.main(128)
		device = torch.device('cuda')
		image = next(iter(dataloader))
		image = image[0].to('cuda')
		for i in range(128):
			img = vutils.make_grid(image[i], padding=2, normalize=True)
			img = img.detach().cpu().numpy()
			plt.figure(figsize=(128/96, 128/96), dpi = 96)
			plt.axis("off")
			plt.imshow(np.transpose(img, (1, 2, 0)))
			plt.savefig(
				"new_real/128-55-"+str(i+j*128)+".png")
			plt.close()

