import load_data
import models
import time
import os
import argparse

# TORCHVISION IMPORTS
import torch
import torch.nn as nn
import torchvision
import torchvision.utils as vutils
import torch.optim as optim
from datetime import datetime
# NUMPY AND MATPLOTLIB - VIZUALISATION
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
from IPython.display import HTML

# Networks and data
import models
import load_data


# backend matplotlib
matplotlib.use('Agg')


#Argument parser
#------------------------------------------------
parser = argparse.ArgumentParser(description='Hyperparameters')
parser.add_argument('-ngpu',type = int, default = 1,help= 'Number of gpus for CUDA, default = 1')
parser.add_argument('-e',type = int, default = 2, help = "Number of epochs" )
parser.add_argument('-lrd', type = float, default = 0.00005, help ="Learning rate discriminator, default = 0.0005")
parser.add_argument('-lrg', type = float, default = 0.0002, help ="Learning rate Generator, default = 0.002") 
parser.add_argument('-one_sided', type = bool, default = True, help= "one-sided-smoothing")
parser.add_argument('-lr', type = bool, default = False, help ="same learning rate")
parser.add_argument('-beta1', type = float, default = 0.5, help ="Beta1 adam optimizer")
parser.add_argument('-image_size', type = int, default = 128, help = "image size, default = 128")
parser.add_argument('-k', type = int, default = 1, help = "Hyperparameter training the discriminator")
parser.add_argument('-checkpoint', type = int, default =5, help = "Checkpoint epochs")
parser.add_argument('--bsize', type = int, default =128, help = "Batch_size")
args = parser.parse_args()

print("\n")
print("PARAMETERS:")
for arg in vars(args):
    if(arg == "e"):
        print("epochs".ljust(12),getattr(args, arg))
    else:
        print(arg.ljust(12), getattr(args, arg))
#------------------------------------------------

ngpu = args.ngpu

# Parameters
#------------------------------------------------
nz = 100
nc = 3
epochs = args.e
image_size = args.image_size
ndf = image_size
ngf = image_size
batch_size = args.bsize
#------------------------------------------------

# Hyper parameters
#------------------------------------------------
k = args.k
lr_generator = args.lrg
lr_discriminator = args.lrd
beta1 = args.beta1

#------------------------------------------------
# one-sided smoothing i.e. real is in (0.85,1.15)
oss = args.one_sided

if(oss == False):
    real = 1
else:
    real = random.uniform(0.85, 1.15)
false = 0


#SETUP
#------------------------------------------------
# Images
dataloader = load_data.main(image_size,batch_size)


# CUDA support -> if ngpu > 0 & Cuda is available
device = torch.device("cuda:0" if(
    torch.cuda.is_available() and ngpu > 0) else "cpu")
    
# Initialize the trainable params as stated in DCGAN paper (Radford, 2016)

def init_weights(model):
    classname = model.__class__.__name__
    # Convolutional layers
    if(classname.find('Conv') != -1):
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    # Batchnorm layers
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1, 0.02)
        # Bias = 0
        nn.init.constant_(model.bias.data, 0)


# Networks
Generator = models.Generator(ngpu, ngf).to(device)
Discriminator = models.Discriminator(ngpu, ndf).to(device)

Discriminator.apply(init_weights)


# Loss function & Optim
if(image_size == 128):
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.BCELoss()

OptimG = optim.Adam(Generator.parameters(), lr=lr_generator, betas=(beta1, 0.999))
OptimD = optim.Adam(Discriminator.parameters(),
                    lr=lr_discriminator, betas=(beta1, 0.999))

gif_noise =torch.randn(128,nz,1,1,device = device)

#------------------------------------------------

#TRAINING
#------------------------------------------------
imagelist = []
lossG = []
lossD = []
iterations = 0
start = time.time()

print("Starting the training procedure")
for e in range(epochs):
    for i, images in enumerate(dataloader, 0):
            # TRAIINING OF THE DISCRIMINATOR
            # See goodfellow, 2014 for pseudocode and explanation
            # Procedure:
            # for epochs
            # for k steps:
            # sample m samples of noise z-> (G(z))
            # sample m samples of real images x
            # Update discriminator using gradient ascent
            # by using G(z) and x and labels
            # end for
            # sample new m samples of noise z2, G(z2)
            # find D(G(z2)) and use gradient ascend to train the generator

            for j in range(k):
                Discriminator.zero_grad()
                real_images = images[0].to(device)
                batch_size = real_images.size(0)
                label = torch.full((batch_size,), real,
                                   dtype=torch.float, device=device)
                output = Discriminator(real_images).view(-1)
                error_D = criterion(output, label)
                error_D.backward()
                # D(x), where x -> real
                D = output.mean().item()
                noise = torch.randn(batch_size, nz, 1, 1, device=device)

                fake_images = Generator(noise)
                label.fill_(false)

                output = Discriminator(fake_images.detach()).view(-1)
                error_D_fake = criterion(output, label)
                error_D_fake.backward()

                # D(G(z1))
                # printing purposes
                D_G_z = output.mean().item()

                errorD = error_D+error_D_fake
                OptimD.step()

            # ---------------------------------
            # GENERATOR TRAINING PROCEDURE
            Generator.zero_grad()
            label.fill_(real)

            output = Discriminator(fake_images).view(-1)
            errorG = criterion(output, label)
            errorG.backward()

            # D(G(z2))
            D_G_zgen = output.mean().item()

            OptimG.step()

            # Printing the progress
            if i % 100 == 0:
                currtim = datetime.now().strftime('%H:%M:%S')
                timesince = (time.time()-start)/60
                
                print('[%s][%d/%d][%d/%d]   %.1f minutes since start \n Loss G = %.3f  loss D = %.3f D(x) = %.3f D(G(z)) = %.3f / %.3f\n' % (currtim, e,epochs, i,len(dataloader),timesince,errorG.item(),errorD.item(),D,D_G_z,D_G_zgen))
                

                
            lossD.append(errorD)
            lossG.append(errorG)

            if(i % 500==0):
                try:
                    with torch.no_grad():
                        fake_images = Generator(gif_noise).detach().cpu()
                    imagelist.append(vutils.make_grid(
                        fake_images, padding=2, normalize=True))
                except:
                    print("couldnt append images")

# ---------------------------------------------------
# Training done-> printint the results
# Loss
plt.figure(figsize=(10, 7))
plt.title("Loss Generator and Discriminator")
plt.plot(lossD, label="Discriminator Loss")
plt.plot(lossG, label="Generator Loss")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.show()
try:
    plt.savefig("Loss" + str(epochs) + "-" + str(image_size))
except:
    print("couldnt save loss")

##GIF-method taken from DCGAN tutorial Pytorch
try:
    fig = plt.figure(figsize = (16,16))
    plt.axis("off")
    im = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in imagelist]
    ani = animation.ArtistAnimation(fig,im,interval = 1000, repeat_delay = 1000, blit = True)
    ani.save('Training/animation.gif', writer='imagemagick', fps=5)
    HTML(ani.to_jshtml())
    plt.savefig("Training/anim.png128-15")
except:
    print("couldnt save gif")
    for image in imagelist:
        i = 0
        plt.figure(figsize=(128*128/92, 128*128/92))
        plt.axis("off")
        plt.imshow(np.transpose(image, (1, 2, 0)))
        plt.savefig(
            "Training/128-55-"+str(i)+".png")
        plt.close()
        i+=1

real = next(iter(dataloader))
#Real images
plt.figure(figsize=(64,64))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real[0].to(device)[:128], padding=5, normalize=True).cpu(),(1,2,0)))
plt.show()


##False imgaes
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(imageelist[-1],(1,2,0)))
plt.show()

plt.savefig('Training/RealAndFake128-15.png')


