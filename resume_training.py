import load_data
import models
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import time
import argparse
from datetime import datetime
import random

parser = argparse.ArgumentParser(description='Hyperparameters')
parser.add_argument('-ngpu', type=int, default=1,
                    help='Number of gpus for CUDA, default = 1')
parser.add_argumer('-curr', type=int, help="Current epochs")
parser.add_argument('-e', type=int, default=50,
                    help="Number of epochs in total")
parser.add_argument('-lrd', type=float, default=0.00005,
                    help="Learning rate discriminator, default = 0.0005")
parser.add_argument('-lrg', type=float, default=0.0002,
                    help="Learning rate Generator, default = 0.002")
parser.add_argument('-one_sided', type=bool, default=True,
                    help="one-sided-smoothing")
parser.add_argument('-lr', type=bool, default=False, help="same learning rate")
parser.add_argument('-beta1', type=float, default=0.5,
                    help="Beta1 adam optimizer")
parser.add_argument('-i', type=int, default=128,
                    help="image size, default = 128")
parser.add_argument('-k', type=int, default=1,
                    help="Hyperparameter training the discriminator")
parser.add_argument('-checkpoint', type=int, default=5,
                    help="Checkpoint epochs")
parser.add_argument('--bsize', type=int, default=128, help="Batch_size")
args = parser.parse_args()

ngpu = args.ngpu
img_size = args.i
ndf = img_size
ngf = img_size
current = args.curr

device = torch.device("cuda:0" if (
    torch.cuda.is_available() and ngpu > 0) else "cpu")
dataloader = load_data.main(128)

filepath = 'trained_models/model'
gen = models.Generator(ngpu, ngf)
dis = models.Discriminator(ngpu, ndf)
try:
    gen.load_state_dict(torch.load(
        filepath + "G_"+str(img_size)+"_" + str(current), map_location=device))
    dis.load_state_dict(torch.load(
        filepath + "D_"+str(img_size)+"_" + str(current), map_location=device))
except:
    print("Models does not exist")
    exit()

nz = 100
if(args.epochs == 128):
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.BCELoss()

progress_gif_noise = torch.randn(128, nz, 1, 1, device=device)

if(args.one_sided == True):
    real_label = random.uniform(0.85, 1.15)
else:
    real_label = 1

false_label = 0

optimD = optim.Adam(dis.parameters(), lr=args.lrd, beta=(args.beta1, 0.999))
optimG = optim.Adam(gen.parameters(), lr=args.lrg, beta=(args.beta1, 0.999))

img_list = []
G_losses = []
D_losses = []
iters = 0

start = time.time()
print("Resuming the training procedure")
print("Current epoch:" + str(args.curr))

for e in range(current, args.e):
    for i, data in enumerate(dataloder, 0):
        for k in range(k):
            dis.zero_grad()
            # Sample real batch x
            real_batch = data[0].to(device)
            b_size = real_batch.size(0)
            label = torch.full((b_size,), real_label,
                               dtype=torch.float, device=device)

            output = dis(real).view(-1)
            error_real_D = criterion(output, label)
            error_real_D.backward()
            Dx = output.mean().item()

            # Sample batch z of noise
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = gen(noise)
            label.fill_(false_label)
            output = dis(fake.detach()).view(-1)
            error_fake_D = criterion(output, label)
            error_fake_D.backward()
            Dz = output.mean().item()
            errorD = error_fake_D + error_real_D
            optimD.step()

        # Generator
        gen.zero_grad()
        label.fill_(real_label)

        output = dis(fake).view(-1)
        errorG = criterion(output, label)
        errorG.backward()
        Dz2 = output.mean().item()
        optimG.step()

        # Loss
        G_losses.append(errorG)
        D_losses.append(errorD)

        # Printing
        if i % 100 == 0:
                currtim = datetime.now().strftime('%H:%M:%S')
                timesince = (time.time()-start)/60
                
                print('[%s][%d/%d][%d/%d]   %.1f minutes since start \n Loss G = %.3f  loss D = %.3f D(x) = %.3f D(G(z)) = %.3f / %.3f\n' % (currtim, e,args.e,i,len(dataloader),timesince,errorG.item(),errorD.item(),Dx,Dz,Dz2))

        if(i % 500):
            with torch.no_grad():
                fake = gen(progress_gif_noise).detach().cpu()
                img_list.append(vutils.make_grid(
                    fake, padding=2, normalize=True))

        if(e % args.checkpoint):
            if(e != current):
                torch.save(dis.state_dict(), 'trained_models/modelD_' +
                           str(img_size) + '_'+str(args.e))
                torch.save(gen.state_dict(), 'trained_models/modelG_' +
                           str(img_size)+'_' + str(args.e))


end_time = (time.time() - start) / 60
print("The training took :" + end_time + " Minutes")


plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
plt.savefig("generated_images/LOSS_"+ str(img_size) + "_" + str(args.e)+".png")



try:
    fig = plt.figure(figsize=(16, 16))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)]
           for i in img_list]
    ani = animation.ArtistAnimation(
        fig, ims, interval=1000, repeat_delay=1000, blit=True)
    ani.save('generated_images/animation_'+ str(img_size) + "_" + str(args.e)+'.gif', writer='imagemagick', fps=5)
    HTML(ani.to_jshtml())
    plt.savefig("generated_images/anim_" + str(img_size) + "_" + str(args.e)+".png")
except:
    print('couldnt save gif')

real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(16, 16))
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[
           :128], padding=5, normalize=True).cpu(), (1, 2, 0)))
plt.show()
#Last epoch
plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
plt.show()

plt.savefig('generated_images/RealAndFake_' + str(img_size)+ '_'+str(args.e)+'.png')
