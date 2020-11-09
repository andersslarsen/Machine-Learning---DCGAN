
# Deep Convolutional Generative Adversarial Network
A DCGAN consists of two neural networks: A discriminator and generator. The generator tries to create synthetic data instances based upon a dataset. The discriminators role is to distinguish the real data from the generated one. 

## Discriminator
The discriminator is a Convolutional Neural Network (CNN). It consists of either 5 or 6 layers, depending on the image size trained on. Batchnormalization is used on every layer, and Leaky Relu is the activation function. THe output is a single scalar, representing the probability of x being from the real dataset.

## Generator
The Generator is also a CNN, but in "reverse" (so called Deconvolutional Neural Network). The input is a nose vector z &sim; &Nscr;(&mu;=0,&theta;<sup>2</sup> = 1), based upon the guidance of Radford, 2016. The output of the generator is a batch of images.

## Training the model:
Download all packages necessary using requirements.txt.
You need to download the CelebA dataset, and change the filepath in load_data.py to your celeba folder.
To run in terminal type with the following argument parsers:
```
python3 train.py [-ngpu] [-e ] [-lrd] [-lrg ] [-one_sided] [-lr][-beta1] [-image_size ] [-k]  [-checkpoint] [--bsize]
```
<ul>
<li>Argument - Description - Default value
<li>ngpu - number of gpus - 5

<li>e - epochs - default - 5

<li>lrd - Learning rate discriminator - 0.0005

<li>lrg - Learning rate discriminator - 0.003

<li>beta1 - Adam optimizer beta 1 - 0.5

<li>image_size - image_size  for training - 128

<li>k - hyperparameter for training discriminator - 1

<li>checkpoint - saves model per checkpoint - 1

<li>bzise - number of images per training iteration - 128
</ul>

There is no point trying to train the model without gpus. 

### Training output when starting the training:
```
PARAMETERS:
ngpu         1
epochs       2 
lrd          5e-05
lrg          0.003
one_sided    True
lr           False
beta1        0.5
image_size   128
k            1
checkpoint   5
bsize        128
Starting the training procedure

[21:12:17][0/2][0/1583]   0.0 minutes since start 
 Loss G = 11.316  loss D = 2.295 D(x) = 0.994 D(G(z)) = 0.772 / -12.562

[21:15:32][0/2][100/1583]   3.3 minutes since start 
 Loss G = 13.021  loss D = 5.194 D(x) = -1.009 D(G(z)) = -1.726 / -14.454

[21:18:49][0/2][200/1583]   6.6 minutes since start 
 Loss G = 10.320  loss D = 2.486 D(x) = 2.259 D(G(z)) = -9.418 / -11.456
```

# Results
![title](images_git/RealAndFake128_50_git.png)
 ### Frèchet Inception Distance (FID)
 The FID was calculated to be 14, using 12,8k real and generated images. To calulate the FID, we used mseitzers implementation with pytorch at https://github.com/mseitzer/pytorch-fid. 

# References & help
<ol>
<li>Ian J. Goodfellow, Jean Pouget-Abadie, M. Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair,
Aaron C. Courville, and Yoshua Bengio. Generative adversarial nets. In NIPS, 2014.
<li> Alec Radford, Luke Metz, and Soumith Chintala. Unsupervised representation learning with deepconvolutional generative adversarial networks, 2016.
<li>https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
