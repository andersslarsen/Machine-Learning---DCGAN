# Deep Convolutional Generative Adversarial Network
A DCGAN consists of two neural networks: A discriminator and generator. The generator tries to create synthetic data instances based upon a dataset. The discriminators role is to distinguish the real data from the generated one. 
## Discriminator
The discriminator is a Convolutional Neural Network (CNN). It consists of either 5 or 6 layers, depending on the image size trained on. Batchnormalization is used on every layer, and Leaky Relu is the activation function. THe output is a single scalar, representing the probability of x being from the real dataset.

## Generator
The Generator is also a CNN, but in "reverse" (so called Deconvolutional Neural Network). The input is a nose vector $z \sim \matchal{N}(\mu = 0,\theta ^2 = 0.02), based upon the guidance of Radford, 2016.$ 

## Training the model:
Requirements.txt are not yet set up. To run in terminal type with the following argument parsers:
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