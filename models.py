import torch.nn as nn
import torch


nz = 100
ngf = 128
nc = 3
ndf = 128


class Generator(nn.Module):
    '''
    Generator network CNN.
    '''

    def __init__(self, ngpu, ngf):
        """Returns the generator where ngpu refers to number of gpus available.
        If ngf is 128, another layer is added."""
        super(Generator, self).__init__()
        self.ngpu = ngpu
        if(ngf == 128):
            layers = [nn.ConvTranspose2d(nz, ngf*16, 4, 1, 0, bias=False),
                      nn.BatchNorm2d(ngf*16),
                      nn.ReLU(True),
                      nn.ConvTranspose2d(ngf*16, ngf*8, 4, 2, 1, bias=False)]
        else:
            layers = [nn.ConvTranspose2d(nz, ngf*8, 4, 2, 0, bias=False)]

        layers.append(nn.BatchNorm2d(ngf*8))
        layers.append(nn.ReLU(True))

        layers.append(nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(ngf*4))
        layers.append(nn.ReLU(True))

        layers.append(nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(ngf*2))
        layers.append(nn.ReLU(True))

        layers.append(nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(ngf))
        layers.append(nn.ReLU(True))

        layers.append(nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False))
        layers.append(nn.Tanh())

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    '''
    Discriminator network. CNN
    '''

    def __init__(self, ngpu, ndf):
        '''
        Returns a CNN. If ndf is 128, another layer is added.
        '''
        super(Discriminator, self).__init__()
        nn.self = ngpu
        layers = [nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                  nn.LeakyReLU(0.2, True),

                  nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
                  nn.BatchNorm2d(ndf*2),
                  nn.LeakyReLU(0.2, True),

                  nn.Conv2d(ndf*2, ndf*4,  4, 2, 1, bias=False),
                  nn.BatchNorm2d(ndf*4),
                  nn.LeakyReLU(0.2, True),

                  nn.Conv2d(ndf*4, ndf*8,  4, 2, 1, bias=False),
                  nn.BatchNorm2d(ndf*8),
                  nn.LeakyReLU(0.2, True)]

        if(ndf == 128):
            layers.append(nn.Conv2d(ndf*8, ndf*16, 4, 2, 1, bias=False))
            layers.append(nn.BatchNorm2d(ndf*16))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(ndf*16, 1, 4, 1, 0, bias=False))
        else:
            layers.append(nn.Conv2d(ndf*8, 1, 4, 2, 0, bias=False))
            layers.append(nn.Sigmoid())

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        return self.main(input)
