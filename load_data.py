import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def main(size):
    """Extracts the images from the CelebA data set and outputs a dataloader (in tensorform).
    It also normalizes the images, and crops them to the image size given.
    Batch-size is set to 128. """

    dataroot = '/Users/HenrikGruner/code/TDT4173/celaba'
    image_size = size
    batch_size = 128
    dataset = datasets.ImageFolder(root=dataroot,
                                   transform=transforms.Compose([
                                       transforms.Resize(image_size),
                                       transforms.CenterCrop(image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize(
                                           (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True)

    return dataloader
