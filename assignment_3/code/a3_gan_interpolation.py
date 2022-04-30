import argparse
import os

import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('No display found. Using Agg backend instead.')
    mpl.use('Agg')

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torchvision import datasets

from a3_gan_template import Generator

def save_image_samples(samples,nrows,filename):
    with torch.no_grad():
        # Sample images
        im_grid = make_grid(samples,nrows,normalize=True)
        npimg = im_grid.numpy()

        # Plot and save
        plt.figure()
        plt.imshow(npimg[0],cmap='Greys')
        plt.axis('off')
        plt.savefig(filename,bbox_inches='tight',pad_inches=0)
        #plt.clf()
        plt.close()

def main():
    # Initialize generator model
    generator = Generator(args.latent_dim)

    # Load generator
    generator.load_state_dict(torch.load("mnist_generator.pt"))

    # Put model on device
    generator = generator.to(device)

    # Turn off batch norm
    generator.eval()

    # Generate images
    z = torch.randn(2,args.latent_dim).to(device)
    space = torch.linspace(0.0,1.0,9).to(device)

    # Use z interpolations
    z_interpolated = space[:,None] * z[0,None,:] + (1 - space[:,None]) * z[1,None,:]

    imgs = generator(z_interpolated)

    # Save interpolation
    save_image_samples(imgs.cpu(),3,"interpolation.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--device', type=str, default="cpu",
                        help='Device to run the model on')
    args = parser.parse_args()

    # Initialize device objects
    device = torch.device(args.device)

    main()
