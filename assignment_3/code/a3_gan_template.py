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


class Generator(nn.Module):
    def __init__(self,latent_dim=100):
        super(Generator, self).__init__()

        # Construct generator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 768
        #   Output non-linearity

        # Basic generator
        self.linear_z = nn.Linear(latent_dim,128)
        self.linear_h1 = nn.Linear(128,256)
        self.batchnorm_h1 = nn.BatchNorm1d(256)
        self.linear_h2 = nn.Linear(256,512)
        self.batchnorm_h2 = nn.BatchNorm1d(512)
        self.linear_h3 = nn.Linear(512,1024)
        self.batchnorm_h3 = nn.BatchNorm1d(1024)
        self.linear_out = nn.Linear(1024,784)

    def forward(self, z):
        # Generate images from z

        # Basic generator
        out = F.leaky_relu(self.linear_z(z),0.2)
        out = self.batchnorm_h1(F.leaky_relu(self.linear_h1(out),0.2))
        out = self.batchnorm_h2(F.leaky_relu(self.linear_h2(out),0.2))
        out = self.batchnorm_h3(F.leaky_relu(self.linear_h3(out),0.2))
        out = F.tanh(self.linear_out(out))
        out = out.view(z.shape[0],1,28,28)

        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Construct distriminator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity

        # Basic discriminator
        #self.linear_x = nn.Linear(784,512)
        #self.linear_h = nn.Linear(512,256)
        #self.linear_out = nn.Linear(256,1)

        # Upgraded discriminator
        self.conv1 = nn.Conv2d(1,16,3,stride=1,padding=1)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.avgpool1 = nn.AvgPool2d(3,stride=2,padding=1)
        self.conv2 = nn.Conv2d(16,32,3,stride=1,padding=1)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.avgpool2 = nn.AvgPool2d(3,stride=2,padding=1)
        self.conv3 = nn.Conv2d(32,64,3,stride=1,padding=1)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.avgpool3 = nn.AvgPool2d(3,stride=2,padding=1)
        self.conv4 = nn.Conv2d(64,128,3,stride=1,padding=1)
        self.batchnorm4 = nn.BatchNorm2d(128)
        self.avgpool4 = nn.AvgPool2d(4,stride=1,padding=0)
        self.linear_out = nn.Linear(128,1)

    def forward(self, img):
        # return discriminator score for img

        # Basic discriminator
        #x = img.view(img.shape[0],-1)
        #out = F.leaky_relu(self.linear_x(x),0.2)
        #out = F.leaky_relu(self.linear_h(out),0.2)
        #out = F.sigmoid(self.linear_out(out))

        # Upgraded discriminator
        out = F.leaky_relu(self.batchnorm1(self.conv1(img)),0.2)
        out = self.avgpool1(out)
        out = F.leaky_relu(self.batchnorm2(self.conv2(out)),0.2)
        out = self.avgpool2(out)
        out = F.leaky_relu(self.batchnorm3(self.conv3(out)),0.2)
        out = self.avgpool3(out)
        out = F.leaky_relu(self.batchnorm4(self.conv4(out)),0.2)
        out = self.avgpool4(out)
        out = F.sigmoid(self.linear_out(out[:,:,0,0]))

        return out


def save_image_samples(samples,nrows,filename):
    with torch.no_grad():
        # Sample images
        im_grid = make_grid(samples.view(nrows**2,1,28,28),nrows,normalize=True)
        npimg = im_grid.numpy()

        # Plot and save
        plt.figure()
        plt.imshow(npimg[0],cmap='Greys')
        plt.axis('off')
        plt.savefig(filename,bbox_inches='tight',pad_inches=0)
        #plt.clf()
        plt.close()

def train(dataloader, discriminator, generator, optimizer_G, optimizer_D):
    # Binary Cross Entropy loss
    bce_loss = nn.BCELoss()

    for epoch in range(args.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            # Puts input batch into device
            imgs = imgs.to(device)

            # Labels
            real = torch.ones(imgs.shape[0],1).to(device)
            fake = torch.zeros(imgs.shape[0],1).to(device)

            # Train Generator
            # ---------------
            optimizer_G.zero_grad()

            # Sample input noise
            z = torch.randn(imgs.shape[0],args.latent_dim).to(device)

            # Generate fake images
            fake_imgs = generator(z)

            # Calculate score for D(G(z))
            score_fake = discriminator(fake_imgs)

            # Calculate generator loss
            #loss_G = -torch.mean(torch.log(score_fake))
            loss_G = bce_loss(score_fake,real)

            # Backward step for generator
            loss_G.backward()

            # Update generator parameters
            optimizer_G.step()

            # Train Discriminator
            # -------------------
            optimizer_D.zero_grad()

            # Calculate scores for reals D(x) and fakes D(G(z))
            score_fake = discriminator(fake_imgs.detach())
            score_real = discriminator(imgs)

            # Calculate discriminator loss
            #loss_D = -(torch.mean(torch.log(score_real)) + 
            #           torch.mean(torch.log(1-score_fake))) / 2

            loss_D = (bce_loss(score_real,real) + bce_loss(score_fake,fake)) / 2

            # Backward step for discriminator
            loss_D.backward()

            # Update discriminator parameters
            optimizer_D.step()

            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                # save_image(gen_imgs[:25],
                #            'images/{}.png'.format(batches_done),
                #            nrow=5, normalize=True)

                print(f"[Epoch {epoch}] Batches_done: {batches_done}, "
                      f"Discriminator loss: {loss_D}, Generator loss: {loss_G}")

                #save_image(fake_imgs[:25],f"images/{batches_done}.png",nrow=5,normalize=True)
                save_image_samples(fake_imgs[:25].cpu(),5,f"images/{batches_done}.png")

def main():
    # Create output image directory
    os.makedirs('images', exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = Generator(args.latent_dim).to(device)
    discriminator = Discriminator().to(device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    torch.save(generator.cpu().state_dict(), "mnist_generator.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--device', type=str, default="cpu",
                        help='Device to run the model on')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    args = parser.parse_args()

    # Initialize device objects
    device = torch.device(args.device)

    main()
