import os
import argparse

import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('No display found. Using Agg backend instead.')
    mpl.use('Agg')

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.stats import norm

import matplotlib.pyplot as plt
from torchvision.utils import make_grid,save_image

from datasets.bmnist import bmnist


class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20, data_dim=1*28*28):
        super().__init__()

        # Initialize linear modules with variational parameters
        self.input_linear = nn.Linear(data_dim,hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim,z_dim)
        self.std_linear = nn.Linear(hidden_dim,z_dim)

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """

        # Compute hidden layer activation
        h = F.tanh(self.input_linear(input))

        # Compute mean and covariance diagonal
        mean = self.mean_linear(h)
        std = torch.exp(self.std_linear(h) / 2)

        return mean, std


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20, data_dim=1*28*28):
        super().__init__()

        self.linear_z = nn.Linear(z_dim,hidden_dim)
        self.linear_h = nn.Linear(hidden_dim,data_dim)

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean with shape [batch_size, 784].
        """

        # Compute Bernoulli means
        h = F.tanh(self.linear_z(input))
        mean = F.sigmoid(self.linear_h(h))

        return mean


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim)
        self.decoder = Decoder(hidden_dim, z_dim)

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """

        # Reshape input into batch of 1D pixel arrays
        x = input.view(input.shape[0],-1)

        # Get mean vector and covariance diagonal from variational parameters
        mu,std = self.encoder(x)

        # Sample decoder inputs from encoder
        z = mu + std * torch.randn(std.shape).to(device)

        # Calculate Bernoulli means
        mean = self.decoder(z)

        # Calculate log of decoder
        logp = x * torch.log(mean) + (1 - x) * torch.log(1 - mean)

        # Reconstruction loss
        recon_loss = -torch.sum(logp,1)

        # Regularization loss
        reg_loss = 0.5 * torch.sum(std**2 + mu**2 - 2 * torch.log(std) - 1,1)

        # Calculate average depressed Elmo
        average_negative_elbo = torch.mean(recon_loss + reg_loss)

        return average_negative_elbo

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """

        # Sample from prior
        z = torch.randn(n_samples,self.z_dim).to(device)

        # Calculate Bernoulli means
        im_means = self.decoder(z)

        # Sample images from decoder through ancestral sampling.
        # Sampling Bernoulli per mean suffices in this case.
        sampled_ims = torch.bernoulli(im_means).to(device)

        return sampled_ims, im_means


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """

    sum_epoch_elbos = 0.0
    num_elbos = 0

    for step, batch_inputs in enumerate(data):

        optimizer.zero_grad()

        # Calculate average depressed Elmo
        negative_elbo = model(batch_inputs.to(device))

        # Apply backwards step to Elmo
        negative_elbo.backward()

        # Update parameters
        optimizer.step()

        # Update Elmo sum
        sum_epoch_elbos -= negative_elbo 
        num_elbos += 1

    # Calculate average Elmo
    average_epoch_elbo = sum_epoch_elbos / num_elbos

    return average_epoch_elbo


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)

def save_image_samples(model,nrows,filename):
    with torch.no_grad():
        # Sample images
        sample_images, sample_means = model.sample(nrows**2)
        im_grid = make_grid(sample_images.view(nrows**2,1,28,28),nrows).cpu()
        npimg = im_grid.numpy()

        # Plot and save
        plt.figure()
        plt.imshow(npimg[0],cmap='Greys')
        plt.axis('off')
        plt.savefig(filename,bbox_inches='tight',pad_inches=0)

    return sample_means

def main():
    data = bmnist()[:2]  # ignore test split

    model = VAE(z_dim=ARGS.zdim).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    save_image_samples(model,5,f"zdim_{ARGS.zdim}_epoch_0_samples.pdf")

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        elbos = run_epoch(model, data, optimizer)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functioanlity that is already imported.
        # --------------------------------------------------------------------

        if (epoch+1) == ARGS.epochs // 2:
            save_image_samples(model,5,f"zdim_{ARGS.zdim}_epoch_{epoch+1}_samples.pdf")

    save_image_samples(model,5,f"zdim_{ARGS.zdim}_epoch_{ARGS.epochs}_samples.pdf")

    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------

    if ARGS.zdim == 2:
        nrows = 20

        # Construct grid batch with z values
        grid_vals = norm.ppf(np.linspace(0.05,0.95,nrows))
        X,Y = np.meshgrid(grid_vals,grid_vals)
        z = torch.tensor([X.flatten(),Y.flatten()]).to(device,torch.float).t()

        # Compute manifold data
        with torch.no_grad():
            # Compute Bernoulli means of data manifold
            means = model.decoder(z)

            manifold = make_grid(means.view(nrows**2,1,28,28),nrows).cpu()
            npimg = manifold.numpy()

        # Plot and save manifold
        plt.figure()
        plt.imshow(npimg[0],cmap='Greys')
        plt.axis('off')
        plt.savefig(f"zdim_{ARGS.zdim}_epoch_{ARGS.epochs}_manifold.pdf",bbox_inches='tight',pad_inches=0)

    save_elbo_plot(train_curve, val_curve, f"zdim_{ARGS.zdim}_epoch_{ARGS.epochs}_elbo.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')
    parser.add_argument('--device', default="cpu", type=str,
                        help='device to export model to')

    ARGS = parser.parse_args()

    # Set device
    device = torch.device(ARGS.device)

    main()
