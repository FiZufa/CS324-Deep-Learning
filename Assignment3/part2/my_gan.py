import argparse
import os
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets

n_epochs = 200
batch_size = 64
learning_rate = 0.0002
latent_dim = 100
save_interval = 500

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
  
        # Construct generator. You should experiment with your model,
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
        self.model = nn.Sequential(
            # input layer
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            # hidden layer 1
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # hidden layer 2
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # hidden layer 3
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            # output layer
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, z):
        # Generate images from z
        img = self.model(z)
        img = img.view(img.size(0), 1, 28, 28)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Construct distriminator. You should experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity
        self.model = nn.Sequential(
            # input layer
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2, inplace=True),
            # input layer
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            # output layer
            nn.Linear(256,1),
            nn.Sigmoid()
        )

    def forward(self, img):
        # return discriminator score for img
        img_flatten = img.view(img.size(0), -1)
        valid = self.model(img_flatten)
        return valid


def sample_images(generator, epoch, batches_done, phase, latent_dim):
    z = torch.randn(25, latent_dim).to(device)
    gen_imgs = generator(z)
    save_image(gen_imgs, f'result_imgs/{phase}_epoch{epoch}_batch{batches_done}.png', nrow=5, normalize=True, value_range=(-1, 1))

# def interpolate_images(generator, z1, z2, steps=7):
#     alpha_vals = np.linspace(0, 1, steps + 2)
#     interpolated_imgs = []
#     for alpha in alpha_vals:
#         z_interpolated = (1 - alpha) * z1 + alpha * z2
#         img = generator(z_interpolated.to(device))
#         interpolated_imgs.append(img)
#     interpolated_imgs = torch.cat(interpolated_imgs)
#     save_image(interpolated_imgs, 'images/interpolation.png', nrow=9, normalize=True, value_range=(-1, 1))


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D, n_epochs, save_interval, latent_dim):
    adv_loss = nn.BCELoss()

    for epoch in range(n_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            real_imgs = imgs.to(device)

            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(imgs.size(0), latent_dim).to(device)
            gen_imgs = generator(z)
            valid = torch.ones(imgs.size(0), 1).to(device)
            g_loss = adv_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            real_loss = adv_loss(discriminator(real_imgs), valid)
            fake = torch.zeros(imgs.size(0), 1).to(device)
            fake_loss = adv_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            print(f'[Epoch {epoch+1}/{n_epochs}] [Batch {i}/{len(dataloader)}] D Loss: {d_loss.item()} G Loss: {g_loss.item()}')

            # Sample images at different training phases
            if epoch == 0 and i == 0:
                sample_images(generator, epoch, i, 'start', latent_dim)
            if epoch == n_epochs // 2 and i == 0:
                sample_images(generator, epoch, i, 'midway', latent_dim)
            if epoch == n_epochs - 1 and i == 0:
                sample_images(generator, epoch, i, 'end', latent_dim)

            # Save images at specified intervals
            batches_done = epoch * len(dataloader) + i
            if batches_done % save_interval == 0:
                save_image(gen_imgs[:25], f'result_imgs/{batches_done}.png', nrow=5, normalize=True, value_range=(-1, 1))

    print("Finished training!")
# def main():
# def main():
#     # Create output image directory
#     os.makedirs('images', exist_ok=True)

#     # load data
#     dataloader = torch.utils.data.DataLoader(
#         datasets.MNIST('./data/mnist', train=True, download=True,
#                        transform=transforms.Compose([
#                            transforms.ToTensor(),
#                            transforms.Normalize((0.5),
#                                                 (0.5))])),
#         batch_size=args.batch_size, shuffle=True)

#     # Initialize models and optimizers
#     generator = Generator()
#     discriminator = Discriminator()
#     optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
#     optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

#     # Start training
#     train(dataloader, discriminator, generator, optimizer_G, optimizer_D)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    # torch.save(generator.state_dict(), "mnist_generator.pt")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--n_epochs', type=int, default=200,
#                         help='number of epochs')
#     parser.add_argument('--batch_size', type=int, default=64,
#                         help='batch size')
#     parser.add_argument('--lr', type=float, default=0.0002,
#                         help='learning rate')
#     parser.add_argument('--latent_dim', type=int, default=100,
#                         help='dimensionality of the latent space')
#     parser.add_argument('--save_interval', type=int, default=500,
#                         help='save every SAVE_INTERVAL iterations')
#     args = parser.parse_args()

#     main()
