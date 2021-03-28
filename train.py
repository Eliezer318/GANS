import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32


def train(g_model: nn.Module, d_model: nn.Module, data_loader: DataLoader, params):
    """
    :param g_model: Generator Model
    :param d_model: Discriminator Model
    :param data_loader: Data Loader of all images
    :param params: contain dictionary with:
        fixed_noise: fixed_noise
        epochs: num of epochs epochs
        latent_dim_cont: dim of continuous tensor
        latent_dim_disc: dim of discrete tensor
        lr_g: learning rate of generator
        lr_d: learning rate of discriminator
    :return:
        list of generator train loss
        list of generator discriminator loss
        list of images of fixed noise to follow
    """
    criterion = nn.BCELoss()

    def create_latent():
        z_continuous = torch.randn(batch_size, params['latent_dim_cont'], 1, 1, device=device)
        z_discrete = 2 * torch.randint(0, 2, (batch_size, params['latent_dim_disc'], 1, 1), device=device) - 1
        return torch.cat((z_continuous, z_discrete), dim=1)

    def train_discriminator(netG: nn.Module, netD: nn.Module, optimizer, real_images: torch.Tensor):
        real_label = torch.ones(batch_size, dtype=torch.float32, device=device)
        fake_label = torch.zeros(batch_size, dtype=torch.float32, device=device)
        netD.zero_grad()
        # Format batch
        # Forward pass real batch through D
        output = netD(real_images)
        # Calculate loss on all-real batch
        errD_real = criterion(output, real_label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = create_latent()
        # Generate fake image batch with G
        fake = netG(noise)
        # Classify all fake batch with D
        output = netD(fake.detach())
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, fake_label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizer.step()
        return errD, D_x, D_G_z1

    def train_generator(netG, netD, optimizer):
        optimizer.zero_grad()
        labels = torch.ones(batch_size, dtype=torch.float32, device=device)
        netG.zero_grad()
        fake = netG(create_latent())
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake)
        # Calculate G's loss based on this output
        errG = criterion(output, labels)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizer.step()
        return errG, D_G_z2

    g_model = g_model.to(device)
    d_model = d_model.to(device)
    d_optimizer = torch.optim.Adam(d_model.parameters(), lr=params['lr_g'], betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(g_model.parameters(), lr=params['lr_d'], betas=(0.5, 0.999))
    g_loss_hist, d_loss_hist, img_list = [], [], []
    count = 0
    for epoch in range(params['epochs']):
        for i, images in enumerate(tqdm(data_loader)):
            batch_size = images.size(0)
            images = images.to(device)
            d_loss, real_score, fake_score_1 = train_discriminator(g_model, d_model, d_optimizer, images)
            d_loss_hist.append(d_loss.item())
            g_loss, fake_score_2 = train_generator(g_model, d_model, g_optimizer)
            g_loss_hist.append(g_loss.item())

            # if i % 25 == 0:
            #     tqdm.write(f'\n[{epoch}/{params["epochs"]}]\t Discriminator Loss is: {d_loss.item(): .4f}\t Generator Loss is: {g_loss.item(): .4f}\t'
            #                f'D(x): {real_score: .4f}\t D(G(z)): {fake_score_1: .4f}/{fake_score_2: .4f}')
            # if (count % 400 == 0) or ((epoch == params['epochs'] - 1) and (i == num_samples - 1)):
            #     with torch.no_grad():
            #         fake = vutils.make_grid(g_model(params['fixed_noise']), padding=2, normalize=True).cpu().detach().numpy()
            #         # plt.imshow(np.transpose(fake, (1, 2, 0)))
            #         # plt.show()
            #     img_list.append(fake)
            count += 1
    return g_loss_hist, d_loss_hist, img_list

