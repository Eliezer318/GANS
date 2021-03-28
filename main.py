import torch
from dataset import MyDataset
import matplotlib.pyplot as plt
from train import train
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from models import Generator, Discriminator
import numpy as np
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot(g_loss_hist, d_loss_hist):
    plt.plot(d_loss_hist, c='b', label='Discriminator Loss')
    plt.title('Discrimnator Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()

    plt.figure()
    plt.plot(g_loss_hist, c='r', label='Generator Loss')
    plt.title('Generator Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()


def main():
    img_dir = '/datashare/img_align_celeba'
    dataset = MyDataset(img_dir)
    train_params = {
        'epochs': 20,
        'latent_dim_cont': 90,
        'latent_dim_disc': 10,
        'lr_g': 2e-4,
        'lr_d': 2e-4,
        'batch_size': 256
    }
    z_continuous = torch.randn(64, train_params['latent_dim_cont'], 1, 1, device=device)
    z_discrete = 2 * torch.randint(0, 2, (64, train_params['latent_dim_disc'], 1, 1), device=device) - 1
    train_params['fixed_noise'] = torch.cat((z_continuous, z_discrete), dim=1)
    dataLoader = DataLoader(dataset, train_params['batch_size'], num_workers=8, shuffle=True)
    g_model = Generator(train_params['latent_dim_cont'], train_params['latent_dim_disc'])
    d_model = Discriminator()

    def init_weights(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0)

    g_model.apply(init_weights)
    d_model.apply(init_weights)
    # g_model.load_state_dict(torch.load('generator2.pkl'))
    # d_model.load_state_dict(torch.load('discriminator2.pkl'))
    g_loss_hist, d_loss_hist, img_list = train(g_model, d_model, dataLoader, train_params)
    plot(g_loss_hist, d_loss_hist)
    losses = {
        'g_loss': g_loss_hist,
        'd_loss': d_loss_hist,
        'img_list': img_list,
    }
    torch.save(losses, 'losses2.pkl')
    torch.save(g_model.state_dict(), 'generator2.pkl')
    torch.save(d_model.state_dict(), 'discriminator2.pkl')


def generate_random_fake_images():
    z_continuous1 = torch.randn(64, 90, 1, 1, device=device)
    z_continuous2 = torch.randn(64, 90, 1, 1, device=device)
    # z_continuous3 = torch.randn(64, 90, 1, 1, device=device)
    z_continuous3 = z_continuous1 + z_continuous2
    z_continuous4 = torch.randn(64, 90, 1, 1, device=device)
    z_continuous5 = torch.randn(64, 90, 1, 1, device=device)

    z_discrete1 = 2 * torch.randint(0, 2, (64, 10, 1, 1), device=device) - 1
    z_discrete2 = 2 * torch.randint(0, 2, (64, 10, 1, 1), device=device) - 1
    z_discrete3 = z_discrete1 * z_discrete2
    # z_discrete3 = 2 * torch.randint(0, 2, (64, 10, 1, 1), device=device) - 1
    z_discrete4 = torch.ones_like(z_discrete1)
    z_discrete5 = -torch.ones_like(z_discrete1)
    z_disc = [z_discrete1, z_discrete2, z_discrete3]
    z_cont = [z_continuous1, z_continuous2, z_continuous3, z_continuous4, z_continuous5]

    model = Generator(90, 10).to(device)
    model.load_state_dict(torch.load('generator2.pkl'))

    z_continuous = z_cont[0]
    for z_discrete in z_disc:
        z = torch.cat((z_continuous, z_discrete), dim=1)

        fake_images = model(z)
        fake = vutils.make_grid(fake_images, padding=2, normalize=True).cpu().detach().numpy()
        plt.imshow(np.transpose(fake, (1, 2, 0)))
        plt.show()
        plt.figure()


def reproduce_hw3():
    main()


def creative_images():
    model = Generator(90, 10).to(device)
    model.load_state_dict(torch.load('generator2.pkl'))
    z_cont1 = torch.randn(1, 90, 1, 1, device=device)
    z_cont2 = torch.randn(1, 90, 1, 1, device=device)
    z_disc = 2 * torch.randint(0, 2, (1, 10, 1, 1), device=device) - 1
    z_conts = [z_cont1, z_cont2]
    for z_cont in z_conts:
        z = torch.cat((z_cont, z_disc), dim=1)
        fake = ((model(z).squeeze(0).cpu().detach().numpy()) * 255).astype(int)
        plt.imshow(np.transpose(fake, (1, 2, 0)))
        plt.show()
        plt.figure()


def another_creative_images():
    model = Discriminator().to(device)
    model.load_state_dict(torch.load('discriminator2.pkl'))
    criterion = torch.nn.BCELoss()
    img = torch.randn(10, 3, 64, 64, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([img], 2e-4)
    label = torch.ones(10, device=device) * 0.0
    for epoch in tqdm(range(922)):
        res = model(img)
        loss = -1 * criterion(res, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    for i in range(10):
        img = img[i]
        fake = ((img.cpu().detach().numpy()) * 255).astype(int)
        plt.imshow(np.transpose(fake, (1, 2, 0)))
        plt.show()
        plt.figure()


if __name__ == '__main__':
    creative_images()
    # main()

