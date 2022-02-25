import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets
import imageio
import numpy as np
import matplotlib
import configuration as configs

from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm


relu_slope = 0.2
dropout_prob = 0.3

class Generator(nn.Module):
  def __init__(self, nz):
    super(Generator, self).__init__()
    self.nz = nz

    self.main = nn.Sequential(
      nn.Linear(self.nz, 256),
      nn.LeakyReLU(relu_slope),

      nn.Linear(256, 512),
      nn.LeakyReLU(relu_slope),

      nn.Linear(512, 1024),
      nn.LeakyReLU(relu_slope),

      nn.Linear(1024, 784),
      nn.Tanh()
    )

  def forward(self, x):
    return self.main(x).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.n_input = 784

    self.main = nn.Sequential(
      nn.Linear(self.n_input, 1024),
      nn.LeakyReLU(relu_slope),
      nn.Dropout(dropout_prob),

      nn.Linear(1024, 512),
      nn.LeakyReLU(relu_slope),
      nn.Dropout(dropout_prob),

      nn.Linear(512, 256),
      nn.LeakyReLU(relu_slope),
      nn.Dropout(dropout_prob),

      nn.Linear(256, 1),
      nn.Sigmoid(),

    )

  def forward(self, x):
    x = x.view(-1, self.n_input)
    return self.main(x)


def labels_real(size, device):
  data = torch.ones(size, 1)
  return data.to(device)


def labels_fake(size, device):
  data = torch.zeros(size, 1)
  return data.to(device)


def create_noise(sample_size, nz, device):
  return torch.randn(sample_size, nz).to(device)


def train_discriminator_step(discriminator, optimizer, criterion, data_real, data_fake, config):
  b_size = data_real.size(0)
  real_label = labels_real(b_size, config.device)
  fake_label = labels_fake(b_size, config.device)

  optimizer.zero_grad()

  output_real = discriminator(data_real)
  loss_real = criterion(output_real, real_label)

  output_fake = discriminator(data_fake)
  loss_fake = criterion(output_fake, fake_label)

  loss_real.backward()
  loss_fake.backward()

  optimizer.step()

  return loss_real + loss_fake


def train_generator_step(optimizer, discriminator, criterion, data_fake, config):
  b_size = data_fake.size(0)
  real_label = labels_real(b_size, config.device)

  optimizer.zero_grad()

  output = discriminator(data_fake)
  loss = criterion(output, real_label)

  loss.backward()
  optimizer.step()

  return loss


if __name__ == '__main__':
  config = configs.test_config

  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
  ])
  to_pil_image = transforms.ToPILImage()

  train_data = datasets.MNIST(root="inputs/data", train=True, download=True, transform=transform)
  train_loader = DataLoader(train_data, config.batch_size, shuffle=True)

  generator = Generator(config.nz).to(config.device)
  discriminator = Discriminator().to(config.device)

  optim_g = optim.Adam(generator.parameters(), config.learning_rate)
  optim_d = optim.Adam(discriminator.parameters(), config.learning_rate)

  criterion = nn.BCELoss()

  losses_g = []
  losses_d = []
  images = []

  generator.train()
  discriminator.train()

  noise = create_noise(config.sample_size, config.nz, config.device)

  for epoch in range(config.epochs):
    loss_g = 0.0
    loss_d = 0.0
    for bi, data in tqdm(enumerate(train_loader), total=int(len(train_data)/train_loader.batch_size)):
      image, _ = data
      image = image.to(config.device)
      b_size = len(image)
      for step in range(config.k):
        data_fake = generator(create_noise(b_size, config.nz, config.device))
        loss_d += train_discriminator_step(discriminator, optim_d, criterion, image, data_fake, config)
      data_fake = generator(create_noise(b_size, config.nz, config.device))
      loss_g += train_generator_step(optim_g, discriminator, criterion, data_fake, config)

    generator.eval()
    generator_images = generator(noise)
    generator_images = make_grid(generator_images)

    save_image(generator_images, f"outputs/gen_img{epoch}.png")
    images.append(generator_images)
    epoch_loss_g = loss_g/bi
    epoch_loss_d = loss_d/bi

    losses_d.append(epoch_loss_d.detach().cpu())
    losses_g.append(epoch_loss_g.detach().cpu())

    print(f"Epoch {epoch} of {config.epochs}")
    print(f"Generator loss: {epoch_loss_g:.8f}, Discriminator loss: {epoch_loss_d:.8f}")

  print('Done Training')
  torch.save(generator.state_dict(), 'outputs/generator.pth')

  images = [np.array(to_pil_image(img)) for img in images]
  imageio.mimsave('outputs/generator_images.gif', images)

  plt.figure()
  plt.plot(losses_g, label='Generator loss')
  plt.plot(losses_d, label='Discriminator loss')
  plt.legend()
  plt.savefig('outputs/loss.png')

