from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import torch.nn as nn
import torch.nn.parallel


class Generator(nn.Module):
  def __init__(self, opt):
    super(Generator, self).__init__()
    self.num_gpus = opt.num_gpus

    self.main = nn.Sequential(
      # input: Z
      nn.ConvTranspose2d(opt.nz, opt.ngf*8, 4, 1, 0, bias=False),
      nn.BatchNorm2d(opt.ngf*8),
      nn.ReLU(inplace=True),
      # state dim: ngf*8 x 4 x 4
      nn.ConvTranspose2d(opt.ngf*8, opt.ngf*4, 4, 2, 1, bias=False),
      nn.BatchNorm2d(opt.ngf*4),
      nn.ReLU(inplace=True),
      # state dim: ngf*4 x 8 x 8
      nn.ConvTranspose2d(opt.ngf*4, opt.ngf*2, 4, 2, 1, bias=False),
      nn.BatchNorm2d(opt.ngf*2),
      nn.ReLU(inplace=True),
      # state dim: ngf*2 x 16 x 16
      nn.ConvTranspose2d(opt.ngf*2, opt.ngf, 4, 2, 1, bias=False),
      nn.BatchNorm2d(opt.ngf),
      nn.ReLU(inplace=True),
      # state dim: ngf x 32 x 32
      nn.ConvTranspose2d(opt.ngf, opt.in_channels, 4, 2, 1, bias=False),
      nn.Tanh()
      # state dim: in_channels x 64 x 64
    )

    for m in self.modules():
      if isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(0.0, 0.02)
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.zero_()

  def forward(self, z):
    gpu_ids = None
    if isinstance(z.data, torch.cuda.FloatTensor) and self.num_gpus > 1:
      gpu_ids = range(self.num_gpus)
    return nn.parallel.data_parallel(self.main, z, gpu_ids)


class Discriminator(nn.Module):
  def __init__(self, opt):
    super(Discriminator, self).__init__()
    self.num_gpus = opt.num_gpus

    self.main = nn.Sequential(
      # input dim: in_channels x 64 x 64
      nn.Conv2d(opt.in_channels, opt.ndf, 4, 2, 1, bias=False),
      nn.LeakyReLU(0.2, inplace=True),
      # state dim: ndf x 32 x 32
      nn.Conv2d(opt.ndf, opt.ndf*2, 4, 2, 1, bias=False),
      nn.BatchNorm2d(opt.ndf*2),
      nn.LeakyReLU(0.2, inplace=True),
      # state dim: ndf*2 x 16 x 16
      nn.Conv2d(opt.ndf*2, opt.ndf*4, 4, 2, 1, bias=False),
      nn.BatchNorm2d(opt.ndf*4),
      nn.LeakyReLU(0.2, inplace=True),
      # state dim: ndf*4 x 8 x 8
      nn.Conv2d(opt.ndf*4, opt.ndf*8, 4, 2, 1, bias=False),
      nn.BatchNorm2d(opt.ndf*8),
      nn.LeakyReLU(0.2, inplace=True),
      # state dim: ndf*8 x 4 x 4
      nn.Conv2d(opt.ndf*8, 1, 4, 1, 0, bias=False),
      nn.Sigmoid()
    )

    for m in self.modules():
      if isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(0.0, 0.02)
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.zero_()

  def forward(self, x):
    gpu_ids = None
    if isinstance(x.data, torch.cuda.FloatTensor) and self.num_gpus > 1:
      gpu_ids = range(self.num_gpus)
    output = nn.parallel.data_parallel(self.main, x, gpu_ids)
    return output.view(-1, 1)