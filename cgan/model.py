from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import torch.nn as nn
import torch.nn.parallel


class Generator(nn.Module):
  def __init__(self, opt, ngf=64):
    super(Generator, self).__init__()
    self.opt = opt

    # input dim: num_channels x 256 x 256
    self.inference_e1 = nn.Conv2d(opt.num_channels, ngf, 4, stride=2, padding=1)
    # input dim: ngf x 128 x 128
    self.inference_e2 = nn.Sequential(
      nn.LeakyReLU(opt.slope, inplace=True),
      nn.Conv2d(opt.num_channels, ngf, 4, stride=2, padding=1),
      nn.BatchNorm2d(ngf*2)
    )
    # input dim: ngf*2 x 64 x 64
    self.inference_e3 = nn.Sequential(
      nn.LeakyReLU(opt.slope, inplace=True),
      nn.Conv2d(ngf*2, ngf*4, 4, stride=2, padding=1),
      nn.BatchNorm2d(ngf*4)
    )
    # input dim: ngf*4 x 32 x 32
    self.inference_e4 = nn.Sequential(
      nn.LeakyReLU(opt.slope, inplace=True),
      nn.Conv2d(ngf*4, ngf*8, 4, stride=2, padding=1),
      nn.BatchNorm2d(ngf*8)
    )
    # input dim: ngf*8 x 16 x 16
    self.inference_e5 = nn.Sequential(
      nn.LeakyReLU(opt.slope, inplace=True),
      nn.Conv2d(ngf*8, ngf*8, 4, stride=2, padding=1),
      nn.BatchNorm2d(ngf*8)
    )
    # input dim: ngf*8 x 8 x 8
    self.inference_e6 = nn.Sequential(
      nn.LeakyReLU(opt.slope, inplace=True),
      nn.Conv2d(ngf*8, ngf*8, 4, stride=2, padding=1),
      nn.BatchNorm2d(ngf*8)
    )
    # input dim: ngf*8 x 4 x 4
    self.inference_e7 = nn.Sequential(
      nn.LeakyReLU(opt.slope, inplace=True),
      nn.Conv2d(ngf*8, ngf*8, 4, stride=2, padding=1),
      nn.BatchNorm2d(ngf*8)
    )
    # input dim: ngf*8 x 2 x 2
    self.inference_e8 = nn.Sequential(
      nn.LeakyReLU(opt.slope, inplace=True),
      nn.Conv2d(ngf*8, ngf*8, 4, stride=2, padding=1),
    )
    # input dim: ngf*8 x 1 x 1
    self.inference_d1 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.ConvTranspose2d(ngf*8, ngf*8, 4, stride=2, padding=1),
      nn.BatchNorm2d(ngf*8),
      nn.Dropout2d(p=opt.dropout)
    )
    # input dim: ngf*8 x 2 x 2
    self.inference_d2 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.ConvTranspose2d(ngf*8*2, ngf*8, 4, stride=2, padding=1),
      nn.BatchNorm2d(ngf*8),
      nn.Dropout2d(p=opt.dropout)
    )
    # input dim: ngf*8 x 4 x 4
    self.inference_d3 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.ConvTranspose2d(ngf*8*2, ngf*8, 4, stride=2, padding=1),
      nn.BatchNorm2d(ngf*8),
      nn.Dropout2d(p=opt.dropout)
    )
    # input dim: ngf*8 x 8 x 8
    self.inference_d4 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.ConvTranspose2d(ngf*8*2, ngf*8, 4, stride=2, padding=1),
      nn.BatchNorm2d(ngf*8)
    )
    # input dim: ngf*8 x 16 x 16
    self.inference_d5 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.ConvTranspose2d(ngf*8*2*2, ngf*4, 4, stride=2, padding=1),
      nn.BatchNorm2d(ngf*4)
    )
    # input dim: ngf*4 x 32 x 32
    self.inference_d6 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.ConvTranspose2d(ngf*4*2, ngf*2, 4, stride=2, padding=1),
      nn.BatchNorm2d(ngf*2)
    )
    # input dim: ngf*2 x 64 x 64
    self.inference_d7 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.ConvTranspose2d(ngf*2*2, ngf, 4, stride=2, padding=1),
      nn.BatchNorm2d(ngf)
    )
    # input dim: ngf x 128 x 128
    self.inference_d8 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.ConvTranspose2d(ngf*2, opt.num_channels, 4, stride=2, padding=1),
      nn.Tanh()
    )
    # output dim: num_channels x 256 x 256

    self.reset_parameters()

  def reset_parameters(self):
    for m in self.modules():
      if isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(0.0, self.opt.std)
        m.bias.data.zero_()
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, self.opt.std)
        m.bias.data.zero_()

  def forward(self, y, z):
    e1 = self.inference_e1(y)
    e2 = self.inference_e2(e1)
    e3 = self.inference_e3(e2)
    e4 = self.inference_e4(e3)
    e5 = self.inference_e5(e4)
    e6 = self.inference_e6(e5)
    e7 = self.inference_e7(e6)
    e8 = self.inference_e8(e7)
    d1 = self.inference_d1(e8)
    d2 = self.inference_d2(torch.cat((d1, e7), 1))
    d3 = self.inference_d3(torch.cat((d2, e6), 1))
    d4 = self.inference_d4(torch.cat((d3, e5), 1))
    d5 = self.inference_d5(torch.cat((d4, e4), 1))
    d6 = self.inference_d6(torch.cat((d5, e3), 1))
    d7 = self.inference_d7(torch.cat((d6, e2), 1))
    d8 = self.inference_d8(torch.cat((d7, e1), 1))
    return d8


class Discriminator(nn.Module):
  def __init__(self, opt, ndf=64):
    super(Discriminator, self).__init__()
    self.opt = opt

    self.inference = nn.Sequential(
      # input dim: num_channels x 256 x 256
      nn.Conv2d(opt.num_channels*2, ndf, 4, stride=2, padding=1),
      nn.LeakyReLU(opt.slope, inplace=True),
      # state dim: ndf x 128 x 128
      nn.Conv2d(ndf, ndf*2, 4, stride=2, padding=1),
      nn.BatchNorm2d(ndf*2),
      nn.LeakyReLU(opt.slope, inplace=True),
      # state dim: ndf*2 x 64 x 64
      nn.Conv2d(ndf*2, ndf*4, 4, stride=2, padding=1),
      nn.BatchNorm2d(ndf*4),
      nn.LeakyReLU(opt.slope, inplace=True),
      # state dim: ndf*4 x 32 x 32
      nn.Conv2d(ndf*4, ndf*8, 4, stride=1, padding=1),
      nn.BatchNorm2d(ndf*4),
      nn.LeakyReLU(opt.slope, inplace=True),
      # state dim: ndf*8 x 31 x 31
      nn.Conv2d(ndf*8, 1, 4, stride=1, padding=1),
      nn.Sigmoid()
      # output dim: ndf*8 x 30 x 30
    )

    self.reset_parameters()

  def reset_parameters(self):
    for m in self.modules():
      if isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(0.0, self.opt.std)
        m.bias.data.zero_()
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, self.opt.std)
        m.bias.data.zero_()

  def forward(self, x):
    return self.inference(x)
