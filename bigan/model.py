from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import torch.nn as nn

class P(nn.Module):
  def __init__(self, opt):
    super(P, self).__init__()
    self.num_gpus = opt.num_gpus

    self.inference = nn.Sequential(
      # input dim: z_dim x 1 x 1
      nn.ConvTranspose2d(opt.z_dim, 256, 4, stride=1, bias=False),
      nn.BatchNorm2d(256),
      nn.LeakyReLU(0.01, inplace=True),
      # state dim: 256 x 4 x 4
      nn.ConvTranspose2d(256, 128, 4, stride=2, bias=False),
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.01, inplace=True),
      # state dim: 128 x 10 x 10
      nn.ConvTranspose2d(128, 64, 4, stride=1, bias=False),
      nn.BatchNorm2d(64),
      nn.LeakyReLU(0.01, inplace=True),
      # state dim: 64 x 13 x 13
      nn.ConvTranspose2d(64, 32, 4, stride=2, bias=False),
      nn.BatchNorm2d(32),
      nn.LeakyReLU(0.01, inplace=True),
      # state dim: 32 x 28 x 28
      nn.ConvTranspose2d(32, 32, 5, stride=1, bias=False),
      nn.BatchNorm2d(32),
      nn.LeakyReLU(0.01, inplace=True),
      # state dim: 32 x 32 x 32
      nn.Conv2d(32, opt.in_channels, 1, stride=1, bias=False),
      nn.Tanh()
      # output dim: in_channels x 32 x 32
    )

    self.reset_parameters()

  def reset_parameters(self):
    # TODO: fix init
    for m in self.modules():
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(0.0, 0.02)
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.zero_()

  def forward(self, z):
    gpu_ids = None
    if isinstance(z.data, torch.cuda.FloatTensor) and self.num_gpus > 1:
      gpu_ids = range(self.num_gpus)
    return nn.parallel.data_parallel(self.inference, z, gpu_ids)


class Q(nn.Module):
  def __init__(self, opt):
    super(Q, self).__init__()
    self.num_gpus = opt.num_gpus

    self.inference = nn.Sequential(
      # input dim: in_channels x 32 x 32
      nn.Conv2d(3, 32, 5, stride=1, bias=False),
      nn.BatchNorm2d(32),
      nn.LeakyReLU(0.01, inplace=True),
      # state dim: 32 x 28 x 28
      nn.Conv2d(32, 64, 4, stride=2, bias=False),
      nn.BatchNorm2d(64),
      nn.LeakyReLU(0.01, inplace=True),
      # state dim: 64 x 13 x 13
      nn.Conv2d(64, 128, 4, stride=1, bias=False),
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.01, inplace=True),
      # state dim: 128 x 10 x 10
      nn.Conv2d(128, 256, 4, stride=2, bias=False),
      nn.BatchNorm2d(256),
      nn.LeakyReLU(0.01, inplace=True),
      # state dim: 256 x 4 x 4
      nn.Conv2d(256, 512, 4, stride=1, bias=False),
      nn.BatchNorm2d(512),
      nn.LeakyReLU(0.01, inplace=True),
      # state dim: 512 x 1 x 1
      nn.Conv2d(512, 512, 1, stride=1, bias=False),
      nn.BatchNorm2d(512),
      nn.LeakyReLU(0.01, inplace=True),
      # state dim: 512 x 1 x 1
      nn.Conv2d(512, opt.z_dim, 1, stride=1, bias=False)
      # output dim: opt.z_dim x 1 x 1
    )

    self.reset_parameters()

  def reset_parameters(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.zero_()

  def forward(self, x):
    gpu_ids = None
    if isinstance(x.data, torch.cuda.FloatTensor) and self.num_gpus > 1:
      gpu_ids = range(self.num_gpus)
    return nn.parallel.data_parallel(self.inference, x, gpu_ids)


class Discriminator(nn.Module):
  def __init__(self, opt):
    super(Discriminator, self).__init__()
    self.num_gpus = opt.num_gpus

    self.inference_x = nn.Sequential(
      # input dim: in_channels x 32 x 32 (no bn)
      nn.Conv2d(3, 32, 5, stride=1, bias=False),
      nn.LeakyReLU(0.01, inplace=True),
      # state dim: 32 x 28 x 28
      nn.Conv2d(32, 64, 4, stride=2, bias=False),
      nn.BatchNorm2d(64),
      nn.LeakyReLU(0.01, inplace=True),
      # state dim: 64 x 13 x 13
      nn.Conv2d(64, 128, 4, stride=1, bias=False),
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.01, inplace=True),
      # state dim: 128 x 10 x 10
      nn.Conv2d(128, 256, 4, stride=2, bias=False),
      nn.BatchNorm2d(256),
      nn.LeakyReLU(0.01, inplace=True),
      # state dim: 256 x 4 x 4
      nn.Conv2d(256, 512, 4, stride=1, bias=False),
      nn.BatchNorm2d(512),
      nn.LeakyReLU(0.01, inplace=True)
      # state dim: 512 x 1 x 1
    )

    self.inference_z = nn.Sequential(
      # input dim: z_dim x 1 x 1 (no bn)
      nn.Conv2d(opt.z_dim, 512, 1, stride=1, bias=False),
      nn.LeakyReLU(0.01, inplace=True),
      # state dim: 512 x 1 x 1 (no bn)
      nn.Conv2d(512, 512, 1, stride=1, bias=False),
      nn.LeakyReLU(0.01, inplace=True)
      # output dim: 512 x 1 x 1
    )

    self.inference_joint = nn.Sequential(
      # input dim: 1024 x 1 x 1 (no bn)
      nn.Conv2d(1024, 1024, 1, stride=1, bias=False),
      nn.LeakyReLU(0.01, inplace=True),
      # state dim: 1024 x 1 x 1 (no bn)
      nn.Conv2d(1024, 1024, 1, stride=1, bias=False),
      nn.LeakyReLU(0.01, inplace=True),
      # state dim: 1024 x 1 x 1 (no bn)
      nn.Conv2d(1024, 1, 1, stride=1, bias=False),
      nn.Sigmoid()
      # output dim: 1 x 1 x 1
    )

    self.reset_parameters()

  def reset_parameters(self):
    # TODO: fix init
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.zero_()

    for m in self.inference_z:
      if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.5)
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.zero_()

  def forward(self, x, z):
    gpu_ids = None
    if isinstance(x.data, torch.cuda.FloatTensor) and self.num_gpus > 1:
      gpu_ids = range(self.num_gpus)
    output_x = nn.parallel.data_parallel(self.inference_x, x, gpu_ids)
    output_z = nn.parallel.data_parallel(self.inference_z, z, gpu_ids)
    output = nn.parallel.data_parallel(self.inference_joint, torch.cat((output_x, output_z), 1), gpu_ids)
    return output
