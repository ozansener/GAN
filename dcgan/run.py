from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import os
from dcgan.model import Generator, Discriminator
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', default='/home/alan/datable/cifar10', help='path to dataset')
parser.add_argument('--num_workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--image_size', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', default=True, help='enables cuda')
parser.add_argument('--num_gpus', default=2, type=int, help='number of GPUs to use')
parser.add_argument('--netG', default='', help='path to netG (to continue training)')
parser.add_argument('--netD', default='', help='path to netD (to continue training)')
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')

opt = parser.parse_args()
# print(opt)

try:
  os.makedirs(opt.outf)
except OSError:
  pass

opt.seed = 1
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
  print('WARNING: You have a CUDA device, so you should probably run with --cuda')

# Setup dataloader
# (Yuliang) Seems that they normalize to [0, 1] (They use sigmoid as G output)
transform = transforms.Compose([transforms.Scale(opt.image_size), transforms.ToTensor()])
dataset = dset.CIFAR10(root=opt.dataset_path, train=True, download=False, transform=transform)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

netG = Generator(opt)
if opt.netG != '':
  netG.load_state_dict(torch.load(opt.netG))
# print(netG)

netD = Discriminator(opt)
if opt.netD != '':
  netD.load_state_dict(torch.load(opt.netD))
# print(netD)

criterion = nn.BCELoss()

input = torch.FloatTensor(opt.batch_size, 3, opt.image_size, opt.image_size)
noise = torch.FloatTensor(opt.batch_size, opt.nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batch_size, opt.nz, 1, 1).normal_(0, 1)
label = torch.FloatTensor(opt.batch_size)
real_label = 1
fake_label = 0

if opt.cuda:
  netD.cuda()
  netG.cuda()
  criterion.cuda()
  input, label = input.cuda(), label.cuda()
  noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

input = Variable(input)
label = Variable(label)
noise = Variable(noise)
fixed_noise = Variable(fixed_noise)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

for epoch in range(opt.niter):
  for i, data in enumerate(dataloader, 0):
    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    # train with real
    netD.zero_grad()
    real_cpu, _ = data
    batch_size = real_cpu.size(0)
    input.data.resize_(real_cpu.size()).copy_(real_cpu)
    label.data.resize_(batch_size).fill_(real_label)

    output = netD(input)
    errD_real = criterion(output, label)
    errD_real.backward()
    D_x = output.data.mean()

    # train with fake
    noise.data.resize_(batch_size, opt.nz, 1, 1)
    noise.data.normal_(0, 1)
    fake = netG(noise)
    label.data.fill_(fake_label)
    output = netD(fake.detach())
    errD_fake = criterion(output, label)
    errD_fake.backward()
    D_G_z1 = output.data.mean()
    errD = errD_real+errD_fake
    optimizerD.step()

    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    netG.zero_grad()
    label.data.fill_(real_label)  # fake labels are real for generator cost
    output = netD(fake)
    errG = criterion(output, label)
    errG.backward()
    D_G_z2 = output.data.mean()
    optimizerG.step()

    print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
          %(epoch, opt.niter, i, len(dataloader), errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
    if i%100 == 0:
      vutils.save_image(real_cpu, '%s/real_samples.png'%opt.outf)
      fake = netG(fixed_noise)
      vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png'%(opt.outf, epoch))

  # do checkpointing
  torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth'%(opt.outf, epoch))
  torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth'%(opt.outf, epoch))
