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
parser.add_argument('--num_gpus', default=2, type=int, help='number of GPUs to use')
parser.add_argument('--netG', default='', help='path to netG (to continue training)')
parser.add_argument('--netD', default='', help='path to netD (to continue training)')
parser.add_argument('--outf', default='/home/alan/datable/cifar10/ckpt', help='folder to output images and model checkpoints')

opt = parser.parse_args()
opt.seed = 1
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
cudnn.benchmark = True

try:
  os.makedirs(opt.outf)
except OSError:
  pass

transform = transforms.Compose([transforms.Scale(opt.image_size), transforms.ToTensor()])
dataset = dset.CIFAR10(root=opt.dataset_path, train=True, download=False, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

netG = Generator(opt).cuda()
if opt.netG != '':
  netG.load_state_dict(torch.load(opt.netG))

netD = Discriminator(opt).cuda()
if opt.netD != '':
  netD.load_state_dict(torch.load(opt.netD))

criterion = nn.BCELoss().cuda()
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

noise = Variable(torch.zeros(opt.batch_size, opt.nz, 1, 1).type(torch.cuda.FloatTensor))
fixed_noise = Variable(torch.zeros(opt.batch_size, opt.nz, 1, 1).type(torch.cuda.FloatTensor).normal_(0, 1))
labels_real = Variable(torch.zeros(opt.batch_size).fill_(1).type(torch.cuda.FloatTensor))
labels_fake = Variable(torch.zeros(opt.batch_size).fill_(0).type(torch.cuda.FloatTensor))

for epoch in range(opt.niter):
  for i, (images, _) in enumerate(data_loader, 0):
    images_real = Variable(images.type(torch.cuda.FloatTensor))
    noise.data.normal_(0, 1)
    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    # train with real
    netD.zero_grad()
    output = netD(images_real)
    errD_real = criterion(output, labels_real)
    errD_real.backward()
    D_x = output.data.mean()

    # train with fake
    images_fake = netG(noise)
    output = netD(images_fake.detach())
    errD_fake = criterion(output, labels_fake)
    errD_fake.backward()
    D_G_z1 = output.data.mean()
    errD = errD_real+errD_fake
    optimizerD.step()

    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    netG.zero_grad()
    output = netD(images_fake)
    errG = criterion(output, labels_fake)
    errG.backward()
    D_G_z2 = output.data.mean()
    optimizerG.step()

    print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
          %(epoch, opt.niter, i, len(data_loader), errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
    if i%100 == 0:
      vutils.save_image(images, '%s/real_samples.png'%opt.outf)
      fake = netG(fixed_noise)
      vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png'%(opt.outf, epoch))

  torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth'%(opt.outf, epoch))
  torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth'%(opt.outf, epoch))
