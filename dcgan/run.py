from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import os
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils
from torch.autograd import Variable
from dcgan import model
from utils import logging


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='/home/alan/datable/cifar10')
parser.add_argument('--num_workers', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--image_size', type=int, default=64)
parser.add_argument('--in_channels', type=int, default=3)
parser.add_argument('--z_dim', type=int, default=100)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--num_gpus', type=int, default=2)
parser.add_argument('--load_ckpt', type=bool, default=False)
parser.add_argument('--ckpt_path', type=str, default='/home/alan/datable/cifar10/ckpt')
parser.add_argument('--print_every', type=int, default=50)

opt = parser.parse_args()
os.makedirs(opt.ckpt_path, exist_ok=True)
logger = logging.Logger(opt.ckpt_path)
opt.seed = 1
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
cudnn.benchmark = True

transform = transforms.Compose([transforms.Scale(opt.image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = dset.CIFAR10(root=opt.dataset_path, train=True, download=False, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

D = model.Discriminator(opt).cuda()
G = model.Generator(opt).cuda()

if opt.load_ckpt:
  D.load_state_dict(torch.load(os.path.join(opt.ckpt_path, 'D.pth')))
  G.load_state_dict(torch.load(os.path.join(opt.ckpt_path, 'G.pth')))

criterion = nn.BCELoss().cuda()
optimizer_d = optim.Adam(D.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_g = optim.Adam(G.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

fixed_z = Variable(torch.randn(opt.batch_size, opt.z_dim, 1, 1).type(torch.cuda.FloatTensor))
labels_real = Variable(torch.zeros(opt.batch_size).fill_(1).type(torch.cuda.FloatTensor))
labels_fake = Variable(torch.zeros(opt.batch_size).fill_(0).type(torch.cuda.FloatTensor))

for epoch in range(opt.num_epochs):
  stats = logging.Statistics(['loss_d', 'loss_g'])

  for step, (images, _) in enumerate(data_loader, 0):
    batch_size = images.size(0)  # batch_size <= opt.batch_size
    G.zero_grad()
    D.zero_grad()

    ''' update D network: maximize log(D(x)) + log(1 - D(G(z))) '''
    # train with real
    x_real = Variable(images.type(torch.cuda.FloatTensor))
    output = D(x_real)
    loss_d_real = criterion(output, labels_real[:batch_size])
    loss_d_real.backward()
    D_x = output.data.mean()
    # train with fake
    z = Variable(torch.randn(batch_size, opt.z_dim, 1, 1).type(torch.cuda.FloatTensor))
    x_fake = G(z)
    output = D(x_fake.detach())
    loss_d_fake = criterion(output, labels_fake[:batch_size])
    loss_d_fake.backward()
    D_g_z1 = output.data.mean()
    # back propagation
    loss_d = loss_d_real+loss_d_fake
    optimizer_d.step()

    ''' update G network: maximize log(D(G(z))) '''
    # train with D's prediction
    output = D(x_fake)
    loss_g = criterion(output, labels_real[:batch_size])
    loss_g.backward()
    D_g_z2 = output.data.mean()
    # back propagation
    optimizer_g.step()

    # logging
    info = stats.update(batch_size, loss_d=loss_d.data[0], loss_g=loss_g.data[0])
    if opt.print_every > 0 and step%opt.print_every == 0:
      logger.log('epoch {}/{}, step {}/{}: {}, '
                 'D(x): {D_x:.4f}, D(G(z)): {D_g_z1:.4f}/{D_g_z2:.4f}'
                 .format(epoch, opt.num_epochs, step, len(data_loader), info, D_x=D_x, D_g_z1=D_g_z1, D_g_z2=D_g_z2))

    if step == 0:
      torchvision.utils.save_image(images, '%s/real_samples.png'%opt.ckpt_path)
      fake = G(fixed_z[:batch_size])
      torchvision.utils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png'%(opt.ckpt_path, epoch))

  info = stats.summary()
  logger.log('[Summary] epoch {}/{}: {}'.format(epoch, opt.num_epochs, info))

  torch.save(D.state_dict(), os.path.join(opt.ckpt_path, 'D.pth'))
  torch.save(G.state_dict(), os.path.join(opt.ckpt_path, 'G.pth'))