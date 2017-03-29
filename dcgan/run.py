from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import os
import shutil
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
parser.add_argument('--image_size', type=int, default=64)
parser.add_argument('--in_channels', type=int, default=3)
parser.add_argument('--num_gpus', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=20)

# logging
parser.add_argument('--clean_ckpt', type=bool, default=True)
parser.add_argument('--load_ckpt', type=bool, default=False)
parser.add_argument('--ckpt_path', type=str, default='/home/alan/datable/cifar10/ckpt')
parser.add_argument('--print_every', type=int, default=50)

# hyperparameters
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--z_dim', type=int, default=100)
parser.add_argument('--lr_adam', type=float, default=2e-4)
parser.add_argument('--lr_rmsprop', type=float, default=2e-4)
parser.add_argument('--beta1', type=float, default=0.5, help='for adam')
parser.add_argument('--slope', type=float, default=0.2, help='for leaky ReLU')
parser.add_argument('--std', type=float, default=0.02, help='for weight')
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--clamp', type=float, default=1e-2)
parser.add_argument('--wasserstein', type=bool, default=False)

opt = parser.parse_args()
if opt.clean_ckpt:
  shutil.rmtree(opt.ckpt_path)
os.makedirs(opt.ckpt_path, exist_ok=True)
logger = logging.Logger(opt.ckpt_path)
opt.seed = 1
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
cudnn.benchmark = True
EPS = 1e-12

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
if opt.wasserstein:
  optimizer_d = optim.RMSprop(D.parameters(), lr=opt.lr_rmsprop)
  optimizer_g = optim.RMSprop(G.parameters(), lr=opt.lr_rmsprop)
else:
  optimizer_d = optim.Adam(D.parameters(), lr=opt.lr_adam, betas=(opt.beta1, 0.999))
  optimizer_g = optim.Adam(G.parameters(), lr=opt.lr_adam, betas=(opt.beta1, 0.999))

fixed_z = Variable(torch.randn(opt.batch_size, opt.z_dim, 1, 1).type(torch.cuda.FloatTensor))

for epoch in range(opt.num_epochs):
  stats = logging.Statistics(['loss_d', 'loss_g'])

  for step, (images, _) in enumerate(data_loader, 0):
    batch_size = images.size(0)  # batch_size <= opt.batch_size
    G.zero_grad()
    D.zero_grad()

    ''' update D network: maximize log(D(x)) + log(1 - D(G(z))) '''
    # train with real
    x_real = Variable(images.type(torch.cuda.FloatTensor))
    output_d_real = D(x_real)
    D_x = output_d_real.data.mean()
    # train with fake
    z = Variable(torch.randn(batch_size, opt.z_dim, 1, 1).type(torch.cuda.FloatTensor))
    x_fake = G(z)
    output_d_fake = D(x_fake.detach())
    D_g_z1 = output_d_fake.data.mean()

    # loss & back propagation
    if opt.wasserstein:
      loss_d = -torch.mean(output_d_real)+torch.mean(output_d_fake)
    else:
      loss_d = -torch.mean(torch.log(output_d_real+EPS)+torch.log(1-output_d_fake+EPS))
    loss_d.backward()
    optimizer_d.step()

    if opt.wasserstein:
      for p in D.parameters():
        p.data.clamp_(-opt.clamp, opt.clamp)

    ''' update G network: maximize log(D(G(z))) '''
    # train with D's prediction
    output = D(x_fake)
    D_g_z2 = output.data.mean()
    # loss & back propagation
    if opt.wasserstein:
      loss_g = -torch.mean(output)
    else:
      # Minimax game: minimize log(1 - D(G(z))) -- Seems not work
      # loss_g = torch.mean(torch.log(1-output+EPS))
      # Non-saturating game: maximize log(D(G(z)))
      loss_g = -torch.mean(torch.log(output+EPS))

    loss_g.backward()
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
