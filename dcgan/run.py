from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
from model import Generator, Discriminator
from utils import *
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils
from torch.autograd import Variable


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='/home/alan/datable/cifar10')
parser.add_argument('--num_workers', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--image_size', type=int, default=64)
parser.add_argument('--in_channels', type=int, default=3)
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--num_gpus', type=int, default=2)
parser.add_argument('--load_ckpt', type=bool, default=False)
parser.add_argument('--ckpt_path', type=str, default='/home/alan/datable/cifar10/ckpt')
parser.add_argument('--print_every', type=int, default=50)

opt = parser.parse_args()
os.makedirs(opt.ckpt_path, exist_ok=True)
logger = get_logger(opt.ckpt_path)
opt.seed = 1
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
cudnn.benchmark = True

transform = transforms.Compose([transforms.Scale(opt.image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = dset.CIFAR10(root=opt.dataset_path, train=True, download=False, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

net_g = Generator(opt).cuda()
net_d = Discriminator(opt).cuda()

if opt.load_ckpt:
  net_g.load_state_dict(torch.load(os.path.join(opt.ckpt_path, 'net_g.pth')))
  net_d.load_state_dict(torch.load(os.path.join(opt.ckpt_path, 'net_d.pth')))

criterion = nn.BCELoss().cuda()
optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_d = optim.Adam(net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

fixed_noise = Variable(torch.zeros(opt.batch_size, opt.nz, 1, 1).type(torch.cuda.FloatTensor).normal_(0, 1))
labels_real = Variable(torch.zeros(opt.batch_size).fill_(1).type(torch.cuda.FloatTensor))
labels_fake = Variable(torch.zeros(opt.batch_size).fill_(0).type(torch.cuda.FloatTensor))

for epoch in range(opt.num_epochs):
  losses_g = AverageMeter()
  losses_d = AverageMeter()
  images, batch_size = None, None

  for step, (images, _) in enumerate(data_loader, 0):
    batch_size = images.size(0)
    images_real = Variable(images.type(torch.cuda.FloatTensor))
    noise = Variable(torch.zeros(batch_size, opt.nz, 1, 1).normal_(0, 1).type(torch.cuda.FloatTensor))

    ''' (1) Update D network: maximize log(D(x)) + log(1 - D(G(z))) '''
    net_d.zero_grad()

    # train with real
    output = net_d(images_real)
    loss_d_real = criterion(output, labels_real[:batch_size])
    loss_d_real.backward()
    D_x = output.data.mean()

    # train with fake
    images_fake = net_g(noise)
    output = net_d(images_fake.detach())
    loss_d_fake = criterion(output, labels_fake[:batch_size])
    loss_d_fake.backward()
    D_g_z1 = output.data.mean()

    loss_d = loss_d_real+loss_d_fake
    optimizer_d.step()

    ''' (2) Update G network: maximize log(D(G(z))) '''
    net_g.zero_grad()

    output = net_d(images_fake)
    loss_g = criterion(output, labels_real[:batch_size])
    loss_g.backward()
    D_g_z2 = output.data.mean()

    optimizer_g.step()

    losses_g.update(loss_g.data[0], batch_size)
    losses_d.update(loss_d.data[0], batch_size)

    if opt.print_every > 0 and step%opt.print_every == 0:
      logger.info('epoch {}/{}, step {}/{}: '
                  'loss_g={loss_g.val:.4f}, avg loss_g={loss_g.avg:.4f}, '
                  'loss_d={loss_d.val:.4f}, avg loss_d={loss_d.avg:.4f}, '
                  'D(x): {D_x:.4f}, D(G(z)): {D_g_z1:.4f}/{D_g_z2:.4f}'
                  .format(epoch, opt.num_epochs, step, len(data_loader), loss_g=losses_g, loss_d=losses_d,
                          D_x=D_x, D_g_z1=D_g_z1, D_g_z2=D_g_z2))

    if step == 0:
      torchvision.utils.save_image(images, '%s/real_samples.png'%opt.ckpt_path)
      fake = net_g(fixed_noise[:batch_size])
      torchvision.utils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png'%(opt.ckpt_path, epoch))

  torch.save(net_g.state_dict(), os.path.join(opt.ckpt_path, 'net_g.pth'))
  torch.save(net_d.state_dict(), os.path.join(opt.ckpt_path, 'net_d.pth'))
