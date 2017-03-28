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
from bigan import model
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
logger = logging.get_logger(opt.ckpt_path)
opt.seed = 1
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
cudnn.benchmark = True

transform = transforms.Compose([transforms.Scale(opt.image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = dset.CIFAR10(root=opt.dataset_path, train=True, download=False, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

D = model.Discriminator(opt).cuda()  # discriminator net D(x, z)
P = model.P(opt).cuda()  # generator net (decoder) P(x|z)
Q = model.Q(opt).cuda()  # inference net (encoder) Q(z|x)

if opt.load_ckpt:
  D.load_state_dict(torch.load(os.path.join(opt.ckpt_path, 'D.pth')))
  P.load_state_dict(torch.load(os.path.join(opt.ckpt_path, 'P.pth')))
  Q.load_state_dict(torch.load(os.path.join(opt.ckpt_path, 'Q.pth')))

criterion = nn.BCELoss().cuda()
optimizer_d = optim.Adam(D.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_pq = optim.Adam(P.parameters()+Q.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

fixed_z = Variable(torch.randn(opt.batch_size, opt.z_dim).type(torch.cuda.FloatTensor))
labels_p = Variable(torch.zeros(opt.batch_size).fill_(0).type(torch.cuda.FloatTensor))
labels_q = Variable(torch.zeros(opt.batch_size).fill_(1).type(torch.cuda.FloatTensor))

for epoch in range(opt.num_epochs):
  losses_d = logging.AverageMeter()
  losses_pq = logging.AverageMeter()
  images, batch_size = None, None

  for step, (images, _) in enumerate(data_loader, 0):
    batch_size = images.size(0)  # batch_size <= opt.batch_size

    ''' P network '''
    z_p = Variable(torch.randn(batch_size, opt.z_dim).type(torch.cuda.FloatTensor))
    x_p = P(z_p)

    ''' Q network '''
    x_q = Variable(images.type(torch.cuda.FloatTensor))
    z_q = Q(x_q)

    ''' D network '''
    output_p = D(x_p, z_p)
    output_q = D(x_q, z_q)

    loss_dp = criterion(output_p, labels_p[:batch_size])
    loss_dq = criterion(output_q, labels_q[:batch_size])
    loss_d = loss_dp+loss_dq
    loss_p = criterion(output_p, labels_q[:batch_size])
    loss_q = criterion(output_q, labels_p[:batch_size])
    loss_pq = loss_p+loss_q

    D.zero_grad()
    P.zero_grad()
    Q.zero_grad()
    loss_d.backward()
    optimizer_d.step()

    D.zero_grad()
    P.zero_grad()
    Q.zero_grad()
    loss_pq.backward()
    optimizer_pq.step()

    losses_d.update(loss_d.data[0], batch_size)
    losses_pq.update(loss_pq.data[0], batch_size)

    if opt.print_every > 0 and step%opt.print_every == 0:
      logger.info('epoch {}/{}, step {}/{}: '
                  'loss_d={loss_d.val:.4f}, avg loss_d={loss_d.avg:.4f}, '
                  'loss_pq={loss_pq.val:.4f}, avg loss_pq={loss_pq.avg:.4f}'
                  .format(epoch, opt.num_epochs, step, len(data_loader), loss_d=losses_d, loss_pq=losses_pq))

    if step == 0:
      torchvision.utils.save_image(images, '%s/real_samples.png'%opt.ckpt_path)
      fake = P(fixed_z[:batch_size])
      torchvision.utils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png'%(opt.ckpt_path, epoch))

  torch.save(D.state_dict(), os.path.join(opt.ckpt_path, 'D.pth'))
  torch.save(P.state_dict(), os.path.join(opt.ckpt_path, 'P.pth'))
  torch.save(Q.state_dict(), os.path.join(opt.ckpt_path, 'Q.pth'))
