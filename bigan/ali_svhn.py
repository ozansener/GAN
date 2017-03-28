# A PyTorch replication of ALI_svhn.py
# Credit: 
# https://github.com/pytorch/examples/blob/master/dcgan/main.py
# https://github.com/IshmaelBelghazi/ALI/blob/master/experiments/ali_svhn.py
#
# Author: Yuliang Zou
from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

try:
	import ipdb
except:
	import pdb as ipdb

# Arg
parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', default='../', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0001')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')

opt = parser.parse_args()
print(opt)

# Setup dataloader
# (Yuliang) Seems that they normalize to [0, 1] (They use sigmoid as G output)
dataset = dsets.CIFAR10(root=opt.dataroot, train=True,
			# download=True,
			transform=transforms.Compose([
				transforms.Scale(opt.imageSize),
				transforms.ToTensor()
		]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
	shuffle=True, num_workers=int(opt.workers))


# custom weights init
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.01)
		# m.bias.data.fill_(0.0)    # this should be default setting
	# (Yuliang) Not sure how to set init value for bn layers??!

# Encoder: x -> z
class _Gz(nn.Module):
	def __init__(self, ngpu):
		super(_Gz, self).__init__()
		self.ngpu = ngpu
		self.main = nn.Sequential(
			# input is 3 x 32 x 32
			# input, output, kernel
			nn.Conv2d(3, 32, 5),    # no padding
			nn.BatchNorm2d(32),
			nn.LeakyReLU(0.01, inplace=True),
			
			nn.Conv2d(32, 64, 4, stride=2),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.01, inplace=True),
			
			nn.Conv2d(64, 128, 4),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.01, inplace=True),			

			nn.Conv2d(128, 256, 4, stride=2),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.01, inplace=True),

			nn.Conv2d(256, 512, 4),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.01, inplace=True),
			
			nn.Conv2d(512, 512, 1),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.01, inplace=True),

			# mean and log var
			nn.Conv2d(512, 512, 1)
		)

	# Stochasitic encoding
	# Credit: https://github.com/pytorch/examples/blob/master/vae/main.py
	def reparametrize(self, mu, logvar):
		std = logvar.mul(0.5).exp_()
		# if args.cuda:
		if not True:
			eps = torch.cuda.FloatTensor(std.size()).normal_()
		else:
			eps = torch.FloatTensor(std.size()).normal_()
		eps = Variable(eps)
		return eps.mul(std).add_(mu)

	def forward(self, input):
		gpu_ids = None
		if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
			gpu_ids = range(self.ngpu)
		enc = nn.parallel.data_parallel(self.main, input, gpu_ids)
		# split to get mu and logvar (Not sure)
		mu, logvar = torch.split(enc, 256, dim=1)
		output = self.reparametrize(mu, logvar)
		return output

# Generator: z -> x
class _Gx(nn.Module):
	def __init__(self, ngpu):
		super(_Gx, self).__init__()
		self.ngpu = ngpu
		self.main = nn.Sequential(
			# input is 256 x 1 x 1
			nn.ConvTranspose2d(256, 256, 4),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.01, inplace=True),
			
			nn.ConvTranspose2d(256, 128, 4, stride=2),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.01, inplace=True),
			
			nn.ConvTranspose2d(128, 64, 4),
			nn.BatchNorm2d(64),	
			nn.LeakyReLU(0.01, inplace=True),
			
			nn.ConvTranspose2d(64, 32, 4, stride=2),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(0.01, inplace=True),
			
			nn.ConvTranspose2d(32, 32, 5),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(0.01, inplace=True),
			
			nn.Conv2d(32, 32, 1),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(0.01, inplace=True),
			
			nn.Conv2d(32, 3, 1),
			nn.Sigmoid()
		)

	def forward(self, input):
		gpu_ids = None
		if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
			gpu_ids = range(self.ngpu)
		output = nn.parallel.data_parallel(self.main, input, gpu_ids)
		return output

# (Yuliang) Not sure the dropout behavior here
# x disc
class _D_x(nn.Module):
	def __init__(self, ngpu):
		super(_D_x, self).__init__()
		self.ngpu = ngpu
		self.main = nn.Sequential(
			# input is 3 x 32 x 32
			nn.Conv2d(3, 32, 5),    # no padding
			nn.LeakyReLU(0.01, inplace=True),

			nn.Conv2d(32, 64, 4, stride=2),
			nn.LeakyReLU(0.01, inplace=True),
			nn.BatchNorm2d(64),

			nn.Conv2d(64, 128, 4),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.01, inplace=True),
			
			nn.Conv2d(128, 256, 4, stride=2),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.01, inplace=True),
			
			nn.Conv2d(256, 512, 4),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.01, inplace=True),
		)

	def forward(self, input):
		gpu_ids = None
		if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
			gpu_ids = range(self.ngpu)
		output = nn.parallel.data_parallel(self.main, input, gpu_ids)
		return output

# z disc
class _D_z(nn.Module):
	def __init__(self, ngpu):
		super(_D_z, self).__init__()
		self.ngpu = ngpu
		self.main = nn.Sequential(
			# input is 256 x 1 x 1
			nn.Conv2d(256, 512, 1),
			nn.LeakyReLU(0.01, inplace=True),

			nn.Conv2d(512, 512, 1),
			nn.LeakyReLU(0.01, inplace=True)
		)

	def forward(self, input):
		gpu_ids = None
		if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
			gpu_ids = range(self.ngpu)
		output = nn.parallel.data_parallel(self.main, input, gpu_ids)
		return output

# joint disc
class _D(nn.Module):
	def __init__(self, ngpu):
		super(_D, self).__init__()
		self.ngpu = ngpu
		self.main = nn.Sequential(
			# input is 1024 x 1 x 1
			nn.Conv2d(1024, 1024, 1),
			nn.LeakyReLU(0.01, inplace=True),

			nn.Conv2d(1024, 1024, 1),
			nn.LeakyReLU(0.01, inplace=True),

			nn.Conv2d(1024, 1, 1),
			nn.Sigmoid()
		)

	def forward(self, input):
		gpu_ids = None
		if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
			gpu_ids = range(self.ngpu)
		output = nn.parallel.data_parallel(self.main, input, gpu_ids)
		return output.view(-1, 1)

if __name__ == '__main__':
	ngpu = 1
	Gz = _Gz(ngpu)
	Gz.apply(weights_init)
	print(Gz)

	Gx = _Gx(ngpu)
	Gx.apply(weights_init)
	print(Gx)

	D_x = _D_x(ngpu)
	D_x.apply(weights_init)
	print(D_x)

	D_z = _D_z(ngpu)
	D_z.apply(weights_init)
	print(D_z)

	D = _D(ngpu)
	D.apply(weights_init)
	print(D)

	criterion = nn.BCELoss()

	input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
	noise = torch.FloatTensor(opt.batchSize, 256, 1, 1)
	fixed_noise = torch.FloatTensor(opt.batchSize, 256, 1, 1).normal_(0, 1)
	label1 = torch.FloatTensor(opt.batchSize)
	label0 = torch.FloatTensor(opt.batchSize)

	real_label = 1
	fake_label = 0

	if opt.cuda:
		Gz.cuda()
		Gx.cuda()
		D_x.cuda()
		D_z.cuda()
		D.cuda()
		input, label1, label0 = input.cuda(), label1.cuda(), label0.cuda()
		noise, fixed_noise = noise.cuda(), fixed_noise()

	input = Variable(input)
	label1 = Variable(label1)
	label0 = Variable(label0)
	noise = Variable(noise)
	fixed_noise = Variable(fixed_noise)

	# set up optimizer
	D_vars = []
	G_vars = []
	for module in [Gz, Gx]:
		for var in module.parameters():
			D_vars.append(var)

	for module in [D_z, D_x, D]:
		for var in module.parameters():
			G_vars.append(var)

	optimizerD = optim.Adam(D_vars, lr=opt.lr, betas=(opt.beta1, 0.999))
	optimizerG = optim.Adam(G_vars, lr=opt.lr, betas=(opt.beta1, 0.999))

	for epoch in range(opt.niter):
		for i, data in enumerate(dataloader, 0):
			# (1) Draw real X samples from dataset
			D.zero_grad()
			real_cpu, _ = data
			batch_size = real_cpu.size(0)
			input.data.resize_(real_cpu.size()).copy_(real_cpu)

			label1.data.resize_(batch_size).fill_(real_label)
			label0.data.resize_(batch_size).fill_(fake_label)

			# (2) Draw real Z samples randomly
			noise.data.resize_(batch_size, 256, 1, 1)
			noise.data.normal_(0, 1)

			# (3) Feed real X into Gz to get fake Z
			fake_z = Gz(input)

			# (4) Feed real Z into Gx to get fake X
			fake_x = Gx(noise)

			# (5) Compute disc prediction
			real_D_x = D_x(input)
			fake_D_z = D_z(fake_z)
			real_fake_D = torch.cat([real_D_x, fake_D_z], 1)
			real_fake_pred = D(real_fake_D)
			Drf = real_fake_pred.data.mean()

			fake_D_x = D_x(fake_x)
			real_D_z = D_z(noise)
			fake_real_D = torch.cat([fake_D_x, real_D_z], 1)
			fake_real_pred = D(fake_real_D)
			Dfr = fake_real_pred.data.mean()

			# (6) Compute disc loss
			# log rho_q
			errD = criterion(real_fake_pred, label1)
			# log (1 - rho_p)
			errD += criterion(fake_real_pred, label0)

			# (7) Compute gen loss
			# log (1 - rho_q)
			errG = criterion(real_fake_pred, label0)
			# log rho_p
			errG += criterion(fake_real_pred, label1)

			# (8) Simultaneously update?
			errD.backward(retain_variables=True)
			optimizerD.step()

			errG.backward()
			optimizerG.step()

			print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x, z^): %.4f D(x^, z) : %.4f'
			% (epoch, opt.niter, i, len(dataloader), errD.data[0], errG.data[0],
				Drf, Dfr))

			ipdb.set_trace()

