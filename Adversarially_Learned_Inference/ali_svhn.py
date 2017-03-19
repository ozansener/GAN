# A PyTorch replication of ALI_svhn.py
# Credit: 
# https://github.com/pytorch/examples/blob/master/dcgan/main.py
# https://github.com/IshmaelBelghazi/ALI/blob/master/experiments/ali_svhn.py
#
# Author: Yuliang Zou

import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dsets
# import torchvision.transforms as transforms
# import torchvision.utils as vutils
from torch.autograd import Variable

try:
	import ipdb
except:
	import pdb as ipdb


# custom weights init
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.01)
		# m.bias.data.fill_(0.0)    # this should be default setting
	# Not sure how to set init value for bn layers??!

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
		mu, logvar = split(enc, 256, dim=1)
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

# Not sure the dropout behavior here
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
		return output

if __name__ == '__main__':
	ngpu = 1
	Gz = _Gz(ngpu)
	Gz.apply(weights_init)
	print Gz

	Gx = _Gx(ngpu)
	Gx.apply(weights_init)
	print Gx

	D_x = _D_x(ngpu)
	D_x.apply(weights_init)
	print D_x

	D_z = _D_z(ngpu)
	D_z.apply(weights_init)
	print D_z

	D = _D(ngpu)
	D.apply(weights_init)
	print D

