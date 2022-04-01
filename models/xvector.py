import math

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.utils import _single, _pair

'''
This is an all in one TDNN (xvector) script

Author: Lim Zhi Hao			Date: 12-Dec-2019


'''


class Identity(nn.Module):
	
	''' Used for removing layers in pytorch '''

	def __init__(self):
		super(Identity, self).__init__()
		
	def forward(self, x):
		return x
	
		
		
def get_nonlinear(config_str, channels):
	
	''' User configuration for TDNN's nonlinearities'''

	nonlinear = nn.Sequential()
	for i, name in enumerate(config_str.split('-')):
		if name == 'relu':
			nonlinear.add_module('relu', nn.ReLU(inplace=True))
		elif name == 'prelu':
			nonlinear.add_module('prelu', nn.PReLU(channels))
		elif name == 'batchnorm':
			nonlinear.add_module('batchnorm', nn.BatchNorm1d(channels))
		elif name == 'batchnorm_':
			nonlinear.add_module('batchnorm', nn.BatchNorm1d(channels, affine=False))
		else:
			raise ValueError('Unexpected module ({}).'.format(name))
	return nonlinear


def statistics_pooling(x, order=2, dim=-1, keepdim=False, unbiased=True, eps=1e-2):
	
	''' Function to do statistical pooling '''

	stats = []
	mean = x.mean(dim=dim)
	stats.append(mean)
	if order >= 2:
		std = x.std(dim=dim, unbiased=unbiased)
		stats.append(std)
	if order >= 3:
		x = (x - mean.unsqueeze(-1)) / std.clamp(min=eps).unsqueeze(-1)
		skewness = x.pow(3).mean(-1)
		stats.append(skewness)
		if order >= 4:
			kurtosis = x.pow(4).mean(-1)
			stats.append(kurtosis)
	stats = torch.cat(stats, dim=-1)
	if keepdim:
		stats = stats.unsqueeze(dim=dim)
	return stats


class StatsPool(nn.Module):

	''' Statistical pooling layer '''

	def __init__(self, order=2):
		super(StatsPool, self).__init__()
		self.order = order

	def forward(self, x):
		return statistics_pooling(x, order=self.order)


class TimeDelay(nn.Module):

	''' A single TDNN Module (1-D CNN)'''

	def __init__(self, in_channels, out_channels, kernel_size,
				 stride=1, padding=0, dilation=1, bias=True):
		super(TimeDelay, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = _single(kernel_size)
		self.stride = _single(stride)
		self.padding = _pair(padding)
		self.dilation = _single(dilation)
		self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels * kernel_size))
		if bias:
			self.bias = nn.Parameter(torch.Tensor(out_channels))
		else:
			self.register_parameter('bias', None)
		self.reset_parameters()

	def reset_parameters(self):
		with torch.no_grad():
			std = 1 / math.sqrt(self.out_channels)
			self.weight.normal_(0, std)
			if self.bias is not None:
				self.bias.normal_(0, std)

	def forward(self, x):
		x = F.pad(x, self.padding).unsqueeze(1)
		x = F.unfold(x, (self.in_channels,)+self.kernel_size, dilation=(1,)+self.dilation, stride=(1,)+self.stride)
		return F.linear(x.transpose(1, 2), self.weight, self.bias).transpose(1, 2)


class TDNNLayer(nn.Module):

	''' A single TDNN Layer: TDNN+FC+Non-linear'''
	
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
				 dilation=1, bias=True, config_str='batchnorm-relu'):
		super(TDNNLayer, self).__init__()
		if padding < 0:
			assert kernel_size % 2 == 1, 'Expect equal paddings, but got even kernel size ({})'.format(kernel_size)
			padding = (kernel_size - 1) // 2 * dilation
		self.linear = TimeDelay(in_channels, out_channels, kernel_size, stride=stride,
					padding=padding, dilation=dilation, bias=bias)
		self.nonlinear = get_nonlinear(config_str, out_channels)

	def forward(self, x):
		x = self.linear(x)
		x = self.nonlinear(x)
		return x


class DenseLayer(nn.Module):

	''' A single FC layer (w/batchnorm) '''

	def __init__(self, in_channels, out_channels, bias=True,
				 config_str='batchnorm-relu'):
		super(DenseLayer, self).__init__()
		self.linear = nn.Linear(in_channels, out_channels, bias=bias)
		self.nonlinear = get_nonlinear(config_str, out_channels)

	def forward(self, x):
		if len(x.shape) == 2:
			x = self.linear(x)
		else:
			x = self.linear(x.transpose(1, 2)).transpose(1, 2)
		x = self.nonlinear(x)
		return x


class xvector(nn.Module):

	''' 
	Implementation of "X-VECTORS: ROBUST DNN EMBEDDINGS FOR SPEAKER RECOGNITION"
	https://www.danielpovey.com/files/2018_icassp_xvectors.pdf
	
	The parameters are according to the paper.
	
	'''

	def __init__(self,
		feat_dim : int = 30,
		hid_dim : int = 512,
		tdnn_size : int = 1500,
		config_str='batchnorm-relu'):

		super(xvector, self).__init__()
		self.feat_dim = feat_dim
		self.hid_dim = hid_dim
		self.tdnn_size = tdnn_size
		self.pool_dim = 2 * self.tdnn_size

		self.xvector = nn.Sequential(OrderedDict([
			('tdnn1', TDNNLayer(self.feat_dim, self.hid_dim, 5, dilation = 1, padding = -1,
								config_str = config_str)),
			('tdnn2', TDNNLayer(self.hid_dim, self.hid_dim, 3, dilation = 2, padding = -1,
								config_str = config_str)),
			('tdnn3', TDNNLayer(self.hid_dim, self.hid_dim, 3, dilation = 3, padding = -1,
								config_str=config_str)),
			('tdnn4', DenseLayer(self.hid_dim, self.hid_dim, config_str=config_str)),
			('tdnn5', DenseLayer(self.hid_dim, self.tdnn_size, config_str=config_str)),
			('stats', StatsPool()),
			('affine', nn.Linear(self.pool_dim, 512))
		]))

	def forward(self, x):
		return self.xvector(x)



class transfer_TDNN(nn.Module):
	
	'''
	Implementation of TDNN used for transfer learning
	Requires: A trained TDNN (a.k.a extractor)
	
	Can be used to extract
	a) xvectors (1X512)
	b) TDNN embeddings (1X3000)
	
	Xvectors can be thought of as low dimension representations
	of TDNN embeddings (Bottleneck features). 
	
	'''	
	
	def __init__(self, path2tdnn,
			feat_dim = 30, 
			dropout = 0.5,
			tdnn_size = 1500, 
			to_transfer = False, 
			use_xvectors = True,
			do_spec_aug = True, 
			config_str = 'batchnorm-relu'):

		super(transfer_TDNN,self).__init__()
		
		# Initiate dimensionsm, dropout rate and regularization option(s)
		self.feat_dim = feat_dim
		self.tdnn_size = tdnn_size
		self.dropout = dropout
		self.config_str = config_str

		# Initiate training choices
		self.do_spec_aug = do_spec_aug
		self.use_xvectors = use_xvectors
		self.to_transfer = to_transfer
	
		# Initiate path to pretrain TDNN
		self.path2tdnn = path2tdnn

		# Initiate TDNN model and load pretrain model
		self.tdnn = TDNN(feat_dim = self.feat_dim, 
					tdnn_size = self.tdnn_size, 
					config_str = self.config_str)

		print('Loading pretrain model from {}'.format(self.path2tdnn))
		self.tdnn.load_state_dict(torch.load(self.path2tdnn)['state_dict'])
		print('Done')

		# If it is to do transfer learning, freeze TDNN layers
		if self.to_transfer:
			print('Done... Freezing TDNN layers')	
			for param in self.tdnn.parameters():
				param.requires_grad = False
			print('Done')

		# Add spectrum augmentation
		if self.do_spec_aug:
			self.mask = spec_aug()

		# Remove affine layer, this is for retraining embedding layers
		if not self.use_xvectors:
			self.tdnn.xvector.affine = Identity()

		
	def forward(self,x):
		if self.training and self.do_spec_aug:
			x = self.mask(x)
		
		return self.tdnn(x)
			
			
			
