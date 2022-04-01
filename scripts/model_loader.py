import os, sys
from pathlib import Path
from torch import nn, cuda, load, device

'''
This module is use to load any model into memory

Author: Lim Zhi Hao		Date: 2019-12-19
Version: st-sensemaking-d2

'''

def get_cuda_num(use_cuda : bool = True, 
		device_str : str = 'cuda:{}',
		num_gpus : int = 1,
		cuda_num : int = 0):

	''' For ST-Sensemaking D2, this function is not used '''

	assert num_gpus >= 0
	default_num = 0
	if use_cuda and cuda.is_available():
		if num_gpus > cuda.device_count():
			device = device_str.format(default_num)
		else:
			if num_gpus > 1:
				device = list(range(num_gpus))
			else:
				device = device_str.format(num_gpus)
	else:
		device = 'cpu'

	#print("Device is " + device)

	return device


class model_loader():
		
	def __init__(self, 
		state_dict_path : Path, 
		use_cuda : bool = True,
		to_eval : bool = False,
		device_str : str = 'cuda:{}',
		cuda_num : int = 0,
		map_location : str = 'cuda',
		default_load_key : str = 'model_state',
		strict : bool = False): 

		self.state_dict_path = state_dict_path
		self.use_cuda = use_cuda
		self.to_eval = to_eval
		self.device_str = device_str
		self.cuda_num = cuda_num
		self.map_location = map_location
		self.default_load_key = default_load_key
		self.strict = strict

		if self.use_cuda:
			self.device = get_cuda_num(cuda_num  = self.cuda_num,
						device_str = self.device_str)
		else:
			self.device = 'cpu'
		


	def __call__(self, model : nn.Module):

		print('Loading model from: {}'.format(self.state_dict_path))
		try:
			checkpoint = load(self.state_dict_path, map_location = self.map_location)
			if self.default_load_key in checkpoint.keys():
				state_dict = checkpoint[self.default_load_key]
			else:
				state_dict = checkpoint
			
			model.load_state_dict(state_dict, strict = self.strict)			
	
			if self.use_cuda:
				model.to(device("cuda"))			
	
			if self.to_eval:
				model.eval()	

			return model, self.device
	
		except Exception as _exception:
			print('Something went wrong')
			print(_exception)
	

