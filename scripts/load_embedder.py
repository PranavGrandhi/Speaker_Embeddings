import os, sys
from pathlib import Path

from parse_yaml import parse_yaml
from model_loader import model_loader
from utils import dict2class

'''
Run single audio feature extraction

Author: Lim Zhi Hao			Date: 2022-03-22

'''

class load_embedder():
		
	def __init__(self, yaml_file : Path, batch_dim : int = 0):

		# Initialize parameters
		self.yaml_file = yaml_file
		self.batch_dim = batch_dim
		
		# Load Configuration filev	
		self.config = parse_yaml(self.yaml_file) # this is a dic of dict basically parsing yaml file

		# Load Embedder class from config dictionary
		self.embedder = dict2class(self.config)	

		# Load Pretain model
		self.load_pretrain()


	def load_pretrain(self):
		# Load pretrain model
		model, device = self.embedder.model_loader(self.embedder.model)
		self.embedder.model = model
		self.device = device

	def __call__(self, audio_file : Path):
		features = self.embedder.load_audio.load(audio_file, 
						self.embedder.extract_feature)

		if features is not None:
			embeddings = self.embedder.model(features.to(self.device))		
			return embeddings.mean(self.batch_dim)	
		else:
			return None
		



def main():

	audio_file = sys.argv[1]
	yaml_file = sys.argv[2]

	# Load form configuration file
	embedder = load_embedder(yaml_file)

	# Extract embeddings
	embedding = embedder(audio_file)

	print('Embedding Dimensions: {}'.format(embedding.shape))

	
if __name__ == '__main__':

	main()
