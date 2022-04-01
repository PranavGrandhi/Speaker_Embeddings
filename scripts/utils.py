from typing import Dict, List, Union
from pathlib import Path

from torch import from_numpy
from sklearn.manifold import TSNE
from umap import UMAP
from pydub import AudioSegment
import numpy as np

from modules_loader import modules_loader


'''
Various utility functions (V1.0)
- dict2class: Load all modules from a yaml file dicitonary
- tsne_projection: Project array into 2 dimensions
- umap_projection: Project array into 2 dimensions
- dict2data: Convert spk2emb to data+labels
- get_slices: Get indices of slices (as a generator)
- to_array: Convert AudioSegment object into numpy array

Author: Lim Zhi Hao		Date: 2022-03-22

'''



def make_spk2utt(pytorch_audiodir : Path, ext : str = 'wav'):

	''' Given a torch data dir (https://pytorch.org/vision/stable/datasets.html)
	    Search for audio files with extension and make spk2utt
	
	    spk2utt: 
		<spk-ID>: [<path-to-audio1>, <path-to-audio2>, ...]

	'''

	if isinstance(pytorch_audiodir, str):
		pytorch_audiodir = Path(pytorch_audiodir)

	audio_paths = pytorch_audiodir.glob('**/*.{}'.format(ext))

	# Code only works if the files are arranged in standard pytorch layout
	spk2utt = {}
	for audiofile in audio_paths:
		if audiofile.is_file():
			# Standard pytorch layout
			spkid = audiofile.parent.stem
			if spkid in spk2utt.keys():
				spk2utt[spkid].append(audiofile)
			else:
				spk2utt[spkid] = [audiofile]

	return spk2utt



class dict2class(object):
 
	''' Dynamically load modules '''
     
	def __init__(self, config : dict):
          
		for key, val_dict in config.items():
			assert isinstance(val_dict, dict)
			setattr(self, key, modules_loader(**val_dict))
  

def tsne_projection(data : np.array, n_components : int = 2):

	''' data: Numpy array (num_samples X feature_dim) '''

	tsne = TSNE(n_components = n_components)
	projections = tsne.fit_transform(data)

	return projections


def umap_projection(data : np.array, n_components : int = 2):

	''' data: Numpy array (num_samples X feature_dim) '''

	reducer = UMAP(n_components = n_components)
	projections = reducer.fit_transform(data)
	
	return projections


def dict2data(spk2emb : Dict[str, np.array], 
		batch_first : bool = True,
		to_numeric : bool = True):

	''' Convert spk2emb dictionary into data, labels '''

	if batch_first:
		axis = 0
	else:
		axis = -1

	spk_labels = []
	embeddings = []
	for i, (spkid, embedding) in enumerate(spk2emb.items(), 1):
		num_samps = embedding.shape[axis]
		if to_numeric:
			labels = np.ones(num_samps) * i
		else:
			labels = np.array([spkid for _ in range(num_samps)])

		spk_labels.append(labels)
		embeddings.append(embedding)

	return np.vstack(embeddings), np.concatenate(spk_labels)




def get_slices(num_samps : int,
		winlen : int, 
		return_last : bool = False,
		match_samps : bool = True):
	'''
	Get slices, given number of samples and window length

	Input:
	- num_samps: Length of audio in milliseconds
	- winlen: Length of window in milliseconds
	- return_last: If true, return the last window
	- match_samps: If true, last index equals num_samps

	The last 2 options are for end point control
	- Sometimes, we discard the last window if winlen is small
	- Sometimes, we need the last few samples

	Outputs:
	- Generator that returns slice objects

	'''

	# Start accumulating slices via indices
	start_idx = range(0, num_samps, winlen)
	for i, start in enumerate(start_idx, 1):
		if start + winlen < num_samps:
			stop = start + winlen
			_slice = slice(*[start, stop])
			yield _slice
		else:
			if return_last:
				if match_samps:
					# The length of last window will be equal to the rest
					start = num_samps - winlen
				
				stop = num_samps
				_slice = slice(*[start, stop])
				yield _slice



def to_array(audio : AudioSegment, to_torch : bool = False):

	'''
	Convert AudioSegment into Numpy arrays

	Input:
	- audio : AudioSegment Object

	'''

	# Get slice of audiosegment object
	samples = audio.get_array_of_samples()
	array = np.array(samples, dtype = np.float32)

	# Reshape with channel information
	outshape = (-1, audio.channels)
	array = array.reshape(outshape).T
	if audio.channels == 1:
		array = array.squeeze(0)	

	# Normalise to (-1,+1) using sample_width (default: 2)
	normalization = 1 << (8 * audio.sample_width - 1)
	array /= normalization

	if to_torch:
		array = from_numpy(array)

	return array
