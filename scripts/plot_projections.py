import numpy as np
from typing import Union, List, Dict

import matplotlib.pyplot as plt
from utils import tsne_projection, umap_projection, dict2data 


'''
Project embedding into 2-dimensions
Use matplotlib to polot visualizations

Author: Lim Zhi Hao		Date: 2022-03-22
Verison: 1.0		

'''



def plot_projections(spk2emb : dict,
		to_numeric : bool = False, 
		markers : str = None, 
		legend : bool = False, 
		title : str = "", 
		mode : str = 'tsne',
		n_components : int = 2):


	_, ax = plt.subplots(figsize=(6, 6))

	num_spks = len(spk2emb)
	colors = np.random.randint(0,255,[num_spks, 3]).astype(np.float32) / 255
	
	data, labels = dict2data(spk2emb, to_numeric = to_numeric)
	
	# Compute the 2D projections.
	# You could also project to another number of dimensions (e.g. 
	# for a 3D plot) or use a different different dimensionality reduction 
	# like PCA or TSNE.

	if mode == 'tsne':
		projs = tsne_projection(data, n_components = n_components)
	elif mode == 'umap':
		projs = umap_projection(data, n_components = n_components)

	# Draw the projections
	speakers = np.array(labels)
	for i, speaker in enumerate(np.unique(speakers)):
		speaker_projs = projs[speakers == speaker]
		marker = "o" if markers is None else markers[i]
		label = speaker if legend else None
		ax.scatter(*speaker_projs.T, c=[colors[i]], marker = marker, label = label)

	if legend:
		ax.legend(title = "Speakers", ncol = 2)

	ax.set_title(title)
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_aspect("equal")
   
	return projs, ax
 
