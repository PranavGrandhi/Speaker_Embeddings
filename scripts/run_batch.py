import os, sys

from load_embedder import load_embedder 
from get_embeddings import get_embeddings 
from plot_projections import plot_projections
from utils import make_spk2utt

'''
Test batch audio feature extraction and embedding

0) Prepare pytorch data layout beforehand	
	Audio-directories
		|--- Class 1
		|--- Class 2
		|--- Class 3
			|--- audio1.path
			|--- audio2.path
			|--- audio3.path

1) Build speaker-to-utterances dictionary (Output: spk2utt)
	Each dictionary has key-value:
	<speaker ID>: [<audio1.path>, <audio2.path>, ...]

	For more information: http://kaldi-asr.org/doc/data_prep.html
	This format is a "leftover" habit from using kaldi for many years.

2) Initate embedder from yaml file (Output: embedder)
	The yaml configuration will load everything dynamically
	and build the embedder accordingly

3) Get embeddings using function get_embeddings (Output: spk2emb)
	This function will apply the embedder onto the spk2utt
	Bascially, 2 for loops
		- Loop 1: Go through all speakers in spk2utt
		- Loop 2: For each speaker, loop through all audio files
		- Stack all embeddings belonging to speaker

	Returns a speaker-to-embeddings dictionary
	


Author: Lim Zhi Hao			Date: 2022-03-22

'''



def main():

	audio_dir = sys.argv[1]
	ext = sys.argv[2]
	yaml_file = sys.argv[3]	

	fig_savename = 'png/run_batch.png'
	fig_mode = 'umap'
	title = "{} Projections".format(fig_mode.upper())
	complete_msg = 'Embedding plot saved to: {}'.format(fig_savename)

	# Make speaker-to-utterances dictionary
	spk2utt = make_spk2utt(audio_dir, ext)

	# Load embedding class
	embedder = load_embedder(yaml_file)

	# Extract embeddings for each speaker
	# E.g. 10 audio per speaker --> 10 row vectors
	spk2emb = get_embeddings(spk2utt, embedder)	

	print(spk2emb)

	# Plot visualisations
	##projs, ax = plot_projections(spk2emb = spk2emb, title = title, mode = fig_mode)
	##ax.figure.savefig(fig_savename)


if __name__ == '__main__':

	main()
