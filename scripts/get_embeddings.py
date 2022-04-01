from tqdm import tqdm
from torch import Tensor, vstack
from torch.nn.functional import normalize

from load_embedder import load_embedder 

def get_embeddings(spk2utt : dict, embedder : load_embedder):

	spk2emb = {}
	num_spks = len(spk2utt)

	print(spk2utt.items())

	for i, (spkid, audio_files) in enumerate(spk2utt.items(), 1):
		embeddings = []
		desc = '{}/{}: Extracting embeddings for {}'.format(i, num_spks, spkid)
		for audio_file in tqdm(audio_files, desc = desc):	
			try:
				#print(audio_file)
				embedding = embedder(audio_file)
				print("YAY")
				if isinstance(embedding, Tensor):
					embeddings.append(embedding)

			except Exception as _exception:
				print("Hello")
				print(_exception)

		# Stack embeddings and normalize each row to norm = 1		
		if embeddings:
			embeddings = vstack(embeddings)
			embeddings = normalize(embeddings, p = 2, dim = 1)

			# Detach and store as numpy arrays on cpu
			spk2emb[spkid] = embeddings.detach().to('cpu').numpy()

	return spk2emb


