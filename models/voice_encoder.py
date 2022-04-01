from torch import nn, cuda, device, load, FloatTensor, norm
from pathlib import Path
from typing import Union

'''
Voice encoder from: https://github.com/resemble-ai/Resemblyzer

Used by rushi in: https://github.com/CorentinJ/Real-Time-Voice-Cloning

Author: Lim Zhi Hao		Date: 2022-03-22

'''


class voice_encoder(nn.Module):
	def __init__(self,
		mel_n_channels : int = 40,
		model_num_layers : int = 3,
		model_hidden_size : int = 256,
		model_embedding_size : int = 256):

		"""
		If None, defaults to cuda if it is available otherwise the model will
		run on cpu. Outputs are always returned on the cpu, as numpy arrays.
		:param weights_fpath: path to "<CUSTOM_MODEL>.pt" file path.
		If None, defaults to built-in "pretrained.pt" model
		"""

		super().__init__()

		# Define the network
		self.lstm = nn.LSTM(mel_n_channels, 
				model_hidden_size, 
				model_num_layers, 
				batch_first = True)

		self.linear = nn.Linear(model_hidden_size, model_embedding_size)
		self.relu = nn.ReLU()


	def forward(self, mels: FloatTensor):
		"""
		Computes the embeddings of a batch of utterance spectrograms.

		:param mels: a batch of mel spectrograms of same duration as a float32 tensor of shape
		(batch_size, n_frames, n_channels)
		:return: the embeddings as a float 32 tensor of shape (batch_size, embedding_size).
		Embeddings are positive and L2-normed, thus they lay in the range [0, 1].
		"""
		# Pass the input through the LSTM layers and retrieve the final hidden state of the last
		# layer. Apply a cutoff to 0 for negative values and L2 normalize the embeddings.
		_, (hidden, _) = self.lstm(mels)
		embeds_raw = self.relu(self.linear(hidden[-1]))
		return embeds_raw / norm(embeds_raw, dim = 1, keepdim = True)
