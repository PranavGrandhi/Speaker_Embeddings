import numpy as np
from librosa.feature import melspectrogram


class extract_melspectrogram():

	def __init__(self, 
		sample_rate : int,
		mel_window_length : int = 25,
		mel_window_step : int = 10,
		mel_n_channels : int = 40,
		msecs : int = 1000):

		"""
		Derives a mel spectrogram ready to be used by the encoder from a 
		preprocessed audio waveform.
		
		Note: this not a log-mel spectrogram.

		"""
	
		self.sample_rate = sample_rate
		self.n_fft = int(self.sample_rate * mel_window_length / msecs)
		self.hop_length = int(sample_rate * mel_window_step / msecs)
		self.mel_n_channels = mel_n_channels


	def __call__(self, array : np.array):

		frames = melspectrogram(y = array,
					sr = self.sample_rate,
					n_fft = self.n_fft,
					hop_length = self.hop_length,
					n_mels = self.mel_n_channels)

		return frames.astype(np.float32).T
