import numpy as np
from librosa.feature import mfcc


class extract_mfcc():

	def __init__(self, 
		sample_rate : int,
		n_mfcc : int = 30,
		mel_window_length : int = 25,
		mel_window_step : int = 10,
		msecs : int = 1000):

		"""
		Derives a mel spectrogram ready to be used by the encoder from a 
		preprocessed audio waveform.
		
		Note: this not a log-mel spectrogram.

		"""
	
		self.sample_rate = sample_rate
		self.n_mfcc = n_mfcc 
		self.n_fft = int(self.sample_rate * mel_window_length / msecs)
		self.hop_length = int(self.sample_rate * mel_window_step / msecs)



	def __call__(self, array : np.array):

		frames = mfcc(y = array,
				sr = self.sample_rate,
				n_fft = self.n_fft,
				hop_length = self.hop_length,
				n_mfcc = self.n_mfcc)

		return frames.astype(np.float32).T
