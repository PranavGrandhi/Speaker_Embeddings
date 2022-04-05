import os, sys
from pathlib import Path

from numpy import stack
from torch import from_numpy
from pydub import AudioSegment

from vad_silero import vad_silero
from vad_webrtc import vad_webrtc 
from normalize_volume import normalize_volume
from remove_silence import remove_silence 
from utils import get_slices, to_array 


'''
Audio class based on AudioSegment from pydub
- Load audio file lazily
- Preprocessing (JIT):
	- Normalize volume
	- Perform VAD
- Load slice by slice, do feature extraction
	- To torch tensor or numpy array

Note. There are 2 analysis windows
- Vad window: 0.03s or 30ms. Used for Voice Activity Detection
- Wav window: 1.6s or 1600ms. Sliding window for feature extraction

Author: Lim Zhi Hao		Date: 2022-03-22

'''


class audio_loader():

	def __init__(self,
		resample : int = None,
		normalize_vol : bool = True,
		target_dBFS : int = -30,		
		increase_only : bool = True,
		wav_winlen : float = 1.6,
		return_last : bool = True,
		match_samps : bool = True,
		trim_silence : bool = True,
		vad_mode : str = 'webrtc',
		msecs : int = 1000,
		to_torch : bool = True):

		# Initialize parameters
		self.normalize_vol = normalize_vol
		self.trim_silence = trim_silence
		self.to_torch = to_torch
		self.resample = resample

		# Initialize for volume normalization
		self.target_dBFS = target_dBFS 
		self.increase_only = increase_only 
	
		# Initialize for voice activity detection
		# TODO: I need a VAD selector module
		self.vad_mode = vad_mode
		if  self.vad_mode == 'webrtc':
			self.vad = vad_webrtc()
		elif self.vad_mode == 'silero':
			self.vad = vad_silero()
	
		# Initialize for getting indices
		self.wav_winlen = wav_winlen
		self.return_last = return_last	
		self.match_samps = match_samps
		self.msecs = msecs


	def get_indices(self, audio: AudioSegment):

		''' Get indices of slices of audio file'''

		# Convert seconds to milliseconds
		num_samps = int(audio.duration_seconds * self.msecs)
		wav_winlen = int(self.wav_winlen * self.msecs)

		# Return slices (indices) of audio
		slices = get_slices(num_samps = num_samps,
				winlen = wav_winlen, 
				return_last = self.return_last,
				match_samps = self.match_samps)	

		return slices


	def preprocess(self, audio : AudioSegment): 
	
		'''
		Performs preprocessing
		- normalize_vol: Normalize volume
		- trim_silence: Perform VAD (Voice Activity Detection)

		'''

		# Normalize audio's volume to target volume
		if self.normalize_vol:
			audio = normalize_volume(audio = audio, 
						target_dBFS = self.target_dBFS,
						increase_only = self.increase_only)
		
		# Remove silence in audio
		if self.trim_silence:
			audio = remove_silence(audio = audio, vad_object = self.vad)
	
		# Return pydub object
		return audio 
		
	
	def load(self, audiofile : Path, feature_extractor = None, to_preprocess : bool = True):
	
		''' 
		- Loads audio file (lazily)
		- Perform pre-processing (If needed, recommended)
		- Get windows/slices of audio file
		- Perform feature extraction (If needed)
		- Returns torch.Tensor or np.array
	
		'''
		
		# Load audio file based on sample rate
		audio = AudioSegment.from_file(audiofile)	
		if self.resample and self.resample != audio.frame_rate:
			audio = audio.set_frame_rate(self.resample)

		# Preprocess if needed
		if to_preprocess:
			audio = self.preprocess(audio)
		
		# Begin extracting window/slices 	
		features = []
		for _slice in self.get_indices(audio):
			array = to_array(audio[_slice])
			# Extract feature if needed
			if feature_extractor:
				feature = feature_extractor(array)		
				features.append(feature)	
			else:
				features.append(array)		

		# Stack them into batch (ndim == 3)
		if features:	
			features = stack(features)
		
			# Convert to torch if needed
			if self.to_torch:
				features = from_numpy(features)
		else:
			features = None	

		return features
		

