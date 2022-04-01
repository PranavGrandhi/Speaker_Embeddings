from typing import Dict, List, Union

from torch import hub
from utils import get_slices, to_array
from pydub import AudioSegment



class vad_silero():


	def __init__(self,
		winlen : float = 0.0976875, 
		msecs : int = 1000,
		repo_or_dir : str = 'snakers4/silero-vad',
		model_name : str = 'silero_vad',
		force_reload : bool = False,
		return_last : bool = True,
		match_samps : bool = True,
		start_key : str = 'start', 
		stop_key : str ='end',
		to_torch : bool = True,
		return_seconds : bool = True,
		verbose : bool = True):

		'''
		Inputs:
			winlen: Window length of VAD (in seconds, 1536 samples for 16kHz)
			msecs: Conversion to or from milliseconds
	
		Output:
			Generator of slices of speech frames
	
		'''

		# Initialise parameters and model
		self.winlen = winlen
		self.repo_or_dir = repo_or_dir
		self.model_name = model_name
		self.force_reload = force_reload
		self.msecs = msecs
		self.return_last = return_last
		self.match_samps = match_samps
		self.start_key = start_key
		self.stop_key = stop_key
		self.to_torch = to_torch
		self.return_seconds = return_seconds
		self.verbose = verbose	

		self.load_model()


	def load_model(self):

		model, utils = hub.load(repo_or_dir = self.repo_or_dir,
					model = self.model_name,
					force_reload = self.force_reload,
					verbose = self.verbose)
	
		(get_speech_timestamps,
		 save_audio,
		 read_audio,
		 VADIterator,
		 collect_chunks) = utils

		self.vad = VADIterator(model)



	def get_windows(self, audio : AudioSegment):

		num_samps = int(audio.duration_seconds * self.msecs)
		winlen = int(self.winlen * self.msecs)

		# Return slices (indices) of audio
		slices = get_slices(num_samps = num_samps,
					winlen = winlen, 
					return_last = self.return_last,
					match_samps = self.match_samps)

		return slices


	def is_speech_frame(self, window : AudioSegment):
		
		array = to_array(window, to_torch = self.to_torch)
		speech_dict = self.vad(array, return_seconds = self.return_seconds)

		if speech_dict:
			if self.start_key in speech_dict.keys():
				self.start = speech_dict[self.start_key]
				self.start *= self.msecs
				return None
			else:
				self.stop = speech_dict[self.stop_key]
				self.stop *= self.msecs
				return slice(*[int(self.start), int(self.stop)])
	

	def get_voice_frames(self, audio : AudioSegment):

		self.start, self.stop = 0, 0
		for _slice in self.get_windows(audio):
			# For each slice/window of audio
			window = audio[_slice]
			_slice =  self.is_speech_frame(window)
			if _slice:
				yield _slice

		self.vad.reset_states()

		

def main():

	import time
	start_time = time.time()

	###############################################################

	audiofile = 'test_audio/test.wav'
	out_wavfile = 'test_audio/test_vad-silero.wav'
	audio = AudioSegment.from_file(audiofile)	
	vad = vad_silero()	

	print('Duration BEFORE vad: {}s'.format(audio.duration_seconds))
	
	combined = AudioSegment.empty()
	for _slice in vad.get_voice_frames(audio):
		combined += audio[_slice]

	###############################################################

	run_time = time.time() - start_time	
	print('Duration AFTER vad: {}s'.format(combined.duration_seconds))
	print('Took {:.2f}s to do vad'.format(run_time))	
	
	combined.export(out_wavfile, format = 'wav')
	print('Trimmed audio saved to: {}'.format(out_wavfile))


if __name__ == '__main__':

	main()

