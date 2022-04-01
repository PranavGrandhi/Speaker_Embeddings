from typing import Dict, List, Union

from utils import get_slices
from pydub import AudioSegment
from webrtcvad import Vad


'''

Use webrtcvad to do VAD on AudioSegment object
- Input: AudioSegment
- Output: List of slices List[slice]

TODO: Make script swapable with any custom VAD module

Author: Lim Zhi Hao		Date: 2022-02-20

'''



class vad_webrtc():

	def __init__(self,
		winlen : float = 0.03, 
		agressive_level : int = 3, 
		msecs : int = 1000):

		'''
		inputs:
			winlen: window length of vad (in milliseconds)
			agressive_level: level of agressiveness of vad
			msecs: conversion to or from milliseconds
	
		output:
			generator of slices of speech frames
	
		'''

		# initialise parameters and model
		self.winlen = winlen
		self.agressive_level = agressive_level
		self.vad = Vad(mode = self.agressive_level)
		self.msecs = msecs

	def get_windows(self, audio : AudioSegment):

		# get sample rate of audio and duration in milliseconds
		sample_rate = audio.frame_rate
		duration = int(self.msecs * audio.duration_seconds)

		# get slices based on window length. returns a generator.
		winlen = int(self.msecs * self.winlen)
		slices = get_slices(num_samps = duration, winlen = winlen)

		return slices


	def is_speech_frame(self, window : AudioSegment, sample_rate : int = 16000):

		window = window.get_array_of_samples().tobytes()		
		return self.vad.is_speech(window, sample_rate)


	def process_audio(self, audio : AudioSegment):

		sample_rate = audio.frame_rate
		self._slice = None
		for _slice in self.get_windows(audio):
			# for each slice/window of audio
			self._slice = _slice	
			window = audio[self._slice]
			yield self.is_speech_frame(window, sample_rate)
			

	def get_voice_frames(self, audio : AudioSegment):

		for is_speech in self.process_audio(audio):
			if is_speech:
				yield self._slice



def main():

	import time
	start_time = time.time()

	###############################################################

	audiofile = 'test_audio/test.wav'
	out_wavfile = 'test_audio/test_vad-webrtc.wav'
	audio = AudioSegment.from_file(audiofile)	
	vad = vad_webrtc()	

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

