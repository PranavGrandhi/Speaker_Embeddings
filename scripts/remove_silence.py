from typing import Dict, List, Union
from pydub import AudioSegment

from utils import get_slices

'''

Use webrtcvad to do VAD on AudioSegment object
- Input: AudioSegment
- Output: List of slices List[slice]

TODO: Make script swapable with any custom VAD module

Author: Lim Zhi Hao		Date: 2022-02-20

'''



def remove_silence(audio : AudioSegment, vad_object):

	'''
	Inputs:
		audio: AudioSegment Object	
		vad_object: Current support only for webrtc and silero

	Output:
		AudioSegment object, with silence removed

	'''

	# Get voiced frames
	voiced_frames = vad_object.get_voice_frames(audio)
	
	# Concatenate audio from voiced frames
	audio_vad = AudioSegment.empty()
	for _slice in voiced_frames:
		audio_vad += audio[_slice]
	
	return audio_vad
