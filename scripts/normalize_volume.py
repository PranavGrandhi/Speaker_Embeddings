from pydub import AudioSegment



'''
Given a wavfile, normalise to target volume (in decibels)

Author: Lim Zhi Hao	Date: 2022-03-20

'''


def normalize_volume(audio : AudioSegment, 
		target_dBFS : int = -30, 
		increase_only : bool = True, 
		decrease_only : bool = False,
		error_msg : str = "Both increase only and decrease only are set"):

	# Check consistency of inputs
	if increase_only and decrease_only:
		raise ValueError(error_msg)

	# Determine volume to adjust
	dBFS_change = target_dBFS - audio.dBFS

	# Check for conflicts
	conflict = False
	if dBFS_change < 0 and increase_only:
		# Target volume is softer
		# but user specify to increase volume
		conflict = True

	if dBFS_change > 0 and decrease_only:
		# Target volume is louder 
		# but user specify to decrease volume
		conflict = True

	# If no user input conflicts
	if not conflict:
		# Proceed to normalise loudness of audio file
		return audio.apply_gain(dBFS_change)
	else:
		return audio
