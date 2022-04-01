from pydub import AudioSegment
from pathlib import Path

audiofile = Path("C:\\Users\\prana\\Desktop\\diarization_temp\\speaker_embeddings\\test_audio\\librispeech_test-other\\1688\\1688-142285-0002.flac")
print(audiofile)

audio = AudioSegment.from_file(audiofile)