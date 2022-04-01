#!/bin/bash


audio_dir=./test_audio/librispeech_test-other
ext=flac
yaml_file=./yaml/resemblyzer.yml

python scripts/run_batch.py ${audio_dir} ${ext} ${yaml_file}


