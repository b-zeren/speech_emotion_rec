import pyaudio
import wave
from sys import byteorder
from array import array
from struct import pack
import speech_recognition as sr
import librosa
import numpy as np
from tensorflow import keras


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt # plots
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import noisereduce as nr

from glob import glob

from models import *

def play(filepath):
		
		filename = filepath
		# Set chunk size of 1024 samples per data frame
		chunk = 1024  
		# Open the sound file 
		wf = wave.open(filename, 'rb')
		# Create an interface to PortAudio
		p = pyaudio.PyAudio()
		# Open a .Stream object to write the WAV file to
		# 'output = True' indicates that the sound will be played rather than recorded
		stream = p.open(format = p.get_format_from_width(wf.getsampwidth()),
		                channels = wf.getnchannels(),
		                rate = wf.getframerate(),
		                output = True)
		# Read data in chunks
		data = wf.readframes(chunk)
		# Play the sound by writing the audio data to the stream
		while data != '':
			stream.write(data)
			data = wf.readframes(chunk)
			if not data:
				stream.stop_stream()
				stream.close()
				p.terminate()
				break



sample_audios=glob("/home/buket/Desktop/codes/kovanstaj/emotion_recognition/Audio samples-20220908T131845Z-001/Audio samples/*.wav")

four_dataset_path='/home/buket/Desktop/codes/kovanstaj/emotion_recognition/4 dataset/buket_res_model.h5'
one_dimensional_path="/home/buket/Desktop/codes/kovanstaj/emotion_recognition/1d_res/buket_best_model_1d/content/best_model_1D"
spectrogram_path="/home/buket/Desktop/codes/kovanstaj/emotion_recognition/spectrogram_model/unmapped_spec_model.pt"

first_model=four_dataset_model(four_dataset_path)
second_model=one_dimensional_model(one_dimensional_path)
third_model=spectrogram_model(spectrogram_path)

i=1
for audio_path in sample_audios:
    
    print("Sample:",i)
    i+=1
    play(audio_path)
    first_result=first_model.predict(audio_path)
    second_result=second_model.predict(audio_path)
    third_result=third_model.predict(audio_path)
    print("RESULT FROM FOUR DATASET MODEL:\t",first_result)
    print("RESULT FROM ONE DIMENTIONAL MODEL:\t",second_result)
    print("RESULT FROM SPECTROGRAM MODEL:\t",third_result)
    print("****************************")