
import pyaudio
import wave
from sys import byteorder
from array import array
from struct import pack
import speech_recognition as sr
import librosa
import numpy as np
from tensorflow import keras

from models import *

four_dataset_path='/home/buket/Desktop/codes/kovanstaj/emotion_recognition/4 dataset/buket_res_model.h5'
one_dimensional_path="/home/buket/Desktop/codes/kovanstaj/emotion_recognition/1d_res/buket_best_model_1d/content/best_model_1D"
spectrogram_path="/home/buket/Desktop/codes/kovanstaj/emotion_recognition/spectrogram_model/unmapped_spec_model.pt"

first_model=four_dataset_model(four_dataset_path)
second_model=one_dimensional_model(one_dimensional_path)
third_model=spectrogram_model(spectrogram_path)
    
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




CONF_THRESHOLD=0.45


if __name__ == "__main__":

    print("Model is loaded")
    
    mic=sr.Microphone()
    r=sr.Recognizer()
    
    with mic as source:

        #calibrate mic based on ambient noise
        r.adjust_for_ambient_noise(source)
        print("Mic is ready")
        
        while True:
            #record from audio and put it in cache folder
            print("listening")
            audio = r.listen(source)
            with open("/home/buket/Desktop/codes/kovanstaj/emotion_recognition/microphone_results.wav", "wb") as f:
                f.write(audio.get_wav_data())
            play("/home/buket/Desktop/codes/kovanstaj/emotion_recognition/microphone_results.wav")
            
            '''
            print("Result from four dataset model:",first_model.predict("/home/buket/Desktop/codes/kovanstaj/4 dataset/cache/microphone-results.wav"))
            print("Result from one dimensional model:",second_model.predict("/home/buket/Desktop/codes/kovanstaj/4 dataset/cache/microphone-results.wav"))
            print("Result from spectrogram model:",third_model.predict("/home/buket/Desktop/codes/kovanstaj/4 dataset/cache/microphone-results.wav"))
            '''
    