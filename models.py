
import speech_recognition as sr
import librosa
import numpy as np
from tensorflow import keras


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt # plots
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import noisereduce as nr


import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backend_bases import RendererBase
from scipy import signal
from scipy.io import wavfile
import os
from PIL import Image
from scipy.fftpack import fft
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class four_dataset_model:
    def __init__(self,model_path):
        self.model=keras.models.load_model(model_path)
        self.emotions=['disgusted','happy','sad','neutral','fear','angry','surprise'] 

    def zcr(self,data,frame_length,hop_length):
        zcr=librosa.feature.zero_crossing_rate(data,frame_length=frame_length,hop_length=hop_length)
        return np.squeeze(zcr)
    def rmse(self,data,frame_length=2048,hop_length=512):
        rmse=librosa.feature.rms(y=data,frame_length=frame_length,hop_length=hop_length)
        return np.squeeze(rmse)
    def mfcc(self,data,sr,frame_length=2048,hop_length=512,flatten:bool=True):
        mfcc=librosa.feature.mfcc(y=data,sr=sr)
        return np.squeeze(mfcc.T)if not flatten else np.ravel(mfcc.T)

    def extract_features(self,data,sr,frame_length=2048,hop_length=512):
        result=np.array([])
        
        result=np.hstack((result,
                        self.zcr(data,frame_length,hop_length),
                        self.rmse(data,frame_length,hop_length),
                        self.mfcc(data,sr,frame_length,hop_length)
                        ))
        return result

    def get_features(self,path,duration=2.5, offset=0.6):
        data,sr=librosa.load(path)
        data=nr.reduce_noise(y=data, sr = sr)
        aud=self.extract_features(data,sr)
        audio=np.array(aud)
        return audio

    def preprocess(self,audio_path):
        X_predict=[]
        features=self.get_features(audio_path)
        for i in features:
            X_predict.append(i)
        X_resized=X_predict[:2376] + [0,]*(2376-len(X_predict))
        X_predict=np.array(X_resized)
        X_predict=X_predict.reshape((-1,2376,1))
        return X_predict

    def predict(self,audio_path):
     #first prediction from 4 dataset model
        X_predict=self.preprocess(audio_path)
        y_pred = self.model.predict(X_predict)
        y_pred_ind = np.argmax(y_pred, axis=1)
        pred_conf=y_pred[0][y_pred_ind][0]

        result_dict={"emotion":self.emotions[y_pred_ind[0]],"conf":pred_conf}
        return result_dict

class one_dimensional_model:
    def __init__(self,model_path):
        self.model=keras.models.load_model(model_path)
        self.emotions={0:"sad",1:"angry",2:"disgusted",3:"fear",4:"happy",5:"neutral"}
        self.sampling_rate=18000

    def adjust_length(self,time_series_list, length):
        n = len(time_series_list)
        for i in range(n):
            audio_length = len(time_series_list[i])
            if audio_length < length:
                time_series_list[i] = np.append(time_series_list[i], [0 for i in range(length-audio_length)])
            else:
                time_series_list[i] = np.array(time_series_list[i][:length])

    def check_for_nan(self,l):
        for x in l:
            if str(x) == 'nan':
                return True
        return False


    def feature_extraction_1D(self,data):

        # Zero Crossing rate
        features = librosa.feature.zero_crossing_rate(y=data)

        # Energy
        features = np.append(features, librosa.feature.rms(y=data), axis=1)

        # Mel-frequency cepstral coefficient
        l = np.mean(librosa.feature.mfcc(y=data, sr=self.sampling_rate, n_mfcc=13), axis=0).reshape(1, 106)
        features = np.append(features, l, axis=1)
        
        # Spectral Centroid
        features = np.append(features, librosa.feature.spectral_centroid(y=data, sr=self.sampling_rate), axis=1)
        
        # Spectral Bandwidth
        features = np.append(features, librosa.feature.spectral_bandwidth(y=data, sr=self.sampling_rate), axis=1)
        
        # Spectral Flatness
        features = np.append(features, librosa.feature.spectral_flatness(y=data), axis=1)
        
        # Spectral Rolloff maximum frequencies
        features = np.append(features, librosa.feature.spectral_rolloff(y=data, sr=self.sampling_rate), axis=1)
        
        # Spectral Rolloff minimum frequencies
        features = np.append(features, librosa.feature.spectral_rolloff(y=data, sr=self.sampling_rate, roll_percent=0.01), axis=1)
        
        return np.array(features)


    def predict(self,audio_path):
        data = [] 
        length_sum = 0
        signal, sr = librosa.load(audio_path, sr = self.sampling_rate)
        reduced_noise = nr.reduce_noise(y=signal, sr = self.sampling_rate)
        if not self.check_for_nan(reduced_noise):
            signal = reduced_noise
        data.append(signal)
        length_sum += len(signal)
        self.adjust_length(data, 3*self.sampling_rate)
        data = np.array(data)

        data_features_extracted_1D = []
        data_features_extracted_1D.append(np.squeeze(np.append(self.feature_extraction_1D(data[0]), 0)))
        data_features_extracted_1D = np.array(data_features_extracted_1D)

        x_test=data_features_extracted_1D
        predicted_classes=self.model.predict(x_test)
        predicted_class_ind = np.argmax(np.round(predicted_classes),axis=1)[0]
        pred_conf=predicted_classes[0][predicted_class_ind]
        result_dict={"emotion":self.emotions[predicted_class_ind],"conf":pred_conf}

        return result_dict


class ModifiedAlexNet(nn.Module):
    def __init__(self, num_classes=9):
        super(ModifiedAlexNet, self).__init__()
        self.num_classes=num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.features(x)
        #print('features',x.shape)
        x=torch.flatten(x, start_dim=2)#a1,a2,a3......al{a of dim c} 
        x=torch.sum(x, dim=2)#a1*alpha1+a2*alpha2+.......+al*alphal
        #print(x.shape)
        x=self.classifier(x)
        #print('classifier',x)
        #x=self.softmax(x)
        #print('softmax',x)
        #x = self.avgpool(x)
        #print('avgpool',x.shape)
        #x = torch.flatten(x, 1)
        #print('flatten',x.shape)
        #x = self.classifier(x)
        return x
   
class spectrogram_model:
    
    def __init__(self,model_path):
        self.model=ModifiedAlexNet()
        self.model=torch.load(model_path)
        self.model.eval()
        self.model.to('cpu')
        self.emotions=["neutral","frustrated","sad","surprise","angry","happy","excited","fear","disgusted"]

    def log_specgram(self,audio, sample_rate, window_size=40,
                 step_size=20, eps=1e-10):
        nperseg = int(round(window_size * sample_rate / 1e3))
        noverlap = int(round(step_size * sample_rate / 1e3))
        #print('noverlap',noverlap)
        #print('nperseg',nperseg)
        freqs, _, spec = signal.spectrogram(audio,
                                        fs=sample_rate,
                                        window='hann',
                                        nperseg=nperseg,
                                        noverlap=noverlap,
                                        detrend=False)
        return freqs, np.log(spec.T.astype(np.float32) + eps)

    def audio2spectrogram(self,filepath):
        #fig = plt.figure(figsize=(5,5))
        try:
            test_sound,samplerate = librosa.load(filepath)
            #print('samplerate',samplerate)
            _, spectrogram = self.log_specgram(test_sound, samplerate)
            return spectrogram
            #print(spectrogram.shape)
        except:
            print("Problem while reading audio file")
        #print(type(spectrogram))
        #plt.imshow(spectrogram.T, aspect='auto', origin='lower')
        
    
    def audio2wave(self,filepath):
        fig = plt.figure(figsize=(5,5))
        samplerate, test_sound  = wavfile.read(filepath,mmap=True)
        plt.plot(test_sound)

    def get_3d_spec(self, Sxx_in, moments=None):
        if moments is not None:
            (base_mean, base_std, delta_mean, delta_std,
                delta2_mean, delta2_std) = moments
        else:
            base_mean, delta_mean, delta2_mean = (0, 0, 0)
            base_std, delta_std, delta2_std = (1, 1, 1)
        h, w = Sxx_in.shape
        right1 = np.concatenate([Sxx_in[:, 0].reshape((h, -1)), Sxx_in], axis=1)[:, :-1]
        delta = (Sxx_in - right1)[:, 1:]
        delta_pad = delta[:, 0].reshape((h, -1))
        delta = np.concatenate([delta_pad, delta], axis=1)
        right2 = np.concatenate([delta[:, 0].reshape((h, -1)), delta], axis=1)[:, :-1]
        delta2 = (delta - right2)[:, 1:]
        delta2_pad = delta2[:, 0].reshape((h, -1))
        delta2 = np.concatenate([delta2_pad, delta2], axis=1)
        base = (Sxx_in - base_mean) / base_std
        delta = (delta - delta_mean) / delta_std
        delta2 = (delta2 - delta2_mean) / delta2_std
        stacked = [arr.reshape((h, w, 1)) for arr in (base, delta, delta2)]
        return np.concatenate(stacked, axis=2)

    def predict(self,fileName):
        try:
            spector=self.audio2spectrogram(fileName)
            spector=self.get_3d_spec(spector)
            npimg = np.transpose(spector,(2,0,1))
            input_tensor=torch.tensor(npimg)
            input = input_tensor.unsqueeze(0) 
            with torch.no_grad():
                if(input.shape[2]>65):
                    #input = sprectrome.to('cuda')
                    #label1=label1.to('cuda')
                    output = self.model(input)
                    probs=torch.nn.functional.softmax(output,dim=1)
                    _, preds = torch.max(output, 1)
                    label=self.emotions[preds.numpy()[0]]
                    conf=probs.numpy()[0][preds.numpy()[0]]
                    return {"emotion":label,"conf":conf}
                else:
                    print("Third model: no result")
                    return {"emotion":"None","conf":0.0}
        except:
            print("An error occured")
            return {"emotion":"None","conf":0.0}
        