import numpy as np
import glob,pickle
import soundfile
import librosa
import pyaudio
import wave
from scipy.io import wavfile
from tqdm import tqdm
import pandas as pd
import os



def envelope(y , rate, threshold):
    mask=[]
    y=pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10) ,  min_periods=1 , center = True).mean()
    for mean in y_mean:
        if mean>threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask




def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result=np.hstack((result, mfccs))
        if chroma:  
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
        result=np.hstack((result, mel))
    return result


#cleaning the audio
def cleanaudio(filename): 
    for file in tqdm(glob.glob(filename)):
        file_name = os.path.basename(file)
        signal , rate = librosa.load(file, sr=16000)
        mask = envelope(signal,rate, 0.0005)
        wavfile.write(filename= r'static\\'+str(file_name), rate=rate,data=signal[mask])

        
        
        

# loading the model
def Predict(file):
    cleanaudio(file)
    ans =[]
    new_feature = extract_feature("static\\"+file, mfcc=True, chroma=True, mel=True)
    ans.append(new_feature)
    ans = np.array(ans)

    loaded_model = pickle.load(open("Emotion_Voice_Detection_Model.pkl", 'rb'))
    #predicting :
    y_pred=loaded_model.predict(ans)
    return y_pred[0]
    


