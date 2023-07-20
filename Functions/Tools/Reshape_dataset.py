# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 12:50:30 2023

@author: gabri
"""

import os
import glob
import scipy.io.wavfile as wav
import scipy.signal as sps
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from tqdm import tqdm

path_osmose_dataset = 'E:/PhD/OSMOSE_TYPE/dataset/'
dataset_ID  = 'APOCADO_IROISE_C2D1_07072022'
LenghtFile = 10
Fs = 144000
Div_Factor = 5

New_Sr = Fs

base_path = path_osmose_dataset + dataset_ID + os.sep
folderName_audioFiles = str(LenghtFile)+'_'+str(int(Fs))
path_audio_files = os.path.join(path_osmose_dataset, dataset_ID, 'raw/audio', folderName_audioFiles)

NEWfolderName_audioFiles = str(int(LenghtFile/Div_Factor)) + '_' + str(int(Fs))

if not os.path.exists(os.path.join(path_osmose_dataset, dataset_ID, 'raw/audio', NEWfolderName_audioFiles)):
    os.makedirs(os.path.join(path_osmose_dataset, dataset_ID, 'raw/audio', NEWfolderName_audioFiles))

NEWpath_audio_files = os.path.join(path_osmose_dataset, dataset_ID, 'raw/audio', NEWfolderName_audioFiles)

list_wav_withEvent_comp = [os.path.basename(x) for x in glob.glob(os.path.join(path_audio_files , '*wav'))]


timestamp_csv = pd.read_csv(path_audio_files+os.sep+'timestamp.csv', header=None)
Sr, sig = wav.read(path_audio_files + os.sep + list_wav_withEvent_comp[0])

ListWavFile = [] 
ListTimeStamp = []


for id_file in tqdm(range(min(1e20,len(timestamp_csv)))):
    Orig_file_name = timestamp_csv[0][id_file]
    Orig_timestamp = timestamp_csv[1][id_file]
    
    Orig_datetime =  datetime.datetime.strptime(Orig_timestamp, '%Y-%m-%dT%H:%M:%S.%fZ')
    Sr, Orig_sig = wav.read(path_audio_files + os.sep + Orig_file_name)
    
    duration = len(Orig_sig)/Sr
    timedelta = datetime.timedelta(seconds = duration/Div_Factor)   
    
    for i in range(Div_Factor):
        
        sig = Orig_sig[i*int(Sr*duration/Div_Factor):(i+1)*int(Sr*duration/Div_Factor)]
        Datetime = Orig_datetime + timedelta*i
        timestamp = datetime.datetime.strftime(Datetime, '%Y-%m-%dT%H:%M:%S.%fZ')[:-4] + 'Z'
        file_name = Orig_file_name[:-4] + '_p'+ str(i)
        
        if Sr != New_Sr:
            # Resample data
            number_of_samples = round(len(sig) * float(New_Sr) / Sr)
            sig = sps.resample(sig, number_of_samples)
        
        wav.write(NEWpath_audio_files + os.sep + file_name + '.wav', Sr, sig)
        
        ListWavFile.append(file_name+'.wav')
        ListTimeStamp.append(timestamp)
        
        
df1 = pd.DataFrame(ListWavFile)  
df2 = pd.DataFrame(ListTimeStamp )

tiemstamp_output = pd.concat([df1, df2], axis=1) 
tiemstamp_output.to_csv(NEWpath_audio_files + os.sep + 'timestamp.csv', index = False, header=False)
    
    

