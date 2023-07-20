# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 20:59:15 2022

@author: gabri
"""



from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import os
import scipy.io.wavfile as wav
from scipy.signal import welch, butter, sosfilt
import sys
import librosa


# CLASS DATASET
class ClassDataset(Dataset):
    def __init__(self, path_osmose_dataset, annotation_file, parameters, transform=None):
        self.path_osmose_dataset = path_osmose_dataset
        self.parameters = parameters
        
        self.annotations = pd.read_csv(annotation_file)
        self.transform = transform
        
        self.input_data_format = parameters['input_data_format']
        self.dataset_ID_tab = list(parameters['dataset_ID_tab'])
        
        #parameters spectrograms
        self.Nfft = parameters['nfft']
        self.window_size = parameters['window_size']
        self.overlap = parameters['overlap']
        self.num_classes = parameters['num_classes']
        self.sample_rate = parameters['sample_rate']
        self.scaling = parameters['scaling']
                
        
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        
        if self.input_data_format == 'audio':
            filename = self.annotations['filename'][index] 
            dataset = self.annotations['dataset'][index]
            index_dataset = self.dataset_ID_tab.index(dataset)
            format_file = self.annotations['format'][index]
            file_path = self.path_osmose_dataset + os.sep + dataset + os.sep +  'data' + os.sep + 'audio' + os.sep + format_file + os.sep   + filename 
            input_data, _ = librosa.load(file_path, sr=None)
            
            self.parameters['index_dataset'] = index_dataset
            spectro,_ = self.gen_spectro(input_data)

            
        if self.input_data_format == 'spectrogram':
            filename = self.annotations['filename'][index][:-4] + '_1_0.npz'
            dataset = self.annotations['dataset'][index]
            index_dataset = self.dataset_ID_tab.index(dataset)
            format_file = self.annotations['format'][index]
            folder_spectro = str(int(self.Nfft[index_dataset])) + '_' + str(int(self.window_size[index_dataset])) + '_' + str(int(self.overlap[index_dataset]))
            file_path = self.path_osmose_dataset + os.sep + dataset + os.sep + 'processed' + os.sep + 'spectrogram' + os.sep + format_file + os.sep + folder_spectro + os.sep + 'matrix' + os.sep + filename  
            
            spectro_npz = np.load(file_path)
            spectro = spectro_npz['Sxx']
            self.parameters['index_dataset'] = index_dataset
            
        input_label = torch.tensor(self.annotations.iloc[index, -1*self.num_classes:], dtype=torch.float)
        
        if self.transform is not None:
            spectro = self.transform(spectro, self.parameters)

        return (spectro, input_label)
    
    def __getlabels__(self):
        return self.annotations.columns[-1*self.num_classes:]
    
    def __getfilename__(self, index):
        if self.input_data_format == 'audio':
            filename = self.annotations['filename'][index] 
        if self.input_data_format == 'spectrogram':
            filename = self.annotations['filename'][index][:-4] + '_1_0.npz'
        return filename
    
    def __getdataset__(self, index):
        dataset = self.annotations['dataset'][index] 
        return dataset
    
    
    def gen_spectro(self, data):
        
        window_size = self.window_size[self.parameters['index_dataset']]
        overlap = self.overlap[self.parameters['index_dataset']]
        sample_rate = self.sample_rate[self.parameters['index_dataset']]
        Nfft = self.Nfft[self.parameters['index_dataset']]
        scaling = self.scaling[self.parameters['index_dataset']]
                
        sos = butter(10, 10, 'hp', fs=sample_rate, output='sos')
        data = sosfilt(sos, data)
        
        data = (data - np.mean(data)) / (np.std(data)  + 1e-16)     

        Noverlap = int(window_size * overlap / 100)

        win = np.hamming(window_size)
        if Nfft < (window_size):
            if scaling == 'density':
                scale_psd = 2.0 
            if scaling == 'spectrum':
                scale_psd = 2.0 * sample_rate
        else:
            if scaling == 'density':
                scale_psd = 2.0 / (((win * win).sum())*sample_rate)
            if scaling == 'spectrum':
                scale_psd = 2.0 / ((win * win).sum())
               
        Nbech = np.size(data)
        Noffset = window_size - Noverlap
        Nbwin = int((Nbech - window_size) / Noffset)
        Freq = np.fft.rfftfreq(Nfft, d = 1 / sample_rate)
        Sxx = np.zeros([np.size(Freq), Nbwin])
        for idwin in range(Nbwin):
            if Nfft < (window_size):
                x_win = data[idwin * Noffset:idwin * Noffset + window_size]
                _, Sxx[:, idwin] = welch(x_win, fs=sample_rate, window='hamming', nperseg=Nfft,
                                                noverlap=int(Nfft / 2) , scaling=scaling)
            else:
                x_win = data[idwin * Noffset:idwin * Noffset + window_size] * win
                Sxx[:, idwin] = (np.abs(np.fft.rfft(x_win, n=Nfft)) ** 2)
            Sxx[:, idwin] *= scale_psd

        return Sxx, Freq
    
