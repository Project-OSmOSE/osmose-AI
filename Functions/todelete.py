# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 13:14:04 2023

@author: gabri
"""

import os
import glob
import pandas as pd
import datetime
from tqdm import tqdm
import numpy as np

LenghtFile_tab = [50, 50]
sample_rate_tab = [250, 250]
dataset_ID_tab = ['Glider', 'Dataset2015_AUS']
AnnotatorsList_tab = [['Julie'], ['expert']]
file_annotation_tab = ['APLOSE_Glider_SPAmsLF_ManualAnnotations_V2_ShortBbAus_results.csv', 'Dataset2015_AUS_results.csv']
orig_LabelsList_tab = [["Bb.Aus"], ['Bw.Ant']]

FinalLabel_Dic = {'Bm.Aus':["Bb.Aus", "Bw.Ant"]}

Crop_duration = 3 #seconds
is_box = None

LabelType = 'classic' #"weak_labels"

i = 0

LenghtFile = LenghtFile_tab[i]
sample_rate = sample_rate_tab[i]
dataset_ID = dataset_ID_tab[i]
file_annotation = file_annotation_tab[i]
LabelsList =  orig_LabelsList_tab[i]
AnnotatorsList = AnnotatorsList_tab[i]


with open('path_osmose.txt') as f:
    path_osmose = f.readlines()[0]

path_osmose_dataset = path_osmose + 'dataset' 
folderName_audioFiles = str(LenghtFile)+'_'+str(int(sample_rate))

#List All WavFile From Dataset
base_path = path_osmose_dataset + os.sep + dataset_ID + os.sep
path_audio_files = os.path.join(path_osmose_dataset, dataset_ID, 'data','audio', folderName_audioFiles )
list_wavfile = [os.path.basename(x) for x in glob.glob(os.path.join(path_audio_files , '*wav'))]


#Load Annotation files
xl_data = pd.read_csv(base_path + 'result' + os.sep  + file_annotation)

#Remove unannotated files
if os.path.exists(base_path + 'result' + os.sep  + file_annotation[:-12]+'_task_status.csv'):
    print("Task Status existing : we're removing all unannotated files ...")
    xl_task_status = pd.read_csv(base_path + 'result' + os.sep  + file_annotation[:-12]+'_task_status.csv')
    xl_status_list_file = list(xl_task_status['filename'])
    list_path_datasets_formats = [x[0] for x in os.walk(os.path.join(path_osmose_dataset, dataset_ID, 'data','audio'))]
    
    for i in range(len(list_path_datasets_formats)):
        if list_path_datasets_formats[i][-5:] != 'audio':
            wav_example = os.path.basename(glob.glob(list_path_datasets_formats[i]+ os.sep + '*.wav')[0])
            if wav_example in xl_status_list_file:
                print(list_path_datasets_formats[i], wav_example, i)
                break
    
    orig_timestamp = pd.read_csv(list_path_datasets_formats[i] + os.sep + 'timestamp.csv', header=None)
    orig_metadata_csv = pd.read_csv(list_path_datasets_formats[i] + os.sep + 'metadata.csv')
    
    orig_duration_file = orig_metadata_csv['audio_file_dataset_duration'][0]
    orig_start_datetime = []
    orig_end_datetime = []
    for j in range(len(orig_timestamp)):
        for annot in AnnotatorsList:
            if orig_timestamp[0][j] in xl_status_list_file:
                if xl_task_status[annot][xl_status_list_file.index(orig_timestamp[0][j])] == 'FINISHED':
                    orig_start_datetime.append(datetime.datetime.strptime(orig_timestamp[1][j]+'+0000', '%Y-%m-%dT%H:%M:%S.%fZ%z').timestamp())
                    orig_end_datetime.append(orig_start_datetime[-1] + orig_duration_file)
                    break
        
        
    audio_timestamp = pd.read_csv(path_audio_files + os.sep + 'timestamp.csv', header=None)
    list_file_ts = list(audio_timestamp[0])
    
    array_to_delete = np.zeros([len(audio_timestamp)])
    for j in range(len(array_to_delete)):
        array_to_delete[j] = datetime.datetime.strptime(audio_timestamp[1][j]+'+0000', '%Y-%m-%dT%H:%M:%S.%fZ%z').timestamp()
    
    
    
    
    list_wavfile_to_be_kept = []
    for file in tqdm(list_wavfile):
        wav_dt_in = datetime.datetime.strptime(audio_timestamp[1][list_file_ts.index(file)]+'+0000', '%Y-%m-%dT%H:%M:%S.%fZ%z').timestamp()
        wav_dt_out = wav_dt_in + LenghtFile
        
        #cond1
        tab_test = np.zeros(len(orig_end_datetime))
        tab_test[np.array(orig_start_datetime) <= wav_dt_in] += 1
        tab_test[np.array(orig_end_datetime) >= wav_dt_in] += 1
        
        if 2 in tab_test: 
            list_wavfile_to_be_kept.append(file)
            continue
        
        #cond2
        tab_test = np.zeros(len(orig_end_datetime))
        tab_test[np.array(orig_start_datetime) <= wav_dt_out] += 1
        tab_test[np.array(orig_end_datetime) >= wav_dt_out] += 1
        
        if 2 in tab_test: 
            list_wavfile_to_be_kept.append(file)
            continue
        
    