# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 15:45:43 2022

@author: gabri
"""

import os
import pandas as pd
import numpy as np

def list_datasets(path_osmose_dataset, nargout=0):
    
    l_ds = sorted(os.listdir(path_osmose_dataset))
    
    if nargout == 0:
        print("Available datasets:")

        for ds in l_ds:
            print("  - {}".format(ds))

    else:
        return l_ds
        
def list_files(startpath, level_max = 2):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        print(level)
        if level <= level_max:
            indent = ' ' * 4 * (level)
            print('{}{}/'.format(indent, os.path.basename(root)))
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                print('{}{}'.format(subindent, f))

def check_available_ai_tasks_bm(path_osmose_AI):
    for root, dirs, files in os.walk(path_osmose_AI):
        level = root.replace(path_osmose_AI, '').count(os.sep)
        indent = ' ' * 4 * (level)
        if level <= 2 and level > 0:
            print('{}{}/'.format(indent, os.path.basename(root)))
        
        
def check_available_annotation(path_osmose_dataset, dataset_ID):
    base_path = path_osmose_dataset + os.sep + dataset_ID + os.sep + 'final' + os.sep + 'Annotation_Aplose' + os.sep
    print('Dataset : ',dataset_ID)
    print('Available Annotation files :')
    print('  ')
    for root, dirs, files in os.walk(base_path):
        level = root.replace(base_path, '').count(os.sep)
        subindent = ' ' * 4 * (level+1)
        for f in files:
            if f[-12:] == '_results.csv':
                print('{}{}'.format(subindent, f))

def check_available_file_resolution(path_osmose_dataset, dataset_ID):
    base_path = path_osmose_dataset + os.sep + dataset_ID + os.sep + 'data' + os.sep + 'audio' + os.sep    
    print('Dataset : ',dataset_ID)
    print('Available Resolution (LengthFile_samplerate) :')
    for root, dirs, files in os.walk(base_path):
        level = root.replace(base_path, '').count(os.sep)
        indent = ' ' * 4 * (level+1)
        print('{}{}'.format(indent, os.path.basename(root)))
    
def check_available_labels_annotators(path_osmose_dataset, dataset_ID, file_annotation):
    base_path = path_osmose_dataset + os.sep + dataset_ID + os.sep 
    xl_data = pd.read_csv(base_path + 'final' + os.sep + 'Annotation_Aplose' + os.sep + file_annotation)
    FullLabelsList = list(dict.fromkeys(xl_data['annotation']))
    FullAnnotatorsList = list(dict.fromkeys(xl_data['annotator']))
    print('Labels Annotated : ',FullLabelsList)
    print('Annotators : ',FullAnnotatorsList)
    
    
def check_available_ai_datasplit(path_osmose_AI, Task_ID, BM_Name):
    print('Datasplits available in this task and this benchmark : ')  
    base_path = path_osmose_AI + os.sep + Task_ID + os.sep + BM_Name + os.sep + 'info_datasplit'
    for root, dirs, files in os.walk(base_path):
        level = root.replace(base_path, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))    

def check_available_ai_model(path_osmose_AI, Task_ID, BM_Name):
    print('Models available in this task and this benchmark : ')      
    base_path = path_osmose_AI + os.sep + Task_ID + os.sep + BM_Name + os.sep + 'models'
    for root, dirs, files in os.walk(base_path):
        level = root.replace(base_path, '').count(os.sep)
        indent = ' ' * 4 * (level)
        if level<2 and level>0:
            print('{}{}/'.format(indent, os.path.basename(root)))       
    
def check_available_formats(path_osmose_dataset, path_osmose_AI, Task_ID, BM_Name):
    base_path = path_osmose_AI + os.sep + Task_ID + os.sep + BM_Name + os.sep + 'info_datasplit' + os.sep
    metadata = np.load(base_path + 'Fdataset_metadata.npz')
    dataset_ID_tab = metadata['dataset_ID_tab']
    for dataset_ID in dataset_ID_tab:
        base_path_dataset = path_osmose_dataset + os.sep + dataset_ID + os.sep + 'processed' + os.sep + 'spectrogram' 
        print('_______________')
        print('Dataset : ',dataset_ID)
        if not os.path.exists(base_path_dataset): print('--- No pre-computed spectrograms ---')
        
        else:
            print('Available Spectrogram Format (nfft_windowsize_overlap) :')

            for folder in os.listdir(base_path_dataset):
                level = 1
                indent = ' ' * 4 * (level)
                print(indent, folder) 
                if os.path.exists(base_path_dataset + os.sep + folder):
                    if folder != 'adjust_metadata.csv':
                        for subfolder in os.listdir(base_path_dataset + os.sep + folder):
                            level = 2
                            indent = ' ' * 4 * (level)
                            print(indent, subfolder)
    
    

def check_available_ai_tasks_benchmark_modeltrained(path_osmose_AI):
    path_osmose_AI = path_osmose_AI + os.sep
    
    for root, dirs, files in os.walk(path_osmose_AI):
        level = root.replace(path_osmose_AI, '').count(os.sep)
        indent = ' ' * 4 * (level)
        if level <= 3:
            if level >= 2: 
                if 'models' not in root:
                    continue
            print('{}{}/'.format(indent, os.path.basename(root)))
            

        
def check_available_ai_trainednetwork(path_osmose_dataset, dataset_ID, Task_ID, BM_Name, LengthFile, Fs):
    folderName_audioFiles = str(LengthFile)+'_'+str(int(Fs))
    base_path = path_osmose_dataset + os.sep + dataset_ID + os.sep + 'AI' + os.sep + Task_ID + os.sep + BM_Name + os.sep + folderName_audioFiles + os.sep + 'models'
        
    for root, dirs, files in os.walk(base_path):
        level = root.replace(base_path, '').count(os.sep)
        if level <= 1:
            indent = ' ' * 4 * (level)
            print('{}{}/'.format(indent, os.path.basename(root)))



def check_available_formats_from_dataset(path_osmose_dataset, dataset_ID):
    base_path_dataset = path_osmose_dataset + os.sep + dataset_ID + os.sep + 'processed' + os.sep + 'spectrogram' 
    print('_______________')
    print('Dataset : ',dataset_ID)
    print('Available Spectrogram Format (nfft_windowsize_overlap) :')
    for root, dirs, files in os.walk(base_path_dataset):
        level = root.replace(base_path_dataset, '').count(os.sep)
        indent = ' ' * 2 * (level-1)
        if level<3 and level>1:
            if 'adjustment_spectros' not in os.path.basename(root):
                print('{}{}/'.format(indent, os.path.basename(root))) 