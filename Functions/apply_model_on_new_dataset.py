# -*- coding: utf-8 -*-
"""
Created on Tue May  9 18:23:52 2023

@author: gabri
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from set_network_functions import transform_simple, transform_ref

def apply_model_on_new_dataset_main(dataset_ID, Task_ID, BM_Name, Version_name, application_parameters):
    
    #%% DEVICE 
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    #device = 'cpu'
    
    #%% Load Paths, Model, Parameters, ...
    #Load Paths
    with open('path_osmose.txt') as f:
        path_osmose = f.readlines()[0]
    
    path_osmose_analysisAI = path_osmose + 'analysis' + os.sep + 'AI'
    path_osmose_dataset = path_osmose + 'dataset' 
    
    path_model = path_osmose + 'analysis' + os.sep + 'AI' + os.sep + Task_ID + os.sep + BM_Name + os.sep + 'models'

    #load some metadata
    file_hyperparameters = np.load(path_model + os.sep + Version_name + os.sep + 'hyper_parameters' + os.sep + Version_name + '_hyperparameters.npz', allow_pickle=True)
    parameters = file_hyperparameters['parameters'].item()
    annot_param = np.load(path_osmose_analysisAI + os.sep + Task_ID + os.sep + BM_Name + os.sep + 'info_datasplit' + os.sep + 'Fdataset_metadata.npz')
    
    LenghtFile = application_parameters['LenghtFile']
    sample_rate = application_parameters['sample_rate']
    input_data_format = application_parameters['input_data_format']
    
    LabelsList = annot_param['FinalLabelsList']
    
    folderName_audioFiles = str(LenghtFile)+'_'+str(int(sample_rate))
    
    #List All WavFile From Dataset
    base_path = path_osmose_dataset + os.sep + dataset_ID + os.sep
    path_audio_files = os.path.join(path_osmose_dataset, dataset_ID, 'data','audio', folderName_audioFiles )
    list_wavfile = [os.path.basename(x) for x in glob.glob(os.path.join(path_audio_files , '*wav'))]
    
    #%% Create Dataframe for the dataset, to read it with class dataset for torch
    '''
    NbFile = len(list_wavfile)
    
    #Load timestamp
    timestamp_file = glob.glob(os.path.join(path_audio_files , '*timestamp.csv'))
    xl_timestamp = pd.read_csv(timestamp_file[0], header=None,)  
    
    # Prepare CSV annotation for DL
    columns_name = LabelsList.copy()
    columns_name.insert(0, "filename")
    
    # Reorganise to save in DataFrame
    TAB_FINAL = []
    
    TAB_AnnotationPerFile = np.zeros([len(list_wavfile), len(LabelsList)])
    
    for i in range(len(list_wavfile)):
        ind = list(xl_timestamp[0]).index(list_wavfile[i])
        line = [dataset_ID]
        line.append([folderName_audioFiles])
        line.append(list([xl_timestamp[0][ind]]))
        for j in range(len(LabelsList)):
            line.append(TAB_AnnotationPerFile[i,j])
        TAB_FINAL.append(line)
        
        
    # Creat DataFrame
    format_dataset_df = pd.DataFrame(columns=columns_name)
    format_dataset_df["filename"] = [[]] * NbFile
    format_dataset_df = pd.DataFrame(TAB_FINAL, columns=columns_name)
    
    #format_dataset_df.to_csv(path_osmose_analysisAI +os.sep + Task_ID + os.sep + BM_Name + os.sep + 'info_datasplit' + os.sep + 'ALLannotations.csv', index = False, header=True)
    format_dataset_df.to_csv('log_todelete.csv', index = False, header=True)
    '''
    #%% Load Model 
    ModelName = parameters['ModelName']
    
    model = torch.jit.load(path_model + os.sep + Version_name + os.sep + 'model_state' + os.sep + Version_name + '_Scripted_model.pt')
    model.to(device) 
    model = model.eval()
    
    # Initialize the model for this run
    ref_model_list = ['resnet18', 'resnet50', 'resnet101', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg19', 'vgg19_bn', 'alexnet']
    if ModelName in ref_model_list:
        trans_spectro = transform_ref
    elif ModelName in ['CNN3_FC3', 'CNN3_FC1'] :
        trans_spectro = transform_simple


    #CSV_annotations_path = path_osmose_analysisAI + os.sep + Task_ID + os.sep + BM_Name + os.sep + 'info_datasplit' + os.sep +  SplitName + os.sep + 'EVALannotations.csv'
    #dataset = ClassDataset(path_osmose_dataset, CSV_annotations_path, parameters, transform=trans)

    print('Done')
    
    #%%
    Nfft = application_parameters['nfft']
    window_size = application_parameters['window_size']
    overlap = application_parameters['overlap']

    if input_data_format == 'spectrogram':
        folder_spectro = str(int(Nfft)) + '_' + str(int(window_size)) + '_' + str(int(overlap))
        file_path = path_osmose_dataset + os.sep + dataset_ID + os.sep + 'processed' + os.sep + 'spectrogram' + os.sep + folderName_audioFiles + os.sep + folder_spectro + os.sep + 'matrix' + os.sep 
        list_file_dataset = glob.glob(file_path + '*.npz')
        print(file_path)

    if input_data_format == 'audio':
        list_file_dataset = glob.glob(path_audio_files, '*.wav')
        #dataset_list = 
    
    
    #%%
    
    for i in tqdm(range(len(list_file_dataset))):
        file_path = list_file_dataset[i]
        
        if input_data_format == 'spectrogram':
            spectro_npz = np.load(file_path)
            spectro = spectro_npz['Sxx']
            
        parameters['index_dataset'] = 0
        spectro = trans_spectro(spectro, parameters)
        spectro = spectro.to(device)
        spectro.requires_grad = False
        spectro = spectro[None, :,:,:]
        output = model(spectro.float())
        print(output)
    
    
    
    
    
    
    
    
    
    