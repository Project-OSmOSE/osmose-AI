# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 13:56:55 2022

@author: gabri
"""

'''
Include Functions to apply a trainned model on the evaluation set and compute evaluation metrics to evaluate the network

Librairies : Please, check the file "requierments.txt"

functions here : 
        - ApplyModelOnEvalSet_main() : MAIN
        - LoadModelHyperParameters() : Load the models hyperparameters, weight, architecture, ...

Paths needed : 
        - path_osmose_dataset : path of the datset with OSmOSE architecture
        - codes_path : path with all the functions needed
            - ClasseDatasetForTorch.py
            - Transfer_Learning/Functions.py
            - EvaluationMetrics_ComputeAndPlot.py
'''

#%% Import 
import numpy as np
import os
import pandas as pd
import torch
from tqdm import tqdm

from class_dataset import ClassDataset
from evaluation_metrics import ComputeEvaluationMetrics, plot_4_curves
from set_network_functions import transform_simple, transform_ref

import matplotlib.pyplot as plt

#%% MAIN
def apply_model_on_eval_set_main(path_osmose_dataset, path_osmose_analysisAI, Task_ID, BM_Name, model_name):

    '''
        INPUTS :
            - dataset_ID : name of dataset (already existing)
            - Task_ID : name of task (already existing)
            - BM_Name : name of Benchmark (already existing)
            - LengthFile : duration of input files (in second)
            - sample_rate : Sampling Rate (in Hz)
            - model_name : Name of the detection network that will be trainnned 
            - Dyn : array with minimum and maximum levels for the spectrograms (in dB) 
            - LabelsList : List of label to be detected
            - SplitName : label the Dev/Eval split to use
        '''    

    #%% DEVICE 
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    
    print('Checking paths and loadings parameters and model ...')
       
    path_model = path_osmose_analysisAI + os.sep + Task_ID + os.sep + BM_Name + os.sep + 'models'

    #load some metadata
    file_hyperparameters = np.load(path_model + os.sep + model_name + os.sep + 'hyper_parameters' + os.sep + model_name + '_hyperparameters.npz', allow_pickle=True)
    parameters = file_hyperparameters['parameters'].item()
    annot_param = np.load(path_osmose_analysisAI + os.sep + Task_ID + os.sep + BM_Name + os.sep + 'info_datasplit' + os.sep + 'Fdataset_metadata.npz')

    #Load Parameters    
    nfft_tab = parameters['nfft']
    window_size_tab = parameters['window_size']
    overlap_tab = parameters['overlap']
    dynamic_min_tab = parameters['dynamic_min']
    dynamic_max_tab = parameters['dynamic_max']
    input_data_format = parameters['input_data_format']
    
    if 'architecture' in list(parameters.keys()):
        architecture = parameters['architecture']
    else: architecture = parameters['ModelName']
    SplitName = parameters['SplitName']
    
    LabelsList = annot_param['FinalLabelsList']
    dataset_ID_tab = annot_param['dataset_ID_tab']
    sample_rate_tab = parameters['sample_rate']
    LenghtFile_tab = annot_param['LenghtFile_tab']
    #Number of labels
    num_classes = len(LabelsList)
    NbDataset = len(dataset_ID_tab)
    
    
    if input_data_format == 'spectrogram':
        dynamic_min = [[] for i in range(NbDataset)]
        dynamic_max = [[] for i in range(NbDataset)]
        for i in range(NbDataset):
            path_spectro_metadata = path_osmose_dataset + os.sep + dataset_ID_tab[i] + os.sep + 'processed' + os.sep + 'spectrogram' +os.sep + str(LenghtFile_tab[i]) + '_' + str(sample_rate_tab[i]) + os.sep + str(nfft_tab[i]) + '_' + str(window_size_tab[i]) + '_' + str(overlap_tab[i]) + os.sep 
            param_spectro_csv = pd.read_csv(path_spectro_metadata + 'metadata.csv')
            dynamic_min[i] = param_spectro_csv['dynamic_min'][0]
            dynamic_max[i] = param_spectro_csv['dynamic_max'][0]
    
        parameters['dynamic_min'] = dynamic_min
        parameters['dynamic_max'] = dynamic_max
    
        
    #%% Load Model and annotation
    model = torch.jit.load(path_model + os.sep + model_name + os.sep + 'model_state' + os.sep + model_name + '_Scripted_model.pt')
    model.eval()
        
    # Initialize the model for this run
    ref_model_list = ['resnet18', 'resnet50', 'resnet101', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg19', 'vgg19_bn', 'alexnet']
    if architecture in ref_model_list:
        trans = transform_ref
    elif architecture in ['CNN3_FC3', 'CNN3_FC1', 'CNN4_FC3'] :
        trans = transform_simple

    CSV_annotations_path = path_osmose_analysisAI + os.sep + Task_ID + os.sep + BM_Name + os.sep + 'info_datasplit' + os.sep +  SplitName + os.sep + 'EVALannotations.csv'
    dataset = ClassDataset(path_osmose_dataset, CSV_annotations_path, parameters, transform=trans)

    print('Done')
    print('Applying model on evaluation set ...')
    #%% Apply Model On All Dataset
    
    labels = np.zeros([len(dataset), len(LabelsList)])
    outputs = np.zeros([len(dataset), len(LabelsList)])
    datasets = [[] for i in range(len(dataset))]
    filename = [[] for i in range(len(dataset))]
    
    for i in tqdm(range(len(dataset))):
        
        #get data and label
        imgs, label = dataset.__getitem__(i)
        
        #to device
        imgs = imgs.to(device)
        labels_batch = label.to(device)
        #apply model
        outputs_batch = model(imgs[None,:].float())
        
        labels[i] = labels_batch.cpu().detach().numpy()[0]
        outputs[i] = outputs_batch.cpu().detach().numpy()
        datasets[i] = dataset.__getdataset__(i)
        filename[i] = dataset.__getfilename__(i)
    #%% Compute Evaluation Index
    Recall, Precision, FP_rate, TP_rate, FN_rate, NormalizedExpectedCost, ProbabilityCost, threshold_array = ComputeEvaluationMetrics(LabelsList, labels, outputs)
    
    print('Done')
    #%% PLOT PR DET ROC Curve
    np.savez(path_model + os.sep + model_name + os.sep + 'train_curves' + os.sep + model_name + '_EvaluationMetrics_DATA.npz', Recall=Recall, Precision=Precision, FP_rate=FP_rate, TP_rate=TP_rate, FN_rate=FN_rate, NormalizedExpectedCost=NormalizedExpectedCost, ProbabilityCost=ProbabilityCost, LabelsList=LabelsList, labels=labels, outputs=outputs, threshold_array=threshold_array)
    for id_specie in range(len(LabelsList)):
        #print('Result for label : ', LabelsList[id_specie])
        
        savepath = path_model + os.sep + model_name + os.sep + 'train_curves' + os.sep + str(model_name) + '_' + str(LabelsList[id_specie]) + '_4Curves.png'
        plot_4_curves(Recall[:,id_specie], Precision[:,id_specie], FP_rate[:,id_specie], TP_rate[:,id_specie], FN_rate[:,id_specie], ProbabilityCost, NormalizedExpectedCost[:,id_specie,:].T, savepath = savepath, color='b', figsize=(8,8), title = 'Result for label : '+ str(LabelsList[id_specie]))
        
        '''
        
        savepath = path_model + os.sep + model_name + os.sep + 'train_curves' + os.sep + str(model_name) + '_' + str(LabelsList[id_specie]) + '_PRCurves.png'
        plot_PR_curve(Recall[:,id_specie], Precision[:,id_specie], savepath = savepath, color='b', xlim=[0,1], ylim=[0,1], figsize=(4,4))
        
        savepath = path_model + os.sep + model_name + os.sep + 'train_curves' + os.sep + str(model_name) + '_' + str(LabelsList[id_specie]) + '_ROCCurves.png'
        plot_ROC_curve(FP_rate[:,id_specie], TP_rate[:,id_specie], savepath = savepath, color='b', xlim=[0,1], ylim=[0,1], figsize=(4,4))
   
        savepath = path_model + os.sep + model_name + os.sep + 'train_curves' + os.sep + str(model_name) + '_' + str(LabelsList[id_specie]) + '_DETCurves.png'
        plot_DET_curve(FP_rate[:,id_specie], FN_rate[:,id_specie], savepath = savepath, color='b', xlim=[0.005,0.8], ylim=[0.005,0.8], figsize=(4,4))
        
        savepath = path_model + os.sep + model_name + os.sep + 'train_curves' + os.sep + str(model_name) + '_' + str(LabelsList[id_specie]) + '_CostCurves.png'
        plot_COST_curve(ProbabilityCost, NormalizedExpectedCost[:,id_specie,:].T, savepath = savepath, color='b', xlim=[0,1], ylim=[0,0.5], figsize=(4,4))
       
        '''

    return LabelsList, labels, outputs, datasets, filename

def plot_examples_from_file_index(path_osmose_dataset, path_osmose_analysisAI, Task_ID, BM_Name, model_name, index_to_plot):

    '''
        INPUTS :
            - dataset_ID : name of dataset (already existing)
            - Task_ID : name of task (already existing)
            - BM_Name : name of Benchmark (already existing)
            - LengthFile : duration of input files (in second)
            - sample_rate : Sampling Rate (in Hz)
            - model_name : Name of the detection network that will be trainnned 
            - Dyn : array with minimum and maximum levels for the spectrograms (in dB) 
            - LabelsList : List of label to be detected
            - SplitName : label the Dev/Eval split to use
        '''    

    #%% DEVICE 
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    
    print('Checking paths and loadings parameters and model ...')
       
    path_model = path_osmose_analysisAI + os.sep + Task_ID + os.sep + BM_Name + os.sep + 'models'

    #load some metadata
    file_hyperparameters = np.load(path_model + os.sep + model_name + os.sep + 'hyper_parameters' + os.sep + model_name + '_hyperparameters.npz', allow_pickle=True)
    parameters = file_hyperparameters['parameters'].item()
    annot_param = np.load(path_osmose_analysisAI + os.sep + Task_ID + os.sep + BM_Name + os.sep + 'info_datasplit' + os.sep + 'Fdataset_metadata.npz')

    #Load Parameters    
    nfft_tab = parameters['nfft']
    window_size_tab = parameters['window_size']
    overlap_tab = parameters['overlap']
    input_data_format = parameters['input_data_format']
    
    if 'architecture' in list(parameters.keys()):
        architecture = parameters['architecture']
    else: architecture = parameters['ModelName']
    SplitName = parameters['SplitName']
    
    LabelsList = annot_param['FinalLabelsList']
    dataset_ID_tab = annot_param['dataset_ID_tab']
    sample_rate_tab = parameters['sample_rate']
    LenghtFile_tab = annot_param['LenghtFile_tab']
    #Number of labels
    NbDataset = len(dataset_ID_tab)
    
    
    if input_data_format == 'spectrogram':
        dynamic_min = [[] for i in range(NbDataset)]
        dynamic_max = [[] for i in range(NbDataset)]
        for i in range(NbDataset):
            path_spectro_metadata = path_osmose_dataset + os.sep + dataset_ID_tab[i] + os.sep + 'processed' + os.sep + 'spectrogram' +os.sep + str(LenghtFile_tab[i]) + '_' + str(sample_rate_tab[i]) + os.sep + str(nfft_tab[i]) + '_' + str(window_size_tab[i]) + '_' + str(overlap_tab[i]) + os.sep 
            param_spectro_csv = pd.read_csv(path_spectro_metadata + 'metadata.csv')
            dynamic_min[i] = param_spectro_csv['dynamic_min'][0]
            dynamic_max[i] = param_spectro_csv['dynamic_max'][0]
    
        parameters['dynamic_min'] = dynamic_min
        parameters['dynamic_max'] = dynamic_max
    
        
    #%% Load Model and annotation
    model = torch.jit.load(path_model + os.sep + model_name + os.sep + 'model_state' + os.sep + model_name + '_Scripted_model.pt')
    model.eval()
        
    # Initialize the model for this run
    ref_model_list = ['resnet18', 'resnet50', 'resnet101', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg19', 'vgg19_bn', 'alexnet']
    if architecture in ref_model_list:
        trans = transform_ref
    elif architecture in ['CNN3_FC3', 'CNN3_FC1', 'CNN4_FC3'] :
        trans = transform_simple

    CSV_annotations_path = path_osmose_analysisAI + os.sep + Task_ID + os.sep + BM_Name + os.sep + 'info_datasplit' + os.sep +  SplitName + os.sep + 'EVALannotations.csv'
    dataset = ClassDataset(path_osmose_dataset, CSV_annotations_path, parameters, transform=trans)

    print('Done')
    #%% Apply Model On All Dataset
    
    labels = np.zeros([len(dataset), len(LabelsList)])
    outputs = np.zeros([len(dataset), len(LabelsList)])
    datasets = [[] for i in range(len(dataset))]
    filename = [[] for i in range(len(dataset))]
    
    for i in index_to_plot:
        
        #get data and label
        imgs, label = dataset.__getitem__(i)
        
        #to device
        imgs = imgs.to(device)
        labels_batch = label.to(device)
        #apply model
        outputs_batch = model(imgs[None,:].float())
        
        labels[i] = labels_batch.cpu().detach().numpy()[0]
        outputs[i] = outputs_batch.cpu().detach().numpy()
        datasets[i] = dataset.__getdataset__(i)
        filename[i] = dataset.__getfilename__(i)
        

        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(1,1,1)
        ax.pcolor(imgs[0].detach().cpu().numpy())
        plt.xlabel('Time (frame)')
        plt.ylabel('Freq (Bins)')
        plt.title(r'Filename : ' + filename[i] + ' \n Labels : ' + str(labels[i]) +  ' \n Outputs : ' + str(outputs[i]))
        plt.tight_layout()
    
    
