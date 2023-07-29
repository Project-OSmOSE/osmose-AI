# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 18:00:06 2023

@author: gabri
"""

import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

from set_network_functions import transform_simple, transform_ref
from class_dataset import ClassDataset
#from evaluation_metrics import ComputeEvaluationMetrics, plot_PR_curve, plot_ROC_curve, plot_DET_curve, plot_COST_curve, plot_4_curves

def apply_model_on_new_format_dataset_main(path_osmose_dataset, path_osmose_analysisAI, Task_ID_Model, BM_Name_Model, model_name_Model, parameters_Model, Task_ID_Evaluation, BM_Name_Evaluation, SplitName_Evaluation, parameters_Evaluation):
    
    #%% DEVICE 
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    #device = 'cpu'
    
    #%% Dataset Path
    path_model = path_osmose_analysisAI + os.sep + Task_ID_Model + os.sep + BM_Name_Model + os.sep + 'models'

    
    
    #%% Load Model 
    if 'architecture' in list(parameters_Model.keys()):
        architecture = parameters_Model['architecture']
    else: architecture = parameters_Model['ModelName']
    
    model = torch.jit.load(path_model + os.sep + model_name_Model + os.sep + 'model_state' + os.sep + model_name_Model + '_Scripted_model.pt')
    model.eval()
        
    # Initialize the model for this run
    ref_model_list = ['resnet18', 'resnet50', 'resnet101', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg19', 'vgg19_bn', 'alexnet']
    if architecture in ref_model_list:
        trans = transform_ref
    elif architecture in ['CNN3_FC3','CNN4_FC3', 'CNN3_FC1'] :
        trans = transform_simple
    else: print('ERROR : Model not in list ...')
    #%% Load Format Dataset
    
    #load some metadata
    annot_param = np.load(path_osmose_analysisAI + os.sep + Task_ID_Evaluation + os.sep + BM_Name_Evaluation + os.sep + 'info_datasplit' + os.sep + 'Fdataset_metadata.npz')

    #Load Parameters    
    nfft_tab = parameters_Evaluation['nfft']
    window_size_tab = parameters_Evaluation['window_size']
    overlap_tab = parameters_Evaluation['overlap']
    dynamic_min_tab = parameters_Evaluation['dynamic_min']
    dynamic_max_tab = parameters_Evaluation['dynamic_max']
    input_data_format = parameters_Evaluation['input_data_format']
    
    LabelsList = annot_param['FinalLabelsList']
    dataset_ID_tab = annot_param['dataset_ID_tab']
    sample_rate_tab = annot_param['sample_rate_tab']
    LenghtFile_tab = annot_param['LenghtFile_tab']
    #Number of labels
    num_classes = len(LabelsList)
    NbDataset = len(dataset_ID_tab)
    parameters_Evaluation['dataset_ID_tab'] = dataset_ID_tab
    parameters_Evaluation['num_classes'] = num_classes
    parameters_Evaluation['sample_rate'] = sample_rate_tab
    
    if input_data_format == 'spectrogram':
        dynamic_min = [[] for i in range(NbDataset)]
        dynamic_max = [[] for i in range(NbDataset)]
        for i in range(NbDataset):
            path_spectro_metadata = path_osmose_dataset + os.sep + dataset_ID_tab[i] + os.sep + 'processed' + os.sep + 'spectrogram' +os.sep + str(LenghtFile_tab[i]) + '_' + str(sample_rate_tab[i]) + os.sep + str(nfft_tab[i]) + '_' + str(window_size_tab[i]) + '_' + str(overlap_tab[i]) + os.sep 
            param_spectro_csv = pd.read_csv(path_spectro_metadata + 'metadata.csv')
            dynamic_min[i] = param_spectro_csv['dynamic_min'][0]
            dynamic_max[i] = param_spectro_csv['dynamic_max'][0]
    
        parameters_Evaluation['dynamic_min'] = dynamic_min
        parameters_Evaluation['dynamic_max'] = dynamic_max
    
    if SplitName_Evaluation == 'ALL_DATASET':
        CSV_annotations_path = path_osmose_analysisAI + os.sep + Task_ID_Evaluation + os.sep + BM_Name_Evaluation + os.sep + 'info_datasplit' + os.sep + 'ALLannotations.csv'

    else:
        CSV_annotations_path = path_osmose_analysisAI + os.sep + Task_ID_Evaluation + os.sep + BM_Name_Evaluation + os.sep + 'info_datasplit' + os.sep +  SplitName_Evaluation + os.sep + 'EVALannotations.csv'
    
    dataset = ClassDataset(path_osmose_dataset, CSV_annotations_path, parameters_Evaluation, transform=trans)

    print('Done')
    print('Applying model on evaluation set ...')
    
    #%% Apply Model On Eval Dataset
    
    labels = np.zeros([len(dataset), len(LabelsList)])
    outputs = np.zeros([len(dataset), len(LabelsList)])
    filename = [0 for _ in range(len(dataset))]
    datasets = [[] for i in range(len(dataset))]

    for i in tqdm(range(len(dataset))):
        
        #get filename
        filename[i] = dataset.__getfilename__(i)
        datasets[i] = dataset.__getdataset__(i)
        
        #get data and label
        imgs, label = dataset.__getitem__(i)
        
        #to device
        imgs = imgs.to(device)
        labels_batch = label.to(device)
        #apply model
        outputs_batch = model(imgs[None,:].float())
        
        labels[i] = labels_batch.cpu().detach().numpy()[0]
        outputs[i] = outputs_batch.cpu().detach().numpy()
    
    
    return LabelsList, labels, outputs, dataset, filename
    
    
    
    

def plot_some_example(path_osmose_dataset, path_osmose_analysisAI, Task_ID_Model, BM_Name_Model, model_name_Model, parameters_Model, Task_ID_Evaluation, BM_Name_Evaluation, SplitName_Evaluation, parameters_Evaluation):
    
    #%% DEVICE 
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    #device = 'cpu'
    
    #%% Dataset Path
    path_model = path_osmose_analysisAI + os.sep + Task_ID_Model + os.sep + BM_Name_Model + os.sep + 'models'
    
    #%% Load Model 
    print('Loading trained model ...')

    if 'architecture' in list(parameters_Model.keys()):
        architecture = parameters_Model['architecture']
    else: architecture = parameters_Model['ModelName']

    model = torch.jit.load(path_model + os.sep + model_name_Model + os.sep + 'model_state' + os.sep + model_name_Model + '_Scripted_model.pt')
    model.eval()
        
    # Initialize the model for this run
    ref_model_list = ['resnet18', 'resnet50', 'resnet101', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg19', 'vgg19_bn', 'alexnet']
    if architecture in ref_model_list:
        trans = transform_ref
    elif architecture in ['CNN3_FC3', 'CNN3_FC1', 'CNN4_FC3'] :
        trans = transform_simple
        
    else: print('ERROR : Model not in list ...')
    #%% Load Format Dataset
    
    #load some metadata
    annot_param = np.load(path_osmose_analysisAI + os.sep + Task_ID_Evaluation + os.sep + BM_Name_Evaluation + os.sep + 'info_datasplit' + os.sep + 'Fdataset_metadata.npz')

    #Load Parameters    
    nfft_tab = parameters_Evaluation['nfft']
    window_size_tab = parameters_Evaluation['window_size']
    overlap_tab = parameters_Evaluation['overlap']
    dynamic_min_tab = parameters_Evaluation['dynamic_min']
    dynamic_max_tab = parameters_Evaluation['dynamic_max']
    input_data_format = parameters_Evaluation['input_data_format']
    
    LabelsList = annot_param['FinalLabelsList']
    dataset_ID_tab = annot_param['dataset_ID_tab']
    sample_rate_tab = annot_param['sample_rate_tab']
    LenghtFile_tab = annot_param['LenghtFile_tab']
    #Number of labels
    num_classes = len(LabelsList)
    NbDataset = len(dataset_ID_tab)
    parameters_Evaluation['dataset_ID_tab'] = dataset_ID_tab
    parameters_Evaluation['num_classes'] = num_classes
    parameters_Evaluation['sample_rate'] = sample_rate_tab
    
    if input_data_format == 'spectrogram':
        dynamic_min = [[] for i in range(NbDataset)]
        dynamic_max = [[] for i in range(NbDataset)]
        for i in range(NbDataset):
            path_spectro_metadata = path_osmose_dataset + os.sep + dataset_ID_tab[i] + os.sep + 'processed' + os.sep + 'spectrogram' +os.sep + str(LenghtFile_tab[i]) + '_' + str(sample_rate_tab[i]) + os.sep + str(nfft_tab[i]) + '_' + str(window_size_tab[i]) + '_' + str(overlap_tab[i]) + os.sep 
            param_spectro_csv = pd.read_csv(path_spectro_metadata + 'metadata.csv')
            dynamic_min[i] = param_spectro_csv['dynamic_min'][0]
            dynamic_max[i] = param_spectro_csv['dynamic_max'][0]
    
        parameters_Evaluation['dynamic_min'] = dynamic_min
        parameters_Evaluation['dynamic_max'] = dynamic_max
    
    if SplitName_Evaluation == 'ALL_DATASET':
        CSV_annotations_path = path_osmose_analysisAI + os.sep + Task_ID_Evaluation + os.sep + BM_Name_Evaluation + os.sep + 'info_datasplit' + os.sep + 'ALLannotations.csv'

    else:
        CSV_annotations_path = path_osmose_analysisAI + os.sep + Task_ID_Evaluation + os.sep + BM_Name_Evaluation + os.sep + 'info_datasplit' + os.sep +  SplitName_Evaluation + os.sep + 'EVALannotations.csv'
    
    dataset = ClassDataset(path_osmose_dataset, CSV_annotations_path, parameters_Evaluation, transform=trans)

    print('Done')
    #print('Applying model on evaluation set ...')
    
    #%% Apply Model On Eval Dataset
    
    filename = [0 for _ in range(len(dataset))]
    
    
     
    Label_Pos = {}
    nombre_exemple = 6
    
    for label in LabelsList:
        Label_Pos[label] = {'Outputs':[], 'Spectros':[], 'Labels':[]}
    
    Label_Neg = {'Outputs':[], 'Spectros':[], 'Labels':[]}
    break_condition = [nombre_exemple]*(num_classes+1)
    break_test = [ 0 for _ in range(num_classes+1)]
    
    random_sequence = list(np.linspace(0,len(dataset)-1, len(dataset), dtype=int))
    random.shuffle(random_sequence)
        
    for i in (random_sequence): 
        
        #get filename
        filename[i] = dataset.__getfilename__(i)
        
        #get data and label
        imgs, label = dataset.__getitem__(i)
        
        #to device
        imgs = imgs.to(device)
        labels = label.to(device)
        #apply model
        outputs = model(imgs[None,:].float())
        
        
        for k in range(len(labels)):
            if int(labels[k]) > 0 and break_test[k]<nombre_exemple:
                Label_Pos[LabelsList[k]]['Outputs'].append(outputs[0].detach().cpu().numpy())
                Label_Pos[LabelsList[k]]['Spectros'].append(imgs[0].detach().cpu().numpy())
                Label_Pos[LabelsList[k]]['Labels'].append(labels[0].detach().cpu().numpy())
                break_test[k] += 1
                break
    
        if labels.detach().cpu().numpy() == np.zeros(num_classes)  and break_test[num_classes]<nombre_exemple:
            Label_Neg['Outputs'].append(outputs[0].detach().cpu().numpy())
            Label_Neg['Spectros'].append(imgs[0].detach().cpu().numpy())
            Label_Neg['Labels'].append(labels[0].detach().cpu().numpy())
            break_test[num_classes] += 1
         
        if break_test == break_condition:
            break
        
    for k in range(num_classes):

        fig = plt.figure(figsize=(10,7), linewidth=3, edgecolor="k")
        for i in range(len(Label_Pos[LabelsList[k]]['Spectros'])):
            ax = fig.add_subplot(2,3,i+1)
            ax.pcolor(Label_Pos[LabelsList[k]]['Spectros'][i])
            plt.xlabel('Time (frame)')
            plt.ylabel('Freq (Bins)')
            plt.title(r'Labels : '+str(Label_Pos[LabelsList[k]]['Labels'][i])+' \n Outputs : '+str(Label_Pos[LabelsList[k]]['Outputs'][i])[:5]+']')
            plt.tight_layout()
        plt.suptitle('Some examples of : '+LabelsList[k], fontsize = 15)    
        plt.tight_layout()
        
    fig = plt.figure(figsize=(10,7), linewidth=5, edgecolor="k")
    for i in range(len(Label_Neg['Spectros'])):
        ax = fig.add_subplot(2,3,i+1)
        ax.pcolor(Label_Neg['Spectros'][i])
        plt.xlabel('Time (frame)')
        plt.ylabel('Freq (Bins)')
        plt.title(r'Labels : '+str(Label_Neg['Labels'][i])+' \n Outputs : '+str(Label_Neg['Outputs'][i])[:5]+']')
        plt.tight_layout()
    plt.suptitle('Some examples of : '+'Negatives', fontsize = 15)        
    plt.tight_layout()
    
    
    
def plot_examples_from_test(model_ft, test_loader, LabelsList):

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    #device = 'cpu'
    num_classes = len(LabelsList)
    Label_Pos = {}
    
    nombre_exemple = 6
    
    for label in LabelsList:
        Label_Pos[label] = {'Outputs':[], 'Spectros':[], 'Labels':[]}
    
    Label_Neg = {'Outputs':[], 'Spectros':[], 'Labels':[]}
    break_condition = [nombre_exemple]*(num_classes+1)
    break_test = [ 0 for _ in range(num_classes+1)]
    
    model_ft.eval()
    
    Nb_ex = len(test_loader.dataset)
    random_sequence = list(np.linspace(0,Nb_ex-1, Nb_ex, dtype=int))
    random.shuffle(random_sequence)
    
    for i in (random_sequence): 
        
        imgs, labels = test_loader.dataset[i]
        imgs = imgs.to(device)
        imgs = torch.unsqueeze(imgs, 0)
        labels = labels.to(device)
        outputs = model_ft(imgs.float())
        
        for k in range(len(labels)):
            if int(labels[k]) > 0 and break_test[k]<nombre_exemple:
                Label_Pos[LabelsList[k]]['Outputs'].append(outputs[0].detach().cpu().numpy())
                Label_Pos[LabelsList[k]]['Spectros'].append(imgs[0][0].detach().cpu().numpy())
                Label_Pos[LabelsList[k]]['Labels'].append(labels[0].detach().cpu().numpy())
                break_test[k] += 1
                break
    
        if labels.detach().cpu().numpy() == np.zeros(num_classes)  and break_test[num_classes]<nombre_exemple:
            Label_Neg['Outputs'].append(outputs[0].detach().cpu().numpy())
            Label_Neg['Spectros'].append(imgs[0][0].detach().cpu().numpy())
            Label_Neg['Labels'].append(labels[0].detach().cpu().numpy())
            break_test[num_classes] += 1
         
        if break_test == break_condition:
            break
        
    for k in range(num_classes):

        fig = plt.figure(figsize=(10,7), linewidth=3, edgecolor="k")
        for i in range(len(Label_Pos[LabelsList[k]]['Spectros'])):
            ax = fig.add_subplot(2,3,i+1)
            ax.pcolor(Label_Pos[LabelsList[k]]['Spectros'][i])
            plt.xlabel('Time (frame)')
            plt.ylabel('Freq (Bins)')
            plt.title(r'Labels : '+str(Label_Pos[LabelsList[k]]['Labels'][i])+' \n Outputs : '+str(Label_Pos[LabelsList[k]]['Outputs'][i])[:5]+']')
            plt.tight_layout()
        plt.suptitle('Some examples of : '+LabelsList[k], fontsize = 15)    
        plt.tight_layout()
        
    fig = plt.figure(figsize=(10,7), linewidth=5, edgecolor="k")
    for i in range(len(Label_Neg['Spectros'])):
        ax = fig.add_subplot(2,3,i+1)
        ax.pcolor(Label_Neg['Spectros'][i])
        plt.xlabel('Time (frame)')
        plt.ylabel('Freq (Bins)')
        plt.title(r'Labels : '+str(Label_Neg['Labels'][i])+' \n Outputs : '+str(Label_Neg['Outputs'][i])[:5]+']')
        plt.tight_layout()
    plt.suptitle('Some examples of : '+'Negatives', fontsize = 15)        
    plt.tight_layout()
                   
   