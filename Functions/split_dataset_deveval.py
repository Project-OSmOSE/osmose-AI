# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 12:22:13 2022

@author: gabri
"""
'''
Include Functions to splitt DEV (developpment) and EVAL (evaluation) sets from already existing dataset for network (as ALL_annotation.csv)

Librairies : Please, check the file "requierments.txt"

functions here : 
        - SplitDataset_DevEval_main() : MAIN

Paths needed : 
        - path_osmose_dataset : path of the datset with OSmOSE architecture
        - codes_path : path with all the functions needed 
'''

import numpy as np
import os
import pandas as pd
import random
import sys
from tqdm import tqdm




#%% MAIN
def SplitDataset_DevEval_main(Task_ID, BM_Name, SplitName, RatioDev, SelectionMethod, parameters):
    
    #Load Paths
    with open('path_osmose.txt') as f:
        path_osmose = f.readlines()[0]
    
    path_osmose_analysisAI = path_osmose + 'analysis' + os.sep + 'AI'
    path_osmose_dataset = path_osmose + 'dataset'
    
    metadata = np.load(path_osmose_analysisAI + os.sep + Task_ID + os.sep + BM_Name + os.sep + 'info_datasplit' + os.sep +  'Fdataset_metadata.npz')
    train_df = pd.read_csv(path_osmose_analysisAI + os.sep + Task_ID + os.sep + BM_Name + os.sep + 'info_datasplit' + os.sep + 'ALLannotations.csv')
    LabelsList = metadata['FinalLabelsList']
    NbFile = len(train_df)
    
    if not os.path.exists(path_osmose_analysisAI + os.sep + Task_ID + os.sep + BM_Name + os.sep + 'info_datasplit' + os.sep + SplitName + os.sep):
        os.makedirs(path_osmose_analysisAI + os.sep + Task_ID + os.sep + BM_Name + os.sep + 'info_datasplit' + os.sep + SplitName + os.sep)
    

    #%% Separate Dev, Validation
    # Label to be saved in the .csv files
    columns_name = list(train_df.columns.values)

    #%% Split Method 
        # For now, only 'FullyRandom' : all files are mixed and then we split according to RatioDev
    
    if  SelectionMethod == 'Continue':
        ord_sequence = list(np.linspace(0,NbFile-1, NbFile, dtype=int))
        
        DevSetArg = (ord_sequence[:int(RatioDev*NbFile)])
        EvalSetArg = (ord_sequence[int(RatioDev*NbFile):])
    
    if  SelectionMethod == 'FullyRandom':
        random_sequence = list(np.linspace(0,NbFile-1, NbFile, dtype=int))
        random.shuffle(random_sequence)
        
        DevSetArg = (random_sequence[:int(RatioDev*NbFile)])
        EvalSetArg = (random_sequence[int(RatioDev*NbFile):])
        
    if  SelectionMethod == 'RandomBySeq':
        NbFileInSequence = parameters['NbFileInSequence']
        NbSequence = round(NbFile/NbFileInSequence)
        
        ord_sequence = list(np.linspace(0,NbFile-1, NbFile, dtype=int))
        random_start_clust = list(np.arange(0,NbFile-NbFileInSequence, NbFileInSequence, dtype=int))
        random.shuffle(random_start_clust)# = sorted(random.choices(ord_sequence, k=round(RatioDev*NbSequence)))
        DevSetArg = []
        
        for file_id in tqdm(random_start_clust[:int(RatioDev*NbSequence)]):
            for i in range(NbFileInSequence):
                if (file_id + i) not in DevSetArg:
                    DevSetArg.append(file_id + i)
                    
        EvalSetArg = []
        for file_id in tqdm(ord_sequence):   
            if file_id not in DevSetArg:
                EvalSetArg.append(file_id)
    
        EvalSetArg = sorted(EvalSetArg)
        DevSetArg = sorted(DevSetArg)
        
    if SelectionMethod == 'SelPositiveRatio':
        PositiveRatio = parameters['PositiveRatio']
        DevSetArg = []
        
        ArgPos =  [ [] for i in range(len(LabelsList))]
        for id_file in range(NbFile):
            for label in LabelsList:
                if train_df[label][id_file] == 1:
                    ArgPos[list(LabelsList).index(label)].append(id_file)
                    
        for id_label in range(len(LabelsList)):
            random.shuffle(ArgPos[id_label])

        for id_label in range(len(LabelsList)):
            for j in range(min(int(RatioDev*NbFile*PositiveRatio), len(ArgPos[id_label]))):
                if ArgPos[id_label][j] not in DevSetArg:
                    DevSetArg.append(ArgPos[id_label][j])
                
        random_sequence = list(np.linspace(0,NbFile-1, NbFile, dtype=int))
        for id_file in random_sequence:
            if len(DevSetArg) <= NbFile*RatioDev:
                neg = 1
                for id_label in range(len(LabelsList)):
                    if id_file in ArgPos[id_label]: neg = 0
                if neg == 1:
                    if id_file not in DevSetArg:
                        DevSetArg.append(id_file)
            else:break
          
        ord_sequence = list(np.linspace(0,NbFile-1, NbFile, dtype=int))
        EvalSetArg = []
        for file_id in tqdm(ord_sequence):   
            if file_id not in DevSetArg:
                EvalSetArg.append(file_id)
        
    EvalSetArg = sorted(EvalSetArg)
    DevSetArg = sorted(DevSetArg)
        
    
    #%%
    
    # Create Dataframe that will be save as .csv for devellopment annotations
    train_df_dev = pd.DataFrame(columns=columns_name)
    train_df_dev["filename"] = [[]] * len(DevSetArg)
    
    # Get files for devellopment annotations
    for i in range(len(DevSetArg)):
        for col_val in columns_name[:3]:
            train_df_dev[col_val][i] = train_df[col_val][DevSetArg[i]]
        for label in LabelsList:
            train_df_dev[label][i] =  train_df[label][DevSetArg[i]]
    
    # Create Dataframe that will be save as .csv for evaluation annotations
    train_df_eval = pd.DataFrame(columns=columns_name)
    train_df_eval["filename"] = [[]] * len(EvalSetArg)
    
    # Get files for evaluation annotations
    for i in range(len(EvalSetArg)):
        for col_val in columns_name[:3]:
            train_df_eval[col_val][i] = train_df[col_val][EvalSetArg[i]]
        for label in LabelsList:
            train_df_eval[label][i] =  train_df[label][EvalSetArg[i]]
    
    # Reorganize Dataframe
    train_df_eval.dropna(subset = [columns_name[3]], inplace=True)
    train_df_dev.dropna(subset = [columns_name[3]], inplace=True)
    
    # Save dataframe as .csv
    train_df_dev.to_csv(path_osmose_analysisAI + os.sep + Task_ID + os.sep + BM_Name + os.sep + 'info_datasplit' + os.sep + SplitName + os.sep + 'DEVannotations.csv', index = False, header=True)
    train_df_eval.to_csv(path_osmose_analysisAI + os.sep + Task_ID + os.sep + BM_Name + os.sep + 'info_datasplit' + os.sep + SplitName + os.sep + 'EVALannotations.csv', index = False, header=True)
    
    
    #%% Print number of files in all subset and % of positive files for each label
    print('DEV :')
    
    print('Nombre de fichier : ', len(train_df_dev))
    for label in LabelsList:
        x = 100*np.count_nonzero(train_df_dev[label])/len(train_df_dev)
        print(label,' -> pourcentage de Positif : ', "{:10.3f}".format(x),'%')
    
    print('EVAL :')
    
    print('Nombre de fichier : ', len(train_df_eval))
    for label in LabelsList:
        try: x = 100*np.count_nonzero(train_df_eval[label])/len(train_df_eval)
        except ZeroDivisionError: x = 0
        print(label,' -> pourcentage de Positif : ', "{:10.3f}".format(x),'%')
        
    print(' ')
    print('Split is done ! You now can train a network on the development set and apply it on the evaluation set.')
    
