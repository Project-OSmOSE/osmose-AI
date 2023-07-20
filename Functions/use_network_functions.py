# -*- coding: utf-8 -*-
"""
Created on Tue May  9 15:18:56 2023

@author: gabri
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch

with open('path_osmose.txt') as f:
    path_osmose = f.readlines()[0]

def load_model_hyperparameters(Task_ID, BM_Name, Version_name):
    '''
        INPUTS :
            - Task_ID : name of task (already existing)
            - BM_Name : name of Benchmark (already existing)
            - Version_name : Name of the detection network that will be trainnned 
        OUTPUTS :
            - parameters : dictionnarry with all parameters 
        '''
        
    base_path = path_osmose + 'analysis' + os.sep + 'AI' + os.sep + Task_ID + os.sep + BM_Name + os.sep + 'models'
     
    file = np.load(base_path + os.sep + Version_name + os.sep + 'hyper_parameters' + os.sep + Version_name + '_hyperparameters.npz', allow_pickle=True)
    parameters = file['parameters'].item()
    #for item in list(parameters.keys()):
        #print(item, ' : ', parameters[item])
    return parameters

def show_epoch_loss(Task_ID, BM_Name, Version_name):
    
    base_path = path_osmose + 'analysis' + os.sep + 'AI' + os.sep + Task_ID + os.sep + BM_Name + os.sep + 'models'
     
    file = np.load(base_path + os.sep + Version_name + os.sep + 'hyper_parameters' + os.sep + Version_name + '_hyperparameters.npz', allow_pickle=True)
    parameters = file['parameters'].item()
    
    trainning_param = np.load(base_path + os.sep + Version_name + os.sep + 'train_curves' + os.sep + Version_name + '_LossCurvesDATA.npz')
    
    loss_tab_train = trainning_param['loss_tab_train']
    loss_tab_test = trainning_param['loss_tab_test']
    epochs = np.linspace(1,len(loss_tab_test),len(loss_tab_test) )
    plt.figure(figsize=(6,2))
    plt.plot(epochs, loss_tab_train, label='Train')
    plt.plot(epochs, loss_tab_test, label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (mean)')
    plt.legend()
    plt.grid()

def set_epoch(Task_ID, BM_Name, Version_name, ID_epoch_for_evaluation):  
       
    path_model = path_osmose + 'analysis' + os.sep + 'AI' + os.sep + Task_ID + os.sep + BM_Name + os.sep + 'models'
    model_ft = torch.jit.load(path_model + os.sep + Version_name + os.sep + 'model_state' + os.sep + 'sub_state' + os.sep + Version_name + '_epoch' + str(int(ID_epoch_for_evaluation))  + '_Scripted_model.pt')
    #save model
    #torch.save(model_ft.state_dict(), path_model + os.sep + Version_name + os.sep + 'model_state' + os.sep + Version_name + '_model_t.pt')
    # save model as script
    model_ft_scripted = torch.jit.script(model_ft) # Export to TorchScript
    model_ft_scripted.save(path_model + os.sep + Version_name + os.sep + 'model_state' + os.sep + Version_name + '_Scripted_model.pt') # Save
    
    
    
    
#def plot_examples_from_test()  
    
    
    
    
    
    