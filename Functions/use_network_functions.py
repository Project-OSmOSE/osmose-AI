# -*- coding: utf-8 -*-
"""
Created on Tue May  9 15:18:56 2023

@author: gabri
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch

def load_model_hyperparameters(path_osmose_analysisAI, Task_ID, BM_Name, model_name):
    '''
        INPUTS :
            - Task_ID : name of task (already existing)
            - BM_Name : name of Benchmark (already existing)
            - model_name : Name of the detection network that will be trainnned 
        OUTPUTS :
            - parameters : dictionnarry with all parameters 
        '''
        
    base_path = path_osmose_analysisAI + os.sep + Task_ID + os.sep + BM_Name + os.sep + 'models'
     
    file = np.load(base_path + os.sep + model_name + os.sep + 'hyper_parameters' + os.sep + model_name + '_hyperparameters.npz', allow_pickle=True)
    parameters = file['parameters'].item()
    #for item in list(parameters.keys()):
        #print(item, ' : ', parameters[item])
    return parameters

def show_epoch_loss(path_osmose_analysisAI, Task_ID, BM_Name, model_name):
    
    base_path = path_osmose_analysisAI + os.sep + Task_ID + os.sep + BM_Name + os.sep + 'models'
    trainning_param = np.load(base_path + os.sep + model_name + os.sep + 'train_curves' + os.sep + model_name + '_LossCurvesDATA.npz')
    
    loss_tab_train = trainning_param['loss_tab_train']
    loss_tab_validation = trainning_param['loss_tab_validation']
    epochs = np.linspace(1,len(loss_tab_validation),len(loss_tab_validation) )
    plt.figure(figsize=(6,2))
    plt.plot(epochs, loss_tab_train, label='Train')
    plt.plot(epochs, loss_tab_validation, label='Valid')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (mean)')
    plt.legend()
    plt.grid()

def set_epoch(path_osmose_analysisAI, Task_ID, BM_Name, model_name, ID_epoch_for_evaluation):  
       
    path_model = path_osmose_analysisAI + os.sep + Task_ID + os.sep + BM_Name + os.sep + 'models'
    if not os.path.exists(path_model + os.sep + model_name + os.sep + 'model_state' + os.sep + 'sub_state' + os.sep + model_name + '_epoch' + str(int(ID_epoch_for_evaluation))  + '_Scripted_model.pt'):
        print('Sub-states already deleted !')

    else:
        model_ft = torch.jit.load(path_model + os.sep + model_name + os.sep + 'model_state' + os.sep + 'sub_state' + os.sep + model_name + '_epoch' + str(int(ID_epoch_for_evaluation))  + '_Scripted_model.pt')
        #save model
        #torch.save(model_ft.state_dict(), path_model + os.sep + model_name + os.sep + 'model_state' + os.sep + model_name + '_model_t.pt')
        # save model as script
        model_ft_scripted = torch.jit.script(model_ft) # Export to TorchScript
        model_ft_scripted.save(path_model + os.sep + model_name + os.sep + 'model_state' + os.sep + model_name + '_Scripted_model.pt') # Save
        print('Done : Model state at epoch '+ str(ID_epoch_for_evaluation) + ' set for evaluation')
    
def del_unused_epochs(path_osmose_analysisAI, Task_ID, BM_Name, model_name):  
       
    path_model = path_osmose_analysisAI + os.sep + Task_ID + os.sep + BM_Name + os.sep + 'models'
    path_subset = path_model + os.sep + model_name + os.sep + 'model_state' + os.sep + 'sub_state'
    
    for file in glob.glob(path_subset + os.sep + '*'):
        os.remove(file)    

    print('Done : all other sub-states removed')
    
#def plot_examples_from_test()  
    
    
    
    
    
    