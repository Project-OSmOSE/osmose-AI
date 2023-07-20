# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 12:22:13 2022

@author: gabri
"""
'''
Include Functions to Train Network on already existing DEVset on datarmore

Librairies : Please, check the file "requierments.txt"

functions here : 
        - TrainNetwork_main() : MAIN
        - train() : loop over train and test loader

Paths needed : 
        - path_osmose_dataset : path of the datset with OSmOSE architecture
        - codes_path : path with all the functions needed 
            - ClasseDatasetForTorch.py
            - Transfer_Learning/Functions.py
'''

import numpy as np
from torch.utils.data import DataLoader
import os
import pandas as pd
import sys
import torch
import random
import matplotlib.pyplot as plt


from class_dataset import ClassDataset
from set_network_functions import CNN3_FC3, CNN4_FC3, CNN3_FC1, transform_simple, initialize_model, transform_ref, bce_loss


#Train Loop
def train(device, model_ft, optimizer, num_epochs, train_loader, test_loader, weight, model_path, Version_name, ModelName):
    
        '''
        INPUTS :
            - device : device cpu or gpu to optimize computation
            - model_ft : torch model to be trainned
            - num_epochs : number of iteration in the trainning
            - train_loader : torch dataset with trainning files and labels for train
            - test_loader : torch dataset with trainning files and labels for test
            - weight : weight on each label - now, there are at put ones 
            
        OUTPUTS :
            - loss_tab_train : loss value for test at each iteration
            - loss_tab_test : loss value for train at each iteration
        '''
    
        #initialize loss tab for plot
        loss_tab_train = []
        loss_tab_test = []
        model_ft.train()
        for epoch in range(num_epochs):
            model_ft.train()
            ite = 1
            loss_sum_train = 0
            epoch_p = epoch + 1
            #Loop For over train set
            for imgs, labels in train_loader:
                #load data and label, send them to device
                imgs = imgs.to(device)
                labels = labels.to(device)
                #apply model
                outputs = model_ft(imgs.float())
                #compute loss and backward (gradient)
                loss = bce_loss(outputs, labels, weight)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #loss mean over the epoch
                loss_sum_train += loss.item()
                PrintedLine = f"Epoch TRAIN [{epoch_p}/{num_epochs}]" + '  -- Loss = '+ str(loss_sum_train/ite) + '  --  ' + f"iteration [{ite}/{len(train_loader)}]" 
                sys.stdout.write('\r'+PrintedLine)
                ite += 1
            loss_tab_train.append(loss_sum_train/ite)
            
            print('  ')
            
            #TEST - SAME AS PREVIEWS ONE WITHOUT BACKWARD
            model_ft.eval()
            ite = 1
            loss_sum_test = 0
            for imgs, labels in test_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model_ft(imgs.float())
                loss = bce_loss(outputs, labels, weight)
                loss_sum_test += loss.item()
                PrintedLine = f"Epoch TEST [{epoch_p}/{num_epochs}]" + '  -- Loss = '+ str(loss_sum_test/ite) + '  --  ' + f"iteration [{ite}/{len(test_loader)}]" 
                sys.stdout.write('\r'+PrintedLine)
                ite += 1
            loss_tab_test.append(loss_sum_test/ite)
            print('  ')
            
            #%% Save Model 
            #save model
            torch.save(model_ft.state_dict(),model_path + os.sep + Version_name + os.sep + 'model_state' + os.sep + 'sub_state' + os.sep + Version_name + '_epoch'+ str(int(epoch_p))+ '_model.pt')
            # save model as script
            model_ft_scripted = torch.jit.script(model_ft) # Export to TorchScript
            model_ft_scripted.save(model_path + os.sep + Version_name + os.sep + 'model_state' + os.sep + 'sub_state' + os.sep + Version_name + '_epoch'+ str(int(epoch_p)) + '_Scripted_model.pt') # Save
        
            
        return loss_tab_train, loss_tab_test

def TrainNetwork_main(Task_ID, BM_Name, SplitName, Version_name, ModelName, parameters):
    '''
        INPUTS :
            - dataset_ID : name of dataset (already existing)
            - Task_ID : name of task (already existing)
            - BM_Name : name of Benchmark (already existing)
            - LengthFile : duration of input files (in second)
            - sample_rate : Sampling Rate (in Hz)
            - SplitName : label the Dev/Eval split to use
            - Version_name : Name of the detection network that will be trainnned 
            - ModelName : name of the reference model to be used (one in already existing list, check in Transfer_Learning/Functions.py file)
            - use_pretrained : True or False if you want to initialize your model with pre-trainned weight - please check PyTorch documentation
            - TrainSetRatio : Ratio between 0 and 1 for the train set from the dev set
            - batch_size : Number of file in one batch
            - learning_rate : learning rate
            - num_epochs : number of iteration in trainning loop
            - Dyn : array with minimum and maximum levels for the spectrograms (in dB) 
        '''
        
        
    #%% DEVICE 
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    #device = 'cpu'
    
    #%% LOAD PARAMETERS AND DEFINE PATHS
    with open('path_osmose.txt') as f:
        path_osmose = f.readlines()[0]
        
    path_osmose_analysisAI = path_osmose + 'analysis' + os.sep + 'AI'
    path_osmose_dataset = path_osmose + 'dataset'
    
    #Parameters model
    use_pretrained = parameters['use_pretrained']
    
    #parameters trainning
    TrainsetRatio = parameters['TrainsetRatio']
    batch_size = parameters['batch_size']
    learning_rate = parameters['learning_rate']
    num_epochs = parameters['num_epochs']
    shuffle = parameters['shuffle']
    input_data_format = parameters['input_data_format']
    

    #path for save trainng data
    model_path = path_osmose_analysisAI + os.sep + Task_ID + os.sep + BM_Name + os.sep + 'models'
    if not os.path.exists(model_path + os.sep + Version_name):
        os.makedirs(model_path + os.sep + Version_name)
        os.makedirs(model_path + os.sep + Version_name + os.sep + 'hyper_parameters')
        os.makedirs(model_path + os.sep + Version_name + os.sep + 'train_curves')
        os.makedirs(model_path + os.sep + Version_name + os.sep + 'model_state')
        os.makedirs(model_path + os.sep + Version_name + os.sep + 'model_state' + os.sep + 'sub_state')

    #Import some param
    #Annotations
    annot_param = np.load(path_osmose_analysisAI + os.sep + Task_ID + os.sep + BM_Name + os.sep + 'info_datasplit' + os.sep + 'Fdataset_metadata.npz')
    #List of labels to be detected
    LabelsList = annot_param['FinalLabelsList']
    dataset_ID_tab = annot_param['dataset_ID_tab']
    sample_rate_tab = annot_param['sample_rate_tab']
    LenghtFile_tab = annot_param['LenghtFile_tab']
    #Number of labels
    num_classes = len(LabelsList)
    NbDataset = len(dataset_ID_tab)
    
    parameters['num_classes'] = num_classes
    parameters['sample_rate'] = sample_rate_tab
    parameters['dataset_ID_tab'] = dataset_ID_tab
    
    #parameters spectrograms
    nfft = parameters['nfft']
    window_size = parameters['window_size']
    overlap = parameters['overlap']

    if input_data_format == 'spectrogram':
        dynamic_min = [[] for i in range(NbDataset)]
        dynamic_max = [[] for i in range(NbDataset)]
        for i in range(NbDataset):
            path_spectro_metadata = path_osmose_dataset + os.sep + dataset_ID_tab[i] + os.sep + 'processed' + os.sep + 'spectrogram' +os.sep + str(LenghtFile_tab[i]) + '_' + str(sample_rate_tab[i]) + os.sep + str(nfft[i]) + '_' + str(window_size[i]) + '_' + str(overlap[i]) + os.sep 
            param_spectro_csv = pd.read_csv(path_spectro_metadata + 'metadata.csv')
            dynamic_min[i] = param_spectro_csv['dynamic_min'][0]
            dynamic_max[i] = param_spectro_csv['dynamic_max'][0]
    
        parameters['dynamic_min'] = dynamic_min
        parameters['dynamic_max'] = dynamic_max
        
    #%% Def AI Param and import model
    print('MODEL INFORMATION : ')
    print(' ')
    pin_memory = True
    feature_extract = True
    num_workers = 1
    drop_last = True
    
    #Useless for now, but it will be possible to change weight in next release
    weight = torch.ones([num_classes]).to(device)

    capacity = 64
    # Initialize the model for this run
    ref_model_list = ['resnet18', 'resnet50', 'resnet101', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg19', 'vgg19_bn', 'alexnet']
    
    if ModelName in ref_model_list:
        model_ft, input_size = initialize_model(ModelName, num_classes, feature_extract, use_pretrained=use_pretrained)
        trans = transform_ref
        
    elif ModelName == 'CNN3_FC3':
        model_ft = CNN3_FC3(output_dim=num_classes, c=capacity)
        trans = transform_simple
        
    elif ModelName == 'CNN3_FC1':
        model_ft = CNN3_FC1(output_dim=num_classes, c=capacity)
        trans = transform_simple
        
    elif ModelName == 'CNN4_FC3':
        model_ft = CNN4_FC3(output_dim=num_classes, c=capacity)
        trans = transform_simple
        
    else : print('ERROR : Model not in list ...')
    
    #send to device
    model_ft.to(device)
    # Print the model we just instantiated
    print(model_ft)
    
#%% Import DEV Annotation, Set Variables
    train_df_dev = pd.read_csv(path_osmose_analysisAI + os.sep + Task_ID + os.sep + BM_Name + os.sep + 'info_datasplit' + os.sep +  SplitName + os.sep + 'DEVannotations.csv')
    # Number of files for developpment
    NbFile = len(train_df_dev)

#%% DEF TEST AND TRAIN SET

    '''
    Nb : For the moment, the split is simply done by cutting in one spot, new possibilities will be added
    '''
    
    CSV_annotations_path = path_osmose_analysisAI + os.sep + Task_ID + os.sep + BM_Name + os.sep + 'info_datasplit' + os.sep + SplitName + os.sep + 'DEVannotations.csv'
    dataset = ClassDataset(path_osmose_dataset, CSV_annotations_path, parameters, transform=trans)

    # Created using indices from 0 to train_size.
    train_set = torch.utils.data.Subset(dataset, range(int(TrainsetRatio*NbFile)))
    train_loader = DataLoader(dataset=train_set, shuffle=shuffle, batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory, drop_last=drop_last)
    
    # Created using indices from Train_size to the end.
    test_set = torch.utils.data.Subset(dataset, range(int(TrainsetRatio*NbFile),NbFile))
    test_loader = DataLoader(dataset=test_set, shuffle=shuffle, batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory, drop_last=drop_last)
    #%% DEFINE PARAM TO BE OPTIMIZED
    
    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)
    
    # Observe that all parameters are being optimized
    optimizer = torch.optim.SGD(params_to_update, lr=learning_rate, momentum=0.9)
    
    print(' ')
    print('TRAINNING : ')
    
    #Launch the training 
    loss_tab_train, loss_tab_test = train(device, model_ft, optimizer, num_epochs, train_loader, test_loader, weight, model_path, Version_name, ModelName)
  
    print('DONE')
    
    #%% Save Model 
    #save model
    torch.save(model_ft.state_dict(), model_path + os.sep + Version_name + os.sep + 'model_state' + os.sep + Version_name + '_model.pt')
    # save model as script
    model_ft_scripted = torch.jit.script(model_ft) # Export to TorchScript
    model_ft_scripted.save(model_path + os.sep + Version_name + os.sep + 'model_state' + os.sep + Version_name + '_Scripted_model.pt') # Save
    
    parameters['Task_ID'] = Task_ID
    parameters['BM_Name'] = BM_Name
    parameters['SplitName'] = SplitName
    parameters['Version_name'] = Version_name
    parameters['ModelName'] = ModelName
    list_keys = list(parameters.keys())
    np.savez(model_path + os.sep + Version_name + os.sep + 'hyper_parameters' + os.sep + Version_name+ '_hyperparameters.npz', parameters=parameters)
    #save metadata as .npz and in a txt file
    
    f= open(model_path + os.sep + Version_name + os.sep + 'hyper_parameters' + os.sep + Version_name+ '_hyperparameters.txt',"w+")
    for i in range(len(parameters)):
         f.write(list_keys[i] + '\t' + str(parameters[list_keys[i]])+'\n')
    f.close()
    #%% PLOT LOSSES
    print(' ')
    print('Train and Test losses over epochs')
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(loss_tab_train, label='Train')
    plt.plot(loss_tab_test, label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (mean)')
    plt.legend()
    plt.grid()
    plt.savefig(model_path + os.sep + Version_name + os.sep + 'train_curves' + os.sep + Version_name + '_LossCurves.png')
    np.savez(model_path + os.sep + Version_name + os.sep + 'train_curves' + os.sep + Version_name + '_LossCurvesDATA.npz', loss_tab_train=loss_tab_train, loss_tab_test=loss_tab_test)

    return model_ft, test_loader, LabelsList
    #%% Apply model on some examples
    
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
               