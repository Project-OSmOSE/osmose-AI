# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 12:57:07 2022

@author: gabri
"""

#%% Import librairies
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import numpy as np
#import torchvision
import torch.nn.functional as F
from torchvision import models, transforms

import matplotlib.pyplot as plt


#%% Model Simple "Homemade"
    
class CNN3_FC3(nn.Module):
    def __init__(self, output_dim, c=64):
        super(CNN3_FC3, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=c, kernel_size=8, stride=4, padding=0), nn.BatchNorm2d(c)) 
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=3, stride=2, padding=0), nn.BatchNorm2d(2*c))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=c*2, out_channels=c*4, kernel_size=3, stride=2, padding=0), nn.BatchNorm2d(4*c))
        self.lin1 = nn.Linear(c*4, 128*output_dim)
        self.lin2 = nn.Linear(output_dim*128, output_dim*16)
        self.lin3 = nn.Linear(output_dim*16, output_dim)
        self.dout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x,_ = torch.max(x, 3, False)
        x,_ = torch.max(x, 2, True)
        x = x.view(x.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x)) 
        x = self.dout(x)
        y = torch.sigmoid(self.lin3(x))
        return y
    
class CNN4_FC3(nn.Module):
    def __init__(self, output_dim, c=64):
        super(CNN4_FC3, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=c, kernel_size=8, stride=4, padding=0), nn.BatchNorm2d(c)) 
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=3, stride=2, padding=0), nn.BatchNorm2d(2*c))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=c*2, out_channels=c*4, kernel_size=3, stride=2, padding=0), nn.BatchNorm2d(4*c))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=c*4, out_channels=c*4, kernel_size=3, stride=2, padding=0), nn.BatchNorm2d(4*c))
        self.lin1 = nn.Linear(c*4, 128*output_dim)
        self.lin2 = nn.Linear(output_dim*128, output_dim*16)
        self.lin3 = nn.Linear(output_dim*16, output_dim)
        self.dout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = torch.mean(x, 3, False)
        x = torch.mean(x, 2, True)
        x = x.view(x.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x)) 
        x = self.dout(x)
        y = torch.sigmoid(self.lin3(x))
        return y
    
    
class CNN3_FC1(nn.Module):
    def __init__(self, output_dim, c=64):
        super(CNN3_FC1, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=c, kernel_size=8, stride=4, padding=0), nn.BatchNorm2d(c)) 
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=3, stride=2, padding=0), nn.BatchNorm2d(2*c))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=c*2, out_channels=c*4, kernel_size=3, stride=2, padding=0), nn.BatchNorm2d(4*c))
        self.lin1 = nn.Linear(c*4, output_dim)
        self.dout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x,_ = torch.max(x, 3, False)
        x,_ = torch.max(x, 2, True)
        x = x.view(x.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        x = self.dout(x)
        x = torch.sigmoid(self.lin1(x))
        return x


def transform_simple(spectro, parameters):
    output = 'torch'
    dynamic_min = parameters['dynamic_min'][parameters['index_dataset']]
    dynamic_max = parameters['dynamic_max'][parameters['index_dataset']]
    
    #Normalize Lvl
    spectro = 10*np.log10(spectro+1e-15)
    spectro[spectro < dynamic_min] = dynamic_min
    spectro[spectro >  dynamic_max] = dynamic_max
    spectro = (spectro - dynamic_min) /  (dynamic_max-dynamic_min)
    
    #plt.pcolor(spectro)
    #plt.savefig('test'+str(np.random.randint(10))+'.png')
    
    X, Y = np.shape(spectro)
    if output == 'np':
        return spectro
    if output == 'torch':
        spectro = torch.from_numpy(spectro.reshape((1 ,X, Y)))
        return spectro
        #return torch.from_numpy(spectro.reshape((3 ,X, Y)))
    else : print('Wrong Output')
    
#%% Model for transfer learning       
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Sigmoid())
        #model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        
    elif model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Sigmoid())
        #model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        
    elif model_name == "resnet101":
        """ Resnet101
        """
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Sigmoid())
        #model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        #model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        model_ft.classifier[6] = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Sigmoid())
        input_size = 224

    elif model_name == "vgg11_bn":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        #model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        model_ft.classifier[6] = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Sigmoid())
        input_size = 224
        
    elif model_name == "vgg11":
        """ VGG11
        """
        model_ft = models.vgg11(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        #model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        model_ft.classifier[6] = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Sigmoid())
        input_size = 224
        
    elif model_name == "vgg13_bn":
        """ VGG13_bn
        """
        model_ft = models.vgg13_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        #model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        model_ft.classifier[6] = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Sigmoid())
        input_size = 224
        
    elif model_name == "vgg13":
        """ VGG13
        """
        model_ft = models.vgg13(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        #model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        model_ft.classifier[6] = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Sigmoid())
        input_size = 224
        
    elif model_name == "vgg19_bn":
        """ VGG19_bn
        """
        model_ft = models.vgg19_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        #model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        model_ft.classifier[6] = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Sigmoid())
        input_size = 224
        
    elif model_name == "vgg19":
        """ VGG19
        """
        model_ft = models.vgg19(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        #model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        model_ft.classifier[6] = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Sigmoid())
        input_size = 224

    else:
        print("Warning : Invalid model name.")
        exit()

    return model_ft, input_size

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def transform_ref(spectro, parameters):
    output = 'torch'
    dynamic_min = parameters['dynamic_min'][parameters['index_dataset']]
    dynamic_max = parameters['dynamic_max'][parameters['index_dataset']]
    #To Image
    input_size = 224
    transform_imm = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    ToPict =  transforms.ToPILImage()
    
    #Normalize Lvl
    spectro = 10*np.log10(spectro+1e-15)
    spectro[spectro < dynamic_min] = dynamic_min
    spectro[spectro >  dynamic_max] = dynamic_max
    spectro = (spectro - dynamic_min) /  (dynamic_max-dynamic_min)
    X, Y = np.shape(spectro)
    
    #Output
    if output == 'np':
        return spectro
    if output == 'torch':
        spectro = torch.from_numpy(spectro).repeat(3,1,1)
        imm = ToPict(spectro)
        spectro = transform_imm(imm)
        return spectro
        #return torch.from_numpy(spectro.reshape((3 ,X, Y)))
    else : print('Wrong Output')
    
#%% Loss BCE

def bce_loss(prediction, truth, weight):
    recon_loss = F.binary_cross_entropy(prediction, truth, reduction='none')
    recon_loss *= weight
    return recon_loss.mean()    