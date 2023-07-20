# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 18:24:24 2023

@author: gabri
"""

'''
Include Functions to prepare OSmOSE dataset already annotated with Aplose to be use with torch for train network and make benchmark

Librairies : Please, check the file "requierments.txt"

functions here : 
        - CreatDatasetForTorch_main() : MAIN

Paths needed : 
        - path_osmose_dataset : path of the datset with OSmOSE architecture
        - codes_path : path with all the functions needed 
'''


#%% Import Bibli
import numpy as np
import os
import glob
import pandas as pd
import datetime 
from tqdm import tqdm

#with open('path_codes.txt') as f:
    #2codes_path = f.readlines()[0]
#sys.path.append(codes_path)


#%% MAIN
def FormatDatasets_main(Task_ID, BM_Name, LenghtFile, Fs, dataset_ID, file_annotation, LabelsList, AnnotatorsList, Crop_duration = 3):
    
    '''
        INPUTS :
            - dataset_ID : name of dataset 
            - Task_ID : name of task
            - BM_Name : name of Benchmark
            - LengthFile : duration of input files (in second)
            - Fs : Sampling Rate (in Hz)
            - LabelsList : List of label to keep from Aplose's annotations
            - AnnotatorsList : List of annotator to keep from Aplose's annotations
            - Crop_duration : time to crop at the start and end of the Aplose's annotations
           
    '''
    #Load Paths
    with open('path_osmose.txt') as f:
        path_osmose = f.readlines()[0]
    
    path_osmose_analysisAI = path_osmose + 'analysis' + os.sep + 'AI'
    path_osmose_dataset = path_osmose + 'dataset' 
    folderName_audioFiles = str(LenghtFile)+'_'+str(int(Fs))

    #Create Paths
    if not os.path.exists(path_osmose_analysisAI + os.sep + Task_ID):
        os.makedirs(path_osmose_analysisAI + os.sep + Task_ID)  
    if not os.path.exists(path_osmose_analysisAI + os.sep + Task_ID + os.sep + BM_Name + os.sep + folderName_audioFiles):
        os.makedirs(path_osmose_analysisAI + os.sep + Task_ID + os.sep + BM_Name + os.sep + folderName_audioFiles)
        os.makedirs(path_osmose_analysisAI + os.sep + Task_ID + os.sep + BM_Name + os.sep + folderName_audioFiles + os.sep + 'info_datasplit')
        os.makedirs(path_osmose_analysisAI + os.sep + Task_ID + os.sep + BM_Name + os.sep + folderName_audioFiles + os.sep + 'models')
    
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

        for i_status in tqdm(range(len(xl_task_status))):
            test_keepfile = 0
            for annot in AnnotatorsList:
                if xl_task_status[annot][i_status] == 'FINISHED':
                    test_keepfile = 1
            if test_keepfile == 0:
                filename = xl_task_status['filename'][i_status][:-4]
                for fileinlist in list_wavfile:
                    if filename in fileinlist:
                        list_wavfile.remove(fileinlist)
    
    NbFile = len(list_wavfile)
    print("Done, "+str(NbFile)+' files remains')
    
    #Load timestamp
    timestamp_file = glob.glob(os.path.join(path_audio_files , '*timestamp.csv'))
    xl_timestamp = pd.read_csv(timestamp_file[0], header=None,)  
    
    # Prepare CSV annotation for DL
    columns_name = LabelsList.copy()
    columns_name.insert(0, "filename")
    format_dataset_df = pd.DataFrame(columns=columns_name)
    format_dataset_df["filename"] = [[]] * NbFile
    Crop_duration_datetime = datetime.timedelta(seconds=Crop_duration)
    TAB_AnnotationPerFile = np.zeros([len(list_wavfile), len(LabelsList)]) 
    
    
    print('Matching annotation with timestamp for each label ...')
    # For Loop over label 
    for label_id in range(len(LabelsList)):
        # First Assignation - Annotation
        label = LabelsList[label_id]
        Annot_starttime = np.empty(0)
        Annot_endtime = np.empty(0)
        
        #For Loop over annotations
        for id_annot in range(len(xl_data)):
            # label wanted ?
            if xl_data['annotation'][id_annot] != label:
                continue
            # annotator wanted ?
            if xl_data['annotator'][id_annot] not in AnnotatorsList:
                continue
            
            #Keep Datedtime in and out
            annot_datetime_in =  datetime.datetime.strptime(xl_data['start_datetime'][id_annot]+'+0000', '%Y-%m-%dT%H:%M:%S.%f+00:00%z') + Crop_duration_datetime
            annot_datetime_out = datetime.datetime.strptime(xl_data['end_datetime'][id_annot]+'+0000', '%Y-%m-%dT%H:%M:%S.%f+00:00%z') - Crop_duration_datetime
            Annot_starttime = np.append(Annot_starttime, annot_datetime_in.timestamp())
            Annot_endtime = np.append(Annot_endtime, annot_datetime_out.timestamp())
            
        #Second Assignation - AudioFile
        datetime_length = datetime.timedelta(seconds = float(LenghtFile))   
        File_starttime = np.zeros(len(list_wavfile))
        File_endtime = np.zeros(len(list_wavfile)) 
        
        #For Loop Over audio files
        for id_file in range(len(list_wavfile)):
            
            ind = list(xl_timestamp[0]).index(list_wavfile[id_file])
            #Keep Datedtime in and out
            file_datetime_in = datetime.datetime.strptime(xl_timestamp[1][ind]+'+0000', '%Y-%m-%dT%H:%M:%S.%fZ%z')
            file_datetime_out = file_datetime_in + datetime_length
            File_starttime[id_file] = file_datetime_in.timestamp()
            File_endtime[id_file] = file_datetime_out.timestamp()
            
         
        # For over right annotation to check match   
        AnnotationPerFile = np.zeros(len(list_wavfile))
        for id_annot in tqdm(range(len(Annot_starttime))):
            
            AST = Annot_starttime[id_annot]
            AET = Annot_endtime[id_annot]
          
            #Cond1 : Annotation Start Time in [File Start time : File End Time] ?
            I1 = np.zeros_like(AnnotationPerFile)
            I1[File_starttime < AST] += 1
            I1[AST < File_endtime] += 1
            
            #Cond2 : Annotation End Time in [File Start time : File End Time] ?
            I2 = np.zeros_like(AnnotationPerFile)
            I2[File_starttime < AET] += 1
            I2[AET < File_endtime] += 1
            
            #Cond3 : File between Annotation Start Time and Annotation End Time ?
            I3 = np.zeros_like(AnnotationPerFile)
            I3[AST < File_starttime] += 1
            I3[File_endtime < AET] += 1
            
            #If at leats one on three condition :
            AnnotationPerFile[I1 == 2] = 1
            AnnotationPerFile[I2 == 2] = 1
            AnnotationPerFile[I3 == 2] = 1
            
        # Save In Tab For The Label 
        TAB_AnnotationPerFile[:,label_id] = AnnotationPerFile
        print(LabelsList[label_id]+' OK !')
        
        
    # Reorganise to save in DataFrame
    TAB_FINAL = []
    for i in range(len(list_wavfile)):
        ind = list(xl_timestamp[0]).index(list_wavfile[i])
        line = list([xl_timestamp[0][ind]])
        for j in range(len(LabelsList)):
            line.append(TAB_AnnotationPerFile[i,j])
        TAB_FINAL.append(line)
       
    # Creat DataFrame
    format_dataset_df = pd.DataFrame(TAB_FINAL, columns=columns_name)
    # Save DataFrame as csv
    format_dataset_df.to_csv (path_osmose_analysisAI + os.sep + Task_ID + os.sep + BM_Name + os.sep + folderName_audioFiles + os.sep + 'info_datasplit' + os.sep +  'ALLannotations.csv', index = False, header=True)
    
    # Print % of positif per label 
    print('Nombre de fichier : ', len(format_dataset_df))
    percent_per_label = []
    for label in LabelsList:
        x = 100*np.sum(format_dataset_df[label])/len(format_dataset_df)
        print(label,' -> pourcentage de Positif : ', "{:10.3f}".format(x),'%')
        percent_per_label.append(x)
        
    print('DONE ! ')
    print('Next step : Define Test, Train and Validation for your network !')
        
    # Save Some Metadata
    metadata_tab = [dataset_ID, LenghtFile, Fs, LabelsList, AnnotatorsList, NbFile, percent_per_label]
    metadata_label = ['dataset_ID', 'LenghtFile', 'Fs', 'LabelsList', 'AnnotatorsList', 'NbFile', 'percent_per_label']
    np.savez(path_osmose_analysisAI + os.sep + Task_ID + os.sep + BM_Name + os.sep + folderName_audioFiles + os.sep + 'info_datasplit' + os.sep +  'Annotations_metadata.npz', dataset_ID=dataset_ID, LenghtFile=LenghtFile, Fs=Fs, LabelsList=LabelsList, AnnotatorsList=AnnotatorsList)
    f= open(path_osmose_analysisAI + os.sep + Task_ID + os.sep + BM_Name + os.sep + folderName_audioFiles + os.sep + 'info_datasplit' + os.sep +  'Annotations_metadata.txt',"w+")
    for i in range(len(metadata_label)):
         f.write(str(metadata_label[i]) + '\t' + str(metadata_tab[i])+'\n')
    f.close()


