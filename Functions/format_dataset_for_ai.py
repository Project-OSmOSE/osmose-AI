# -*- coding: utf-8 -*-
"""
Created on Tue May  2 14:10:35 2023

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

#%% MAIN

def MergeTimestampWithAnnotation(LenghtFile, sample_rate, dataset_ID, file_annotation, LabelsList, AnnotatorsList, Crop_duration = 3, is_box = None, LabelType = "classic"):
    print('Processing dataset : ' + dataset_ID)
    #Load Paths
    with open('path_osmose.txt') as f:
        path_osmose = f.readlines()[0]
    
    path_osmose_dataset = path_osmose + 'dataset' 
    folderName_audioFiles = str(LenghtFile)+'_'+str(int(sample_rate))
    
    #List All WavFile From Dataset
    base_path = path_osmose_dataset + os.sep + dataset_ID + os.sep
    path_audio_files = os.path.join(path_osmose_dataset, dataset_ID, 'data','audio', folderName_audioFiles )
    list_wavfile = [os.path.basename(x) for x in glob.glob(os.path.join(path_audio_files , '*wav'))]
    
    #Load Annotation files
    xl_data = pd.read_csv(base_path + 'final' + os.sep + 'Annotation_Aplose' + os.sep  + file_annotation)
    
    
    #Check if annotators and labels are in the annotation file (results.csv) 
    available_annotators = list(dict.fromkeys(xl_data['annotator']))
    for annot in AnnotatorsList:
        if annot not in available_annotators:
            print('WARNING : ', annot, ' not in the annotators list of your annotation file !')
    available_labels = list(dict.fromkeys(xl_data['annotation']))
    for lab in LabelsList:
        if lab not in available_labels:
            print('WARNING : ', lab, ' not in the label list of your annotation file !')
            
            
    #Remove unannotated files
    if os.path.exists(base_path + 'final' + os.sep + 'Annotation_Aplose' + os.sep  + file_annotation[:-12]+'_task_status.csv'):
        print("Task Status existing : we're removing all unannotated files ...")
        xl_task_status = pd.read_csv(base_path + 'final' + os.sep + 'Annotation_Aplose' + os.sep  + file_annotation[:-12]+'_task_status.csv')
        xl_status_list_file = list(xl_task_status['filename'])
        list_path_datasets_formats = [x[0] for x in os.walk(os.path.join(path_osmose_dataset, dataset_ID, 'data','audio'))]
        
        for i in range(len(list_path_datasets_formats)):
            if list_path_datasets_formats[i][-5:] != 'audio':
                wav_example = os.path.basename(glob.glob(list_path_datasets_formats[i]+ os.sep + '*.wav')[0])
                if wav_example in xl_status_list_file:
                    break
        
        orig_timestamp = pd.read_csv(list_path_datasets_formats[i] + os.sep + 'timestamp.csv', header=None)
        orig_metadata_csv = pd.read_csv(list_path_datasets_formats[i] + os.sep + 'metadata.csv')
        
        orig_duration_file = orig_metadata_csv['audio_file_dataset_duration'][0]
        orig_start_datetime = []
        orig_end_datetime = []
        for j in range(len(orig_timestamp)):
            for annot in AnnotatorsList:
                if orig_timestamp[0][j] in xl_status_list_file:
                    if xl_task_status[annot][xl_status_list_file.index(orig_timestamp[0][j])] == 'FINISHED':
                        orig_start_datetime.append(datetime.datetime.strptime(orig_timestamp[1][j]+'+0000', '%Y-%m-%dT%H:%M:%S.%fZ%z').timestamp())
                        orig_end_datetime.append(orig_start_datetime[-1] + orig_duration_file)
                        break
            
            
        audio_timestamp = pd.read_csv(path_audio_files + os.sep + 'timestamp.csv', header=None)
        list_file_ts = list(audio_timestamp[0])
        
        array_to_delete = np.zeros([len(audio_timestamp)])
        for j in range(len(array_to_delete)):
            array_to_delete[j] = datetime.datetime.strptime(audio_timestamp[1][j]+'+0000', '%Y-%m-%dT%H:%M:%S.%fZ%z').timestamp()

        
        list_wavfile_to_be_kept = []
        for file in tqdm(list_wavfile, desc='Check over all audio file : '):
            wav_dt_in = datetime.datetime.strptime(audio_timestamp[1][list_file_ts.index(file)]+'+0000', '%Y-%m-%dT%H:%M:%S.%fZ%z').timestamp()
            wav_dt_out = wav_dt_in + LenghtFile
            
            #cond1
            tab_test = np.zeros(len(orig_end_datetime))
            tab_test[np.array(orig_start_datetime) <= wav_dt_in] += 1
            tab_test[np.array(orig_end_datetime) >= wav_dt_in] += 1
            
            if 2 in tab_test: 
                list_wavfile_to_be_kept.append(file)
                continue
            
            #cond2
            tab_test = np.zeros(len(orig_end_datetime))
            tab_test[np.array(orig_start_datetime) <= wav_dt_out] += 1
            tab_test[np.array(orig_end_datetime) >= wav_dt_out] += 1
            
            if 2 in tab_test: 
                list_wavfile_to_be_kept.append(file)
                continue
        
        '''
        list_wavfile_to_be_kept = []
        for file in tqdm(list_wavfile):
            flag = False
            for i_status in range(len(xl_task_status)):
                for annot in AnnotatorsList:
                    if xl_task_status['filename'][i_status][:-4] in file:
                        if xl_task_status[annot][i_status] == 'FINISHED':
                            list_wavfile_to_be_kept.append(file) 
                            flag = True
                            break
                    if flag == True: break
         ''' 
        list_wavfile = list_wavfile_to_be_kept
        
    else: print('Warning : No task_status.csv file')
    
    NbFile = len(list_wavfile)
    print("Done, "+str(NbFile)+' files remains')
    
    #Load timestamp
    timestamp_file = glob.glob(os.path.join(path_audio_files , '*timestamp.csv'))
    xl_timestamp = pd.read_csv(timestamp_file[0], header=None,)  
    
    # Prepare CSV annotation for DL
    columns_name = LabelsList.copy()
    columns_name.insert(0, "filename")
    columns_name.insert(0, "format")
    columns_name.insert(0, "dataset")

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
        print('Label : ', label , ' ...')
        for id_annot in tqdm(range(len(xl_data)), desc='Checking annotation datetime : '):
            # label wanted ?
            if xl_data['annotation'][id_annot] != label:
                continue
            # annotator wanted ?
            if xl_data['annotator'][id_annot] not in AnnotatorsList:
                continue
            if is_box != None:
                if is_box == 1:
                    if xl_data['is_box'][id_annot] == 0:
                        continue
                if is_box == 0:
                    if xl_data['is_box'][id_annot] == 1:
                        continue
            
            #Keep Datedtime in and out
            #annot_datetime_in =  datetime.datetime.strptime(xl_data['start_datetime'][id_annot]+'+0000', '%Y-%m-%dT%H:%M:%S.%f+00:00%z') + Crop_duration_datetime
            annot_datetime_in =  datetime.datetime.strptime(xl_data['start_datetime'][id_annot][:-6]+'+0000', '%Y-%m-%dT%H:%M:%S.%f%z') + Crop_duration_datetime
            annot_datetime_out = datetime.datetime.strptime(xl_data['end_datetime'][id_annot][:-6]+'+0000', '%Y-%m-%dT%H:%M:%S.%f%z') - Crop_duration_datetime
            
            if annot_datetime_in.timestamp() < annot_datetime_out.timestamp():
                Annot_starttime = np.append(Annot_starttime, annot_datetime_in.timestamp())
                Annot_endtime = np.append(Annot_endtime, annot_datetime_out.timestamp())
            
        #Second Assignation - AudioFile
        datetime_length = datetime.timedelta(seconds = float(LenghtFile))   
        File_starttime = np.zeros(len(list_wavfile))
        File_endtime = np.zeros(len(list_wavfile)) 
        
        #For Loop Over audio files
        for id_file in tqdm(range(len(list_wavfile)), desc='Checking audio file datetime : '):
            
            ind = list(xl_timestamp[0]).index(list_wavfile[id_file])
            #Keep Datedtime in and out
            file_datetime_in = datetime.datetime.strptime(xl_timestamp[1][ind]+'+0000', '%Y-%m-%dT%H:%M:%S.%fZ%z')
            file_datetime_out = file_datetime_in + datetime_length
            File_starttime[id_file] = file_datetime_in.timestamp()
            File_endtime[id_file] = file_datetime_out.timestamp()
            
         
        # For over right annotation to check match   
        AnnotationPerFile = np.zeros(len(list_wavfile))
        for id_annot in tqdm(range(len(Annot_starttime)), desc='Matching overlap : '):
            
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
            I3[AST <= File_starttime] += 1
            I3[File_endtime <= AET] += 1
            
            #If at leats one on three condition :
            if LabelType == 'classic':
                AnnotationPerFile[I1 >= 2] = 1
                AnnotationPerFile[I2 >= 2] = 1
                AnnotationPerFile[I3 >= 2] = 1
            elif LabelType == 'weak_labels':
                AnnotationPerFile_i = np.zeros_like(AnnotationPerFile)
                AnnotationPerFile_i[I1 >= 2] = 1
                AnnotationPerFile_i[I2 >= 2] = 1
                AnnotationPerFile_i[I3 >= 2] = 1
                AnnotationPerFile += AnnotationPerFile_i
            else : print('!! ERROR : LabelType Unknown !!')
        # Save In Tab For The Label 
        if LabelType == 'weak_labels':
            print(AnnotationPerFile)
            AnnotationPerFile = AnnotationPerFile/len(AnnotatorsList)
            AnnotationPerFile[AnnotationPerFile>1] = 1
            AnnotationPerFile[AnnotationPerFile<0.25] = 0
            print(AnnotationPerFile)
        
        TAB_AnnotationPerFile[:,label_id] = AnnotationPerFile
        print(LabelsList[label_id]+' OK')
        
        
    # Reorganise to save in DataFrame
    TAB_FINAL = []
    
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
    
    print('Number of files : ', len(format_dataset_df))
    for label_fin in LabelsList:
        x = 100*np.count_nonzero(format_dataset_df[label_fin])/len(format_dataset_df)
        print(label_fin,' -> ratio of positive : ', "{:10.3f}".format(x),'%')
    
    print('   ')
    return format_dataset_df


def FormatDatasets_main(Task_ID, BM_Name, LenghtFile_tab, sample_rate_tab, dataset_ID_tab, file_annotation_tab, orig_LabelsList_tab, FinalLabel_Dic, AnnotatorsList_tab, Crop_duration, is_box=None, LabelType = "classic"):
    
    '''
        INPUTS :
            - dataset_ID : name of dataset 
            - Task_ID : name of task
            - BM_Name : name of Benchmark
            - LengthFile : duration of input files (in second)
            - sample_rate : Sampling Rate (in Hz)
            - LabelsList : List of label to keep from Aplose's annotations
            - AnnotatorsList : List of annotator to keep from Aplose's annotations
            - Crop_duration : time to crop at the start and end of the Aplose's annotations
           
    '''
    #Load Paths
    with open('path_osmose.txt') as f:
        path_osmose = f.readlines()[0]
    
    path_osmose_analysisAI = path_osmose + 'analysis' + os.sep + 'AI'
    path_osmose_dataset = path_osmose + 'dataset' 

    #Create Paths
    if not os.path.exists(path_osmose_analysisAI + os.sep + Task_ID):
        os.makedirs(path_osmose_analysisAI + os.sep + Task_ID)  
    if not os.path.exists(path_osmose_analysisAI + os.sep + Task_ID + os.sep + BM_Name):
        os.makedirs(path_osmose_analysisAI + os.sep + Task_ID + os.sep + BM_Name)
        os.makedirs(path_osmose_analysisAI + os.sep + Task_ID + os.sep + BM_Name + os.sep + 'info_datasplit')
        os.makedirs(path_osmose_analysisAI + os.sep + Task_ID + os.sep + BM_Name + os.sep + 'models')
    
    
    Nb_dataset = len(dataset_ID_tab)
    Dic_all_DF = dict.fromkeys(dataset_ID_tab)
    
    for i in range(Nb_dataset):
        
        #get all parameters from users for the i-th dataset
        LenghtFile = LenghtFile_tab[i]
        sample_rate = sample_rate_tab[i]
        dataset_ID = dataset_ID_tab[i]
        file_annotation = file_annotation_tab[i]
        LabelsList =  orig_LabelsList_tab[i]
        AnnotatorsList = AnnotatorsList_tab[i]
        
        folderName_audioFiles = str(LenghtFile)+'_'+str(int(sample_rate))
        format_dataset_df = MergeTimestampWithAnnotation(LenghtFile, sample_rate, dataset_ID, file_annotation, LabelsList, AnnotatorsList, Crop_duration, is_box)

        Dic_all_DF[dataset_ID] = format_dataset_df 
    
    
    print('Merging All Datasets ...')
    TAB_FINAL = []
    FinalLabelsList = list(FinalLabel_Dic.keys())
    NbFileTOT = 0
    for i in range(Nb_dataset):
        
        LenghtFile = LenghtFile_tab[i]
        sample_rate = sample_rate_tab[i]
        dataset_ID = dataset_ID_tab[i]
        file_annotation = file_annotation_tab[i]
        LabelsList =  orig_LabelsList_tab[i]
        AnnotatorsList = AnnotatorsList_tab[i]
        
        folderName_audioFiles = str(LenghtFile)+'_'+str(int(sample_rate))
        
        dic_dataset = Dic_all_DF[dataset_ID]
        
        NbFileTOT += len(dic_dataset)
        
        for j in tqdm(range(len(dic_dataset))):
            line = [dataset_ID] #dataset name
            line.append(folderName_audioFiles) #format
            line.append(dic_dataset['filename'][j][0]) #filename
            for label_fin in FinalLabelsList:
                annot = 0
                for label in  LabelsList:
                    if label in FinalLabel_Dic[label_fin]:
                        annot += dic_dataset[label][j]
                if annot >= 1: annot = 1 
                line.append((annot)) #add annotation for label
            
            TAB_FINAL.append(line)
            

    
    
    final_columns_names = ['dataset', 'format', 'filename'] + FinalLabelsList
    final_df = pd.DataFrame(columns=final_columns_names)
    final_df["filename"] = [[]] * NbFileTOT
    final_df = pd.DataFrame(TAB_FINAL, columns=final_columns_names)
        
    
    # Save DataFrame as csv
    final_df.to_csv (path_osmose_analysisAI +os.sep + Task_ID + os.sep + BM_Name + os.sep + 'info_datasplit' + os.sep + 'ALLannotations.csv', index = False, header=True)
    
    
    # Print % of positif per label 
    print('Number of files : ', len(final_df))
    percent_per_label = []
    for label_fin in FinalLabelsList:
        x = 100*np.count_nonzero(final_df[label_fin])/len(final_df)
        print(label_fin,' -> ratio of positive : ', "{:10.3f}".format(x),'%')
        percent_per_label.append(x)
        
    # Save Some Metadata
    metadata_tab = [dataset_ID_tab, LenghtFile_tab, sample_rate_tab, orig_LabelsList_tab, AnnotatorsList_tab, NbFileTOT, FinalLabelsList, percent_per_label, FinalLabel_Dic]
    metadata_label = ['dataset_ID_tab', 'LenghtFile_tab', 'sample_rate_tab', 'orig_LabelsList_tab', 'AnnotatorsList_tab', 'NbFileTOT', 'FinalLabelsList', 'percent_per_label', 'FinalLabel_Dic']
    np.savez(path_osmose_analysisAI + os.sep + Task_ID + os.sep + BM_Name + os.sep + 'info_datasplit' + os.sep +  'Fdataset_metadata.npz', dataset_ID_tab=dataset_ID_tab, LenghtFile_tab=LenghtFile_tab, sample_rate_tab=sample_rate_tab, orig_LabelsList_tab=orig_LabelsList_tab, AnnotatorsList_tab=AnnotatorsList_tab, NbFileTOT=NbFileTOT, FinalLabelsList=FinalLabelsList, percent_per_label=percent_per_label, FinalLabel_Dic=FinalLabel_Dic)
    f= open(path_osmose_analysisAI + os.sep + Task_ID + os.sep + BM_Name + os.sep + 'info_datasplit' + os.sep +  'Fdataset_metadata.txt',"w+")
    for i in range(len(metadata_label)):
         f.write(str(metadata_label[i]) + '\t' + str(metadata_tab[i])+'\n')
    f.close()
        
    print('DONE ! ')
    print('Next step : Define DEV [train+val] and EVAL sets for your network !')
        
    