# working on it from feb15 2022

import numpy as np
import pandas as pd
# import nibabel as nib
import os
import glob
import csv
import sys
# import time
from termcolor import colored
import matplotlib.pyplot as plt
# from matplotlib.pyplot import *
from pathlib import Path, PurePath
from nilearn import image, plotting
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker
from nilearn.glm.first_level import FirstLevelModel
from load_confounds import Params9
import csv
from csv import reader
# from sklearn.model_selection import KFold,LeaveOneGroupOut,train_test_split,cross_val_score  
# from nilearn.decoding import Decoder
# from IPython.display import Markdown, display
# from sklearn.svm import LinearSVC

    
def postproc_beta_map_check(subject, task_label, region_approach, resolution, HRFlag_process): 
    
    proc_data_path = '/home/SRastegarnia/hcptrt_decoding_Shima/data/'
    raw_data_path = '/data/neuromod/DATA/cneuromod/hcptrt/derivatives/fmriprep-20.2lts/fmriprep/'
    mask_name = 'space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'
    
    print(colored((subject, region_approach, resolution), 'red',attrs=['bold']))
    
    tpl_mask = raw_data_path + '{}/ses-001/func/{}_ses-001_task-{}_run-1_'.format(subject,subject,task_label) + mask_name
    print(tpl_mask)
    
    final_bold_path = proc_data_path + 'processed_data/proc_fMRI/{}/{}/{}/' \
                             '{}_{}_{}_final_fMRI.npy'.format(region_approach, resolution, subject,  
                                                              subject, task_label, HRFlag_process)
    
    print(final_bold_path)
    final_bold = np.load(final_bold_path)
    print('Shape of api_file:', np.shape(final_bold))
    
  
    final_labels_path = proc_data_path + 'processed_data/proc_events/{}/{}/{}/' \
                               '{}_{}_{}_final_labels.csv'.format(region_approach, resolution, subject, 
                                                                  subject, task_label, HRFlag_process)
    
    print(final_labels_path)
    final_labels = pd.read_csv(final_labels_path, encoding = "utf8", header=None)
    print(final_labels)
    print('Number of events:' ,len(final_labels))
    
    ################################################################################################
#     data = pd.read_csv(final_labels_path, header=None)
    with open(final_labels_path, 'r') as read_obj:
        csv_reader = reader(read_obj)
        column1 = []
        for row in csv_reader:
            column1.append(row)
    print(len(column1))
    flat_list = [item for sublist in column1 for item in sublist]
    print(flat_list)
    print(len(flat_list))
#     print("Converting CSV to tab-delimited file...")
#     with open(final_labels_path) as inputFile: 
#         outPath = final_labels_path.split('.csv')[0] + '_tab.tsv'
#         with open(outPath, 'w', newline='') as outputFile:
#             reader = csv.DictReader(inputFile, delimiter=',') 
#             writer = csv.DictWriter(outputFile, reader.fieldnames, delimiter='\t')
# #             writer = csv.writer(outputFile, reader.fieldnames, delimiter='\t')
#             writer.writeheader()
#             writer.writerows(reader)
#     print("Conversion complete.")
    
    ################################################################################################         
    
#     glm = FirstLevelModel(mask_img=tpl_mask, t_r=1.49, high_pass=0.01)
#     glm.fit(run_imgs=final_bold_path, events=outPath)
#     print('glm is fitted')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    