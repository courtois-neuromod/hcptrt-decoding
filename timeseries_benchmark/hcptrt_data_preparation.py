import numpy as np
import pandas as pd
# import os
import glob
# import h5py
# import matplotlib.pyplot as plt
from load_confounds import Params9, Params24
from nilearn.input_data import NiftiLabelsMasker
from sklearn import preprocessing
from numpy import savetxt

import utils
# from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
# from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, cross_val_score, train_test_split
# from sklearn.svm import SVC

# from keras.models import Sequential
# from keras.layers import Dense
# from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
# from nilearn.plotting import plot_anat, show, plot_stat_map, plot_matrix
# from sklearn.neural_network import MLPClassifier


#################### Global variables ####################

TR = 1.49
out_path = '/home/SRastegarnia/hcptrt_decoding_Shima/hcptrt_decoding/timeseries_benchmark/outputs/'
bold_suffix = '_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
pathdata = '/data/neuromod/DATA/cneuromod/hcptrt/derivatives/fmriprep-20.2lts/fmriprep/'
pathevents = '/data/neuromod/DATA/cneuromod/hcptrt/'


#################### Data preparation ####################


def load_fmri_data(subject, modality, confounds):    
    
    """   
    Parameters
    ----------
    subject: str 
        e.g. 'sub-01'
    modality: str
        e.g. 'motor'
    confounds: fMRI confounds generating strategy
        e.g. Params9()
    """
    data_path = sorted(glob.glob(pathdata + '{}/**/*{}*'
                     .format(subject, modality) + bold_suffix, recursive=True))
    
    print('The number of bold files:', len(data_path))

    bold_files = []
    for dpath in data_path:    
        masker = NiftiLabelsMasker(labels_img = 'MIST_444.nii.gz', standardize=True, 
                                  detrend = False, smoothing_fwhm = 5).fit()
        data_fmri = masker.transform(dpath, confounds = confounds.load(dpath))    
        bold_files.append(data_fmri)

    bold_outname = out_path + subject + '_' + modality + '_fMRI2.npy'
    np.save(bold_outname, bold_files)
    
    a = np.load(bold_outname, allow_pickle=True)
    bold_files = a
    
    print('### Reading Nifiti files is done!')
    print('-------------------------------------------------')
    
    return bold_files, masker, data_path
   
    
def load_events_files(subject, modality):
    
    """   
    Parameters
    ----------
    events_path: list
        path to events files
    modality: str
        e.g. 'motor'
    """   
    events_path = sorted(glob.glob(pathevents + '{}/**/func/*{}*_events.tsv'
                                   .format(subject, modality), recursive=True))
    
    print('The number of events files:', len(events_path))
    
    events_files = []
    for epath in events_path: 
        event = pd.read_csv(epath, sep = "\t", encoding = "utf8", header = 0)
        
        if modality == 'wm':                    
            event.trial_type = event.trial_type.astype(str) + '_' + \
            event.stim_type.astype(str)

        if modality == 'relational':                    
            event.trial_type = event.trial_type.astype(str) + '_' + \
            event.instruction.astype(str)

        events_files.append(event)
    
    print('### Reading events files is done!')
    print('-----------------------------------------------')
    
    return events_files


#################### Checking data ####################

def check_input(bold_files, events_files):
            
    """   
    Parameters
    ----------
    bold_files: list
        output of load_fmri_data function
    events_files: list
        output of load_events_files function
    """
    
    data_lenght = len(bold_files)
    data_lenght = int (data_lenght or 0)

    for i in range(0, data_lenght-1):
        if bold_files[i].shape > bold_files[i+1].shape:         
            a = np.shape(bold_files[i])[0] - np.shape(bold_files[i+1])[0]        
            bold_files[i] = bold_files[i][0:-a, 0:]
            print('The bold file number', i, 'had', a, 'extra volumes')

    if len(events_files) != len(bold_files):
        print('Miss-matching between events and fmri files')
        print('Number of Nifti files:' ,len(bold_files))
        print('Number of events files:' ,len(events_files))
    else:
        print('Events and fMRI files are Consistent.')

    for d in range(0, data_lenght-1):
        if bold_files[d].shape != bold_files[d].shape:
            print('There is mismatch in BOLD file size!')
            
    print('### Cheking data is done!')
    print('-----------------------------------------------')


#################### Labeling ####################

def volume_labeling(bold_files, events_files, confounds, subject, modality, masker, data_path):

    """
    Generating labels files for each volumes using 
    'trial_type' column of events files.
    
    Parameters
    ----------
    events_files: list
        output of load_events_files function
    """
    
    data_lenght = len(bold_files)
    labels_files = []
    for events_file in events_files:
        task_durations = []
        task_modalities = []
        row_counter = 0

        task_modalities.append(events_file.iloc[0]['trial_type'])
        rows_no = len(events_file.axes[0])

        for i in range(1, rows_no):
            if (events_file.iloc[i]['trial_type'] != events_file.iloc[i-1]['trial_type']):
                task_modalities.append(events_file.iloc[i]['trial_type'])
                duration = (events_file.iloc[i]['onset']) - \
                            (events_file.iloc[row_counter]['onset'])
                task_durations.append(duration)
                row_counter = i

        task_durations.append(events_file.iloc[i]['duration'])

        if (len(task_durations) != len(task_modalities)):
            print('error: tasks and durations do not match')

        task_durations = np.array(task_durations)
        task_modalities = np.array(task_modalities)

        # Generate volume No. array for each task condition
        volume_no = []
        for t in task_durations:
            volume_round = np.round((t)/TR).astype(int)
            volume_no.append(volume_round)

        # Find the Qty of null ending volumes
        ans_round = utils.sum_(volume_no)
#         sample_data = pathdata + subject + '/ses-001/func/' + subject + \
#                        '_ses-001_task-' + modality + '_run-1' + bold_suffix
               
        sample_fmri = masker.transform(data_path[0], confounds = confounds.
                                     load(data_path[0]))        
        null_ending = sample_fmri.shape[0] - ans_round 

        # Generate timeseries labels considering the volume No.           
        final_array = []
        if (len(task_modalities) == len(task_durations) == len(volume_no)):
            for l in range (len(task_modalities)):
                f = ((task_modalities[l],) * volume_no[l])
                final_array.append(f)

        # Add the null label for the ending volumes
        if null_ending > 0:
            end_volume = (('null',) * null_ending)
            final_array.append(end_volume)

        # Generate a flat list of labels
        flat_list = [item for sublist in final_array for item in sublist]
        volume_labels = np.array(flat_list)
        labels_files.append(volume_labels)

    # Generate a flat list of labels
    flat_labels_files = [item for sublist in labels_files for item in sublist]
    flat_volume_labels = np.array(flat_labels_files)

    shape = np.shape(bold_files[1])[0]
    flat_volume_labels = np.reshape(flat_volume_labels, (data_lenght * shape, 1))

    # Generate a flat list of bold matrices
    flat_bold = [item for sublist in bold_files for item in sublist]
    flat_bold_files = np.array(flat_bold)  

    # Cheking the same lenght of the flat bold and label file
    if (len(flat_bold_files[:, 0]) != len(flat_volume_labels[:, 0])):
        print('error: labels and bold flat files mismatche')

    print('### Concatenating fMRI & events files is done!')
    print('-----------------------------------------------')
        
    return flat_bold_files, flat_volume_labels

'--------------------------------------------------------------------------------'

def HRFlag_labeling(flat_volume_labels):
    
    HRFlag_volume_labels = []
    b = 0
    l = len(flat_volume_labels[:, 0]) 

    while (b < (l- 1)):  
        if (flat_volume_labels[b, 0] != flat_volume_labels[b + 1, 0]):
            HRFlag_volume_labels.append(flat_volume_labels[b, 0])

            if (flat_volume_labels[b + 1, 0] == flat_volume_labels[b + 2, 0] == 
                flat_volume_labels[b + 3, 0] == flat_volume_labels[b + 4, 0]):
                for j in range (1, 4):
                    HRFlag_volume_labels.append('HRF_lag')
                b = b + 4  
            else:
                b = b + 1

        else:
            HRFlag_volume_labels.append(flat_volume_labels[b, 0])
            b = b + 1

    HRFlag_volume_labels.append(flat_volume_labels[l - 1, 0])
    
    print('### HRF lag labeling is done!')
    print('-----------------------------------------------')
    
    return HRFlag_volume_labels

'--------------------------------------------------------------------------------'

def unwanted_label_removal(events_files, HRFlag_volume_labels, flat_bold_files, modality):
                                                                    
    
    categories = list(events_files[0].trial_type)
    unwanted = {'countdown','cross_fixation','Cue','new_bloc_right_hand', 
                'new_bloc_right_foot','new_bloc_left_foot','new_bloc_tongue', 
                'new_bloc_left_hand','new_bloc_control','new_bloc_relational',
                'new_bloc_shape','new_bloc_face', 'countdown_nan','Cue_nan', 
                'HRF_lag', 'null'}
    
    categories = [c for c in categories if c not in unwanted]
    conditions = list(set(categories))
    num_cond = len(set(categories))
    
    final_volume_labels = []
    parcel_no = np.shape(flat_bold_files[1])[0]
    final_bold_files = np.empty((0, parcel_no), int)

    for i in range (0, len(HRFlag_volume_labels)):
        if (HRFlag_volume_labels[i] not in unwanted):
            final_volume_labels.append(HRFlag_volume_labels[i])
            final_bold_files = np.append(final_bold_files, 
                                         np.array([flat_bold_files[i, :]]), axis=0)
    
    final_label_path = out_path + modality +'_final_labels.csv'
    df_lable = pd.DataFrame(final_volume_labels)
    df_lable.to_csv(final_label_path, sep=',' ,index=False, header=None)

    final_fMRI_path = out_path + modality + '_final_fMRI.npy'
    np.save(final_fMRI_path, final_bold_files) 
    
    print('### Final volume label file is generated from events files!')
    print('File path:', final_label_path)
    print('-----------------------------------------------')
    print('### Final numpy.ndarray file is generated from fMRI files!')
    print('File path:', final_fMRI_path)

    return final_volume_labels, final_bold_files




