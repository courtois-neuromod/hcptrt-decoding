import numpy as np
import pandas as pd
import pickle
import glob
import os
import sys
import nilearn.datasets
#from load_confounds import Params9, Params24
#from nilearn.input_data import NiftiLabelsMasker, NiftiMasker, NiftiMapsMasker
from nilearn.maskers import NiftiLabelsMasker, NiftiMasker, NiftiMapsMasker
from nilearn.interfaces.fmriprep import load_confounds_strategy
from sklearn import preprocessing
from numpy import savetxt
from termcolor import colored

#sys.path.append(os.path.join(".."))
import utils

"""
Utilities for extracting desired volumes 
and labeling/relabeling data.
The outputs are final post-processed data.
"""

def _reading_fMRI2(subject, modality, fMRI2_out_path, region_approach, resolution):    

    bold_outname = fMRI2_out_path + subject + '_' + modality + '_fMRI2.npy'
    bold_files = np.load(bold_outname, allow_pickle=True)
    
    return bold_files


def _reading_events2(subject, modality, events2_out_path, region_approach, resolution):
    
    events_outname = events2_out_path + subject+ '_' + modality + '_events2'
    pickle_in = open(events_outname, "rb")
    events_files = pd.read_pickle(events_outname)
    
    return events_files
    
    
def _volume_labeling(bold_files, events_files, subject, 
                     modality, masker, data_path, TR): # confounds,

    """
    Generating labels files for each volumes using 
    'trial_type' column of events files.
    
    Parameters
    ----------
    events_files: list
        output of load_events_files function
    """
    
    expected_volumes = np.shape(bold_files[1])[0]

    conf = load_confounds_strategy(data_path[0], denoise_strategy="simple", motion="basic", global_signal="basic") #new nilearn
    sample_fmri = masker.fit_transform(data_path[0], confounds= conf[0])

    data_lenght = len(bold_files)
    
    labels_files = []    
    for events_file in events_files:
        task_durations = []
        task_modalities = []
        row_counter = 0
        rows_no = len(events_file.axes[0])
        
        task_modalities.append(events_file.iloc[0]['trial_type'])
        
        for i in range(1, rows_no):
            if (events_file.iloc[i]['trial_type'] != events_file.iloc[i-1]['trial_type']):
                task_modalities.append(events_file.iloc[i]['trial_type'])
                duration = (events_file.iloc[i]['onset']) - \
                            (events_file.iloc[row_counter]['onset'])
                task_durations.append(duration)
                row_counter = i
                
            if (i == rows_no-1):
                duration = (events_file.iloc[i]['onset']) - \
                (events_file.iloc[row_counter]['onset']) + (events_file.iloc[i]['duration'] + TR)
                task_durations.append(duration)

#         task_durations.append(events_file.iloc[i]['duration'])

        if (len(task_durations) != len(task_modalities)):
            print('error: tasks and durations do not match')

        task_durations = np.array(task_durations)
        task_modalities = np.array(task_modalities)

        # Generate volume No. array for each task condition
        volume_no = []
        for t in task_durations:
            volume_round = np.round((t)/TR).astype(int)
            volume_no.append(volume_round)
            
            # check that the volume labels doesn't exceed the expected_volumes of bolds file                        
            if sum(volume_no) > expected_volumes:
                i = 0.1
                while sum(volume_no) > expected_volumes:
                    volume_no = []                
                    for t in task_durations:
                        volume_round = np.round((t-i)/TR).astype(int)
                        volume_no.append(volume_round)
                    i = i + 0.1

        # Find the Qty of null ending volumes
        ans_round = utils.sum_(volume_no)       
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


    flat_labels_files = [item for sublist in labels_files for item in sublist]
    flat_volume_labels = np.array(flat_labels_files)

    shape = np.shape(bold_files[1])[0]
    flat_volume_labels = np.reshape(flat_volume_labels, ((data_lenght)* shape, 1))# for 'gambling'

    # Generate a flat list of bold matrices
#     for sublist in bold_files:# for 'gambling'
#         del sublist[1]   # for 'gambling'
    flat_bold = [item for sublist in bold_files for item in sublist]
    flat_bold_files = np.array(flat_bold)  

    # Cheking the same lenght of the flat bold and label file
    if (len(flat_bold_files[:, 0]) != len(flat_volume_labels[:, 0])):
        print('error: labels and bold flat files mismatche')

        
    print('bold shape:', np.shape(flat_bold_files))
    print('events shape', np.shape(flat_volume_labels))
    
    print('bold type:', type(flat_bold_files))
    print('events type', type(flat_volume_labels)) 
    print(flat_volume_labels)
    print('### Concatenating fMRI & events files is done!')
    print('-----------------------------------------------')
    
    
    
    return flat_bold_files, flat_volume_labels



def _HRFlag_labeling(flat_volume_labels, HRFlag_process):

    if (HRFlag_process == '3volumes'):    
        """
        Labeling the first 3 volumes of stimulus longer than 5 seconds as HRF_lag
        """

        HRFlag_volume_labels = []
        counter = 0
        lenght = len(flat_volume_labels[:, 0]) 

        while (counter < (lenght - 1)):  
            if (flat_volume_labels[counter, 0] != flat_volume_labels[counter + 1, 0]):
                HRFlag_volume_labels.append(flat_volume_labels[counter, 0])
                
                if (counter < (lenght - 4)): 

                    if (flat_volume_labels[counter + 1, 0] == flat_volume_labels[counter + 2, 0] == 
                        flat_volume_labels[counter + 3, 0] == flat_volume_labels[counter + 4, 0]):
                        for j in range (1, 4):
                            HRFlag_volume_labels.append('HRF_lag')
                        counter = counter + 4  
                    else:
                        counter = counter + 1
                else:
                    # when the last 3 or less volumes should be labeled as HRF
                    for ii in range (counter, (lenght - 1)):
                        print((lenght - 1)-counter)
                        HRFlag_volume_labels.append('HRF_lag')
                    counter = counter + ((lenght - 1)-counter)
                    if (counter == (lenght - 1)):
                        print('Exceptional lenght')
                    
            else:
                HRFlag_volume_labels.append(flat_volume_labels[counter, 0])
                counter = counter + 1

        HRFlag_volume_labels.append(flat_volume_labels[lenght - 1, 0])

    elif (HRFlag_process == '2-1volumes'):    
        """
        Labeling the first volume of stimulus longer than 5 seconds for  
        previous stimulus(overlap) also the second and third as HRF_lag 
        """    

        HRFlag_volume_labels = []
        counter = 0
        lenght = len(flat_volume_labels[:, 0]) 

        while (counter < (lenght - 1)):  
            if (flat_volume_labels[counter, 0] != flat_volume_labels[counter + 1, 0]):
                HRFlag_volume_labels.append(flat_volume_labels[counter, 0])

                if (flat_volume_labels[counter + 1, 0] == flat_volume_labels[counter + 2, 0] == 
                    flat_volume_labels[counter + 3, 0] == flat_volume_labels[counter + 4, 0]):

                    HRFlag_volume_labels.append(flat_volume_labels[counter, 0])
                    for j in range (1, 3):
                        HRFlag_volume_labels.append('HRF_lag')
                    counter = counter + 4  
                else:
                    counter = counter + 1

            else:
                HRFlag_volume_labels.append(flat_volume_labels[counter, 0])
                counter = counter + 1

        HRFlag_volume_labels.append(flat_volume_labels[lenght - 1, 0])

    else:
        """
        Labeling without considering the HRF lag
        """  
        temp = flat_volume_labels.tolist()
        HRFlag_volume_labels = [item for sublist in temp for item in sublist]
        
    print('### HRF lag labeling is done!')
    print('-----------------------------------------------')
    
    return HRFlag_volume_labels


def _unwanted_label_removal(events_files,HRFlag_volume_labels, 
                            flat_bold_files,final_bold_out_path,
                            final_labels_out_path,modality, 
                            subject,region_approach,HRFlag_process):
                                                                        
    categories = list(events_files[0].trial_type)
    unwanted = {'countdown','cross_fixation','Cue','new_bloc_right_hand', 
                'new_bloc_right_foot','new_bloc_left_foot','new_bloc_left_hand', 
                'new_bloc_tongue','new_bloc_control','new_bloc_relational',
                'new_bloc_shape','new_bloc_face','countdown_nan','Cue_nan', 
                'HRF_lag','null','nan_Cue','nan_countdown'}
    
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
                     
    final_label_path = final_labels_out_path + subject + '_' + modality + '_' + HRFlag_process + '_final_labels.csv'
    df_lable = pd.DataFrame(final_volume_labels)
    df_lable.to_csv(final_label_path, sep=',' ,index=False, header=None)

#     final_fMRI_path = proc_data_path + modality + '_volumes_final_fMRI.npy'
    final_fMRI_path = final_bold_out_path + subject + '_' + modality + '_' + HRFlag_process + '_final_fMRI.npy'
    np.save(final_fMRI_path, final_bold_files) 
    
    print('### Final volume label file is generated from events files!')
    print('File path:', final_label_path)
    print('-----------------------------------------------')
    print('### Final numpy.ndarray file is generated from fMRI files!')
    print('File path:', final_fMRI_path)

    return final_volume_labels, final_bold_files

    

def postproc_data_prep(subject, modalities, region_approach, HRFlag_process, resolution): # confounds, 
        
    """ 
    Outputs are 
    """    
    TR = 1.49

#    ##### Elm #####
#    proc_data_path = '/home/SRastegarnia/hcptrt_decoding_Shima/data/'

    ##### CC #####
    proc_data_path = '/home/rastegar/projects/def-pbellec/rastegar/hcptrt_decoding_shima/data/'

    bold_suffix = '_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
    raw_atlas_dir = os.path.join(proc_data_path, "raw_atlas_dir")
    
    #     print(colored(subject, 'red', attrs=['bold']))                
    print(colored('{}, {}, {}, res={}:'.format(subject, region_approach,
                                          HRFlag_process, resolution), attrs=['bold']))  

    if region_approach == 'MIST':
        masker = NiftiLabelsMasker(labels_img = '{}_{}.nii.gz'.format(region_approach,resolution), 
                                   standardize=True, smoothing_fwhm = 5)

    elif region_approach == 'difumo':

#             num_parcels = int(region_approach.split("_", 1)[1])
        atlas = nilearn.datasets.fetch_atlas_difumo(data_dir = raw_atlas_dir, 
                                                    dimension = resolution)
        atlas_filename = atlas['maps']
        atlas_labels = atlas['labels']            
        masker = NiftiMapsMasker(maps_img=atlas['maps'], standardize=True, verbose=5)

    elif region_approach == 'dypac':

        path_dypac = '/data/cisl/pbellec/models'
        file_mask = os.path.join(path_dypac, 
                                 '{}_space-MNI152NLin2009cAsym_label-GM_mask.nii.gz'.format(subject))
        file_dypac = os.path.join(path_dypac, 
                                  '{}_space-MNI152NLin2009cAsym_desc-dypac{}_components.nii.gz'.format(
                                      subject, resolution))

        masker = NiftiMasker(standardize=True, detrend=False, smoothing_fwhm=5, mask_img=file_mask)

    else:
        masker = NiftiMasker(standardize=True)



    fMRI2_out_path = proc_data_path + 'medial_data/fMRI2/{}/{}/{}/'.format(region_approach,
                                                                           resolution, subject)         
    events2_out_path = proc_data_path + 'medial_data/events2/{}/{}/{}/'.format(region_approach,
                                                                               resolution, subject)

    final_bold_out_path = proc_data_path + 'processed_data/proc_fMRI/{}/{}/{}/'.format(region_approach,
                                                                                    resolution, subject)       
    final_labels_out_path = proc_data_path + 'processed_data/proc_events/{}/{}/{}/'.format(region_approach,
                                                                                        resolution, subject)

    if not os.path.exists(final_bold_out_path):
        os.makedirs(final_bold_out_path)

    if not os.path.exists(final_labels_out_path):
        os.makedirs(final_labels_out_path)

    for modality in modalities:
        print(colored(modality, attrs = ['bold']))

#        data_path = sorted(glob.glob('/data/neuromod/DATA/cneuromod/hcptrt/'\
#                                     'derivatives/fmriprep-20.2lts/fmriprep/{}/**/*{}*'
#                                     .format(subject, modality) + bold_suffix, recursive = True)) 

        data_path = sorted(glob.glob('/home/rastegar/scratch/hcptrt/'\
                                     'derivatives/fmriprep-20.2lts/fmriprep/{}/**/*{}*'
                                     .format(subject, modality) + bold_suffix, recursive = True))


        bold_files = _reading_fMRI2(subject, modality, fMRI2_out_path, region_approach, resolution)

        events_files = _reading_events2(subject, modality, events2_out_path, region_approach, resolution)

        flat_bold_files, flat_volume_labels = _volume_labeling(bold_files = bold_files,
                                                               events_files = events_files, 
                                                             #   confounds = confounds, 
                                                               subject = subject, 
                                                               modality = modality, 
                                                               masker = masker, 
                                                               data_path = data_path,
                                                               TR = TR)

        HRFlag_volume_labels = _HRFlag_labeling(flat_volume_labels,HRFlag_process)

        final_volume_labels, final_bold_files = _unwanted_label_removal(events_files, 
                                                                        HRFlag_volume_labels, 
                                                                        flat_bold_files, 
                                                                        final_bold_out_path,
                                                                        final_labels_out_path, 
                                                                        modality, subject,
                                                                        region_approach,
                                                                        HRFlag_process)    
            
###############################################################################################################################

# از این خط به پایین باید در کدها اضافه شود.

def mean_volumes_windows(volume_num,flat_bold_files,flat_volume_labels):
    
    num_rows, num_cols = flat_bold_files.shape 
    
    # Initializing w3_bold_file
    i = 0
    w3_bold_file = np.mean(flat_bold_files[i:i+3, :], axis=0)
    temp_bold = np.reshape(w3_bold_file, (len(w3_bold_file),1))
    w3_bold_file = temp_bold
    w3_bold_file = w3_bold_file.T
    
    for row in range (3, num_rows, 3):
        w3_bold_mean = np.mean(flat_bold_files[row:row+3, :], axis=0)
        w3_bold_file  = np.append(w3_bold_file, [w3_bold_mean], axis=0)
        
    # Initializing w3_events_file
    w3_events_file = flat_volume_labels[0]
    temp_event = np.reshape(w3_events_file, (len(w3_events_file),1))
    w3_events_file = temp_event

    for row in range (3, num_rows, 3):
        if (flat_volume_labels[row] == flat_volume_labels[row+1] == flat_volume_labels[row+2]):
            w3_events_file  = np.append(w3_events_file, 
                                        [flat_volume_labels[row]], axis=0) 
        elif (flat_volume_labels[row] == flat_volume_labels[row+1] != flat_volume_labels[row+2]):
            w3_events_file  = np.append(w3_events_file, 
                                        [flat_volume_labels[row]], axis=0)
        elif (flat_volume_labels[row] != flat_volume_labels[row+1] == flat_volume_labels[row+2]):
            w3_events_file  = np.append(w3_events_file, 
                                        [flat_volume_labels[row+1]], axis=0)
        else:
            w3_events_file  = np.append(w3_events_file, 
                                [flat_volume_labels[row+1]], axis=0)
    w3_events_file = [item for sublist in w3_events_file for item in sublist]
    
    return w3_bold_file, w3_events_file


'--------------------------------------------------------------------------------'

def same_duration_labels(concat_bold_file,concat_events_file):
    
    
    return concat_final_bold, concat_events_bold

'--------------------------------------------------------------------------------'





