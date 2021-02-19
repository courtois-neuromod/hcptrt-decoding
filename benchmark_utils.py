#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from pathlib import Path, PurePath
from nilearn import image, plotting
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker
from nilearn.glm.first_level import FirstLevelModel
from load_confounds import Params9
from sklearn.model_selection import KFold, LeaveOneGroupOut,train_test_split 
from nilearn.decoding import Decoder


######## Return string between methods(used to extract seesions and runs No.) ########
def _between(value, a, b):
    # Find and validate before-part.
    pos_a = value.find(a)
    if pos_a == -1: return ""
    # Find and validate after part.
    pos_b = value.find(b)
    if pos_b == -1: return ""
    # Return middle part.
    adjusted_pos_a = pos_a + len(a)
    if adjusted_pos_a >= pos_b: return ""
    return value[adjusted_pos_a:pos_b]


######## Generate Beta_maps ########
def _generate_beta_maps(scans, confounds, events, conditions, mask, fname, task_label):
    """
    Parameters
    ----------
    scans: list
        A list of Niimg-like objects or file paths for the
        appropriate BOLD files collected during the task
    confounds: list
        Any confounds to correct for in the cleaned data set
    events: list
        A list of pd.DataFrames or file paths for BIDS-formatted
        events.tsv files
    conditions: list
        A list of conditions for which to generate beta maps.
        Must correspond to trial_types in provided events
    mask: str
        The mask within which to process data
    fname: str
        The filename with which to save the resulting maps.
    task_label: string
        The name of the task for which to generate beta maps.
        e.g., 'motor'
    """

    if len(scans) != len(events):
        err_msg = ("Number of event files and BOLD files does not match.")
        raise ValueError(err_msg)

    glm = FirstLevelModel(mask_img=mask, t_r=1.49, high_pass=0.01, smoothing_fwhm=5, standardize=True)

    z_maps, condition_idx, session_idx = [], [], []
    for scan, event, confound in zip(scans, events, confounds):
        
        ses = scan.split('_task')[0].split('fmriprep-20.1.0/')[1].partition('_')[2]
        temp_run = scan.split('run')[1]
        run = _between(temp_run, "-", "_")
        session = ses + '_run-' + run 

        glm.fit(run_imgs=scan, events=event, confounds=confound)

        for condition in conditions:
            z_maps.append(glm.compute_contrast(condition))
            condition_idx.append(condition)
            session_idx.append(session)

    sid = fname.split('_')[0]  # safe since we set the filename
    nib.save(image.concat_imgs(z_maps), fname)
    np.savetxt('{}_{}_new_labels.csv'.format(sid,task_label), condition_idx, fmt='%s')
    np.savetxt('{}_{}_new_runs.csv'.format(sid,task_label), session_idx, fmt='%s')
    

######## Extract intended conditions for each task ######## 
def conditions(event_file):
    
    df = pd.read_table(event_file)
#     category = df.trial_type.str.split('_', n=1, expand=True)[1]
    categories = list(df.trial_type)
    unwanted = {'countdown', 'cross_fixation', 'Cue', 'new_bloc_right_hand', 
                'new_bloc_right_foot','new_bloc_left_foot','new_bloc_tongue', 
                'new_bloc_left_hand', 'new_bloc_control', 'new_bloc_relational',
                'new_bloc_shape', 'new_bloc_face' 
                }
    categories = [c for c in categories if c not in unwanted]
    conditions = list(set(categories))
    return(conditions)

    
######## External function for running all the generating Beta-map process ########
def postproc_task(subject, task_label, conditions, tpl_mask):
    """
    Parameters
    ----------
    subject: str
        The full subject identifier, e.g. 'sub-01'
    task_label: str
        The task label used in naming the files, dropping the 'task-' key,
        e.g., 'wm'
    conditions: list
        The trial_types in events file whic is the output of 
        return_conditions function
    tpl_mask: str
        The local file path to the grey matter mask derived from
        the template
    """

    root = '/home/SRastegarnia/hcptrt_decoding_Shima/DATA/cneuromod/hcptrt/sub-01/fmriprep-20.1.0/'
    func = 'space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
    regr = 'desc-confounds_regressors.tsv'

    scans = sorted(Path(root).rglob('*_task-{}*{}'.format(task_label, func)))
    scans = [str(s) for s in scans]

    regressors = sorted(Path(root).rglob('*_task-{}*{}'.format(task_label, regr)))
    confounds = [pd.DataFrame.from_records(Params9().load(str(r))) for r in regressors]

    events = sorted(Path(root).rglob('*_task-{}*events.tsv'.format(task_label)))
    events = [pd.read_table(e) for e in events]

    _generate_beta_maps(
        scans=scans, confounds=confounds, events=events, conditions=conditions, mask=tpl_mask,
        fname='{}_task-{}_{}'.format(subject, task_label, func.replace('preproc_bold', 'postproc_new_P9')),
        task_label=task_label)
    

######## Decoder function ########    
def check_decoding(task_dir, task_label, tpl_mask):   
    """
    Parameters
    ----------
    task_dir: str
        File path to outputs from `postproc_task`
    task_label: str
        The task label used in naming the files, dropping the 'task-' key,
        e.g., 'wm'
    tpl_mask: str
        The local file path to the grey matter mask derived from
        the template
    """

    z_maps = sorted(Path(task_dir).rglob('*_task-{}*-postproc_new_P9.nii.gz'.format(task_label)))# SHIMA
    z_maps = [str(z) for z in z_maps]# SHIMA    
    conditions = Path(task_dir).rglob('*_{}_new_labels.csv'.format(task_label))
    sessions = Path(task_dir).rglob('*_{}_new_runs.csv'.format(task_label))
    
    for z_map, condition, session in zip(z_maps, conditions, sessions):
        condition_idx = pd.read_table(condition, header=None).values.ravel()
        session_idx = pd.read_table(session, header=None).values.ravel()
        # instantiate the relevant objects
        cv = KFold(n_splits=5, random_state=None, shuffle=False)
        decoder = Decoder(estimator='svc', mask=tpl_mask,
                                   standardize=False, cv=cv,
                                   scoring='accuracy')
        decoder.fit(z_map, condition_idx, groups=session_idx)
        scores_dict = decoder.cv_scores_
        for key in scores_dict:
            print(key, np.mean(scores_dict[key]))
    # plot weight maps for the last subject to get a sense of contributing vox
    weights_dict = decoder.coef_img_
    for key in weights_dict:
        plotting.plot_stat_map(weights_dict[key],
                               title=f'Weight map for {key}', colorbar=True, 
                               threshold=0.00007, display_mode='ortho')
    plt.show() 

