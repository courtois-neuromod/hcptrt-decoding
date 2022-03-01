# working on it from feb15 2022
import numpy as np
import pandas as pd
import nibabel as nib
import os
import csv
import sys
import time
import matplotlib.pyplot as plt
# from matplotlib.pyplot import *
from pathlib import Path, PurePath
from nilearn import image, plotting
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker
from nilearn.glm.first_level import FirstLevelModel
from load_confounds import Params9
from sklearn.model_selection import KFold,LeaveOneGroupOut,train_test_split,cross_val_score  
from nilearn.decoding import Decoder
from IPython.display import Markdown, display
from sklearn.svm import LinearSVC

# from os.path import join as opj
# from collections import namedtuple

# from numpy import nan

sys.path.append(os.path.join("../"))
import utils
   
    

def _new_conditions(event, task_label):
    
    """
    In some HCPtrt tasks, the trial types are relabeled, in order to have 
    more clear labels for decoding. (HCPtrt labels are similar to HCP dataset)
     
    Parameters
    ----------
    event: str
        File-path to a BIDS-formatted events.tsv file in the CNeuroMod
        project
     
    """
    df = pd.read_table(event)
    
    if task_label == 'emotion': 
        df.trial_type = df['trial_type'].replace(['response_face',
                                                  'response_shape'],
                                                 ['fear','shape'])
    

    elif task_label == 'wm':
#         print(df.stim_type)
        df.trial_type = df.stim_type.astype(str) + '_' + df.trial_type.astype(str)
        df.trial_type = df['trial_type'].replace(['Body_0-Back','Body_2-Back',
                                                  'Face_0-Back','Face_2-Back',
                                                  'Place_0-Back','Place_2-Back',
                                                  'Tools_0-Back','Tools_2-Back',
                                                  'nan_Cue','nan_countdown'],
                                                 ['body0b','body2b','face0b',
                                                  'face2b','place0b','place2b',
                                                  'tool0b','tool2b','Cue','countdown'])
                        
    elif task_label == 'language':
        df.trial_type = df['trial_type'].replace(['presentation_story',
                                                  'question_story',
                                                  'response_story',
                                                  'presentation_math',
                                                  'question_math',
                                                  'response_math'],
                                                 ['story','story','story',
                                                  'math','math','math']) 

    elif task_label == 'motor':             
        df.trial_type = df['trial_type'].replace(['response_left_foot',
                                                  'response_left_hand',
                                                  'response_right_foot',
                                                  'response_right_hand',
                                                  'response_tongue'],
                                                 ['footL','handL','footR',
                                                  'handR','tongue']) 

    elif task_label == 'relational':                   
        df.trial_type = df['trial_type'].replace(['Control','Relational'],
                                                 ['match','relational']) 
    
    pd.set_option('display.max_rows', df.shape[0]+1)
    print('Number of events:' ,len(df))
    print(np.unique(df.trial_type))
    print(df.iloc[:,[0]])

    return df
    

    
    
def _generate_beta_maps(scans, confounds, events, conditions, mask, fname, task_label, out_path):
    
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
        A list of conditions for which to generate beta maps
        (correspond to trial_types in provided events)
    mask: str
        The mask within which to process data.
    fname: str
        The filename with which to save the resulting maps
    task_label: string
        The name of the task for which to generate beta maps
        e.g., 'motor'
    out_path: string
        Path for saving the post process files.
    """

    if len(scans) != len(events):
        err_msg = ("Number of event files and BOLD files does not match.")
        raise ValueError(err_msg)

    glm = FirstLevelModel(mask_img=mask[0], t_r=1.49, high_pass=0.01, 
                          smoothing_fwhm=5, standardize=True)

    z_maps, condition_idx, session_idx = [], [], []
    i = 1
    for scan, event, confound in zip(scans, events, confounds):
        print(i)
        i = i + 1
        
        ses = scan.split('_task')[0].split('fmriprep-20.2lts/')[1].partition('_')[2]
        temp_run = scan.split('run')[1]
        run = utils.between(temp_run, "-", "_")
        session = ses + '_run-' + run
        print(session)

        glm.fit(run_imgs=scan, events=event, confounds=confound)
        print('glm is fitted')

        for condition in conditions:
            print(condition)
            z_maps.append(glm.compute_contrast(condition))
            print('z_maps append', condition)
            condition_idx.append(condition)
            session_idx.append(session)

    sid = fname.split('_')[0]  # safe since we set the filename
    nib.save(image.concat_imgs(z_maps), out_path + fname)
    print(out_path + fname)
    print('Saving z_maps file is done!', '\n')
    
    np.savetxt(out_path + '{}_{}_add_labels.csv'.format(sid,task_label), condition_idx, fmt='%s')
    print(out_path + '{}_{}_add_labels.csv'.format(sid,task_label))
    print('Saving task labels file is done!', '\n')
    
    np.savetxt(out_path + '{}_{}_add_runs.csv'.format(sid,task_label), session_idx, fmt='%s')
    print(out_path + '{}_{}_add_runs.csv'.format(sid,task_label))
    print('Saving runs file is done!')
    print('----------------------------------------------------------------', '\n')


    
    
def _beta_maps_svm_decoder(out_path, fname, task_label, mask):  
    
    """
    SVC decoding model using decoder object
    """
    sid = fname.split('_')[0] 
    z_map =  out_path + fname
    condition = out_path + '{}_{}_add_labels.csv'.format(sid,task_label)
    session = out_path + '{}_{}_add_runs.csv'.format(sid,task_label)
    
    condition_idx = pd.read_table(condition, header=None).values.ravel()
    session_idx = pd.read_table(session, header=None).values.ravel()

    cv = KFold(n_splits=5, random_state=None, shuffle=True)

    decoder = Decoder(estimator='svc', mask=mask[0],
                      standardize=False, cv=cv, scoring='accuracy')

    decoder.fit(z_map, condition_idx, groups=session_idx)

    cv_sco = []
    scores_dict = decoder.cv_scores_
    for key in scores_dict:
        cv_sco.append(np.mean(scores_dict[key]))
        
        print(key, round(np.mean(scores_dict[key]), 2))            
        print('mean value:', round(np.mean(cv_sco), 2), '\n')
        
    # plot weight maps to get a sense of contributing vox
    weights_dict = decoder.coef_img_    
    for key in weights_dict:
        
#         plotting.plot_glass_brain(weights_dict[key], title=f'Weight map for {key}', 
#                        colorbar=True,threshold=0.00008, cmap = 'magma',
#                        plot_abs=False, display_mode='lyrz')
        
        plotting.plot_stat_map(weights_dict[key],
                           title=f'{key}', colorbar=True, threshold=0.00005, 
                               display_mode='ortho', black_bg = 'True')
        
    plt.show() 
        

        
def postproc_task(subject, task_label):
    
    """
    External function for getting all the data ready for generating beta maps.
    
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
        
    """

    raw_data_path = '/data/neuromod/DATA/cneuromod/hcptrt/derivatives/fmriprep-20.2lts/fmriprep/'
    func = 'space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
    regr = 'desc-confounds_timeseries.tsv'
    mask_name = 'space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'
    pathevents = '/data/neuromod/projects/ml_models_tutorial/data/hcptrt/HCPtrt_events_DATA/'
    out_path = '/data/neuromod/projects/ml_models_tutorial/data/hcptrt/postproc_beta_maps/'
    
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    
    scans = sorted(Path(raw_data_path,subject).rglob(
            '*_task-{}*{}'.format(task_label, func)))
    scans = [str(s) for s in scans]

    regressors = sorted(Path(raw_data_path,subject).rglob(
            '*_task-{}*{}'.format(task_label, regr)))
    confounds = [pd.DataFrame.from_records(Params9().load(str(r))) for r in regressors]

    tpl_mask = sorted(Path(raw_data_path,subject).rglob(
            '*_task-{}*{}'.format(task_label, mask_name)))
    tpl_mask = [str(s) for s in tpl_mask]
    
    events = sorted(Path(pathevents,subject).rglob(
            '*_task-{}*events.tsv'.format(task_label)))
    events = [_new_conditions(e, task_label) for e in events]
    
    conditions = list(set(events[0].trial_type))
    print('conditions:', conditions)

    # we know that sub-03 has only 13 sessions of the WM task
    # so we'll subset all subjects to the first 13 runs for balanced data
    # availability. This can be removed later once additional data is collected    
    scans = scans[:13]
    confounds = confounds[:13]
    events = events[:13] 
    
    fname='{}_task-{}_{}'.format(subject, task_label, func.replace('preproc_bold',
                                                                  'postproc_beta_maps_P9'))

    _generate_beta_maps(scans=scans, confounds=confounds, events=events, 
                        conditions=conditions, mask=tpl_mask,fname=fname,
                        task_label=task_label, out_path=out_path)
    
    _beta_maps_svm_decoder(out_path=out_path, fname=fname, 
                           task_label=task_label, mask=tpl_mask)
    
    
# --------------------------------------------------------------------------------------------------------------------  


def _conditions(event_file):
    
    """
    Remove unwanted conditions from events file.
    """
    
#     df = pd.read_table(event_file)
#     category = df.trial_type.str.split('_', n=1, expand=True)[1]
    categories = list(event_file.trial_type)
    unwanted = {'countdown','cross_fixation','Cue','new_bloc_right_hand', 
                'new_bloc_right_foot','new_bloc_left_foot','new_bloc_tongue', 
                'new_bloc_left_hand','new_bloc_control','new_bloc_relational',
                'new_bloc_shape','new_bloc_face', 'countdown_nan','Cue_nan'
                }
    categories = [c for c in categories if c not in unwanted]
    conditions = list(set(categories))
    return(conditions)

######## SVC decoding model ######## 

def fetch_decoding_data(subject, task_dir, task_label):

#     decoding_subjects = [opj(
#         data_dir, "{}.nii.gz".format(subject)) for subject in subjects]
#     decoding_conditions = [np.hstack(pd.read_csv(
#         opj(data_dir, "{}_labels.csv".format(subject)), header=None).to_numpy()) for subject in subjects]
#     decoding_runs = [np.hstack(pd.read_csv(
#         opj(data_dir, "{}_runs.csv".format(subject)), header=None).to_numpy()) for subject in subjects]
#     return np.asarray(decoding_conditions), np.asarray(decoding_subjects), np.asarray(decoding_runs)

    decoding_subjects = sorted(Path(task_dir).rglob('{}*_task-{}-postproc_P9.nii.gz'.format(subject,task_label))) # SH
    decoding_subjects  = [str(d) for d in decoding_subjects] # SH    
    decoding_conditions = Path(task_dir).rglob('{}_{}_labels.csv'.format(subject,task_label))
    decoding_runs = Path(task_dir).rglob('{}_{}_runs.csv'.format(subject,task_label))


# The same just add a NiftiMasker call and svc.LinearSVC call in place of the decoding.Decoder call.
# def within_subject_decoding(subject, root_folder, n_jobs=1, mask, task_dir, out_dir, task_label):
    
def within_subject_decoding(subject, task_dir, task_label, mask):

    decoding_subjects = sorted(Path(task_dir).rglob('{}_task-{}*-postproc_P9.nii.gz'.format(subject,task_label))) # SH
    decoding_subjects = [str(d) for d in decoding_subjects]    
    
#     decoding_subjects = Path(task_dir).rglob('{}_task-{}*-postproc_P9.nii.gz'.format(subject,task_label)) # SH   
    decoding_conditions = Path(task_dir).rglob('{}_{}_labels.csv'.format(subject,task_label))
    decoding_runs = Path(task_dir).rglob('{}_{}_runs.csv'.format(subject,task_label))
    
    masker = NiftiMasker(mask_img=mask).fit()
#     decoding_conditions, decoding_subjects, decoding_sessions = fetch_decoding_data(
#         subject, task_dir, task_label)

    scores = []
    for ims, labs, runs in zip(decoding_subjects, decoding_conditions, decoding_runs):
        labs_idx = pd.read_table(decoding_conditions, header=None).values.ravel()
        runs_idx = pd.read_table(decoding_runs, header=None).values.ravel()
        
        images = np.array(masker.transform(ims))
        decoder = LinearSVC(max_iter=10000)
        cv = KFold(n_splits=5, random_state=None, shuffle=True)
#         cv = LeaveOneGroupOut())
#         score = cross_val_score(
#             decoder, images, np.array(labs), groups=np.array(runs), cv=cv, n_jobs=15)
        score = cross_val_score(
            decoder, images, labs_idx, groups=runs_idx, cv=cv, n_jobs=15)
        scores.append([np.mean(score)])
        
    with open(path_to_score, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows([scores])
        

    

######## Visualizing cross-validation behavior in scikit-learn ########

def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):
    """Create a sample plot for indices of a cross-validation object."""
    
    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                   c=indices, marker='_', lw=lw, cmap=cmap_cv,
                   vmin=-.2, vmax=1.2)

    # Plot the data classes and groups at the end
    ax.scatter(range(len(X)), [ii + 1.5] * len(X),
               c=y, marker='_', lw=lw, cmap=cmap_data)

    ax.scatter(range(len(X)), [ii + 2.5] * len(X),
               c=group, marker='_', lw=lw, cmap=cmap_data)

    # Formatting
    yticklabels = list(range(n_splits)) + ['class', 'group']
    ax.set(yticks=np.arange(n_splits+2) + .5, yticklabels=yticklabels,
           xlabel='Sample index', ylabel="CV iteration",
           ylim=[n_splits+2.2, -.2], xlim=[0, 100])
    ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
    return ax
        

# ######## Add a counter to each duplicated trials ######## 

# def single_trial_type(datapath, events):   
#     """
#     Relabeling the trial_type to make each single 
#     sub trial unique per sessions
     
#     Parameters
#     ----------
#     datapath: str
#         File-path to a BIDS-formatted events.tsv file in the CNeuroMod
#         project.
#     task_label: string
#         The name of the task for which to generate beta maps 
#     events: list
#         A list of pd.DataFrames or file paths for BIDS-formatted
#         events.tsv files
#     """
#     trial_ev_file_idx = []

#     for events_file in events_files:
#         ev_file = pd.read_table(events_file)

#     #     print(events_file)

#         ev_name = events_file.split('fmriprep-20.1.0/')[1]
#         mod_ev_filename = 'mod_' + ev_name
#         mod_ev_filename = os.path.join(mod_ev_path, mod_ev_filename)
#         dups = (ev_file.loc[ev_file['trial_type'].duplicated(),'trial_type'] + '_' +
#                 ev_file.groupby('trial_type').cumcount().astype(str))+ '_' 
#         ev_file.loc[dups.notnull(),'trial_type'] = dups

#     #     ev_file['trial_type'] = ev_file['trial_type'] + ev_file['trial']


#         mod_events_file = ev_file.to_csv(mod_ev_filename, sep="\t")
#         trial_ev_file_idx.append(mod_events_file)


#         trial_ev_file = mod_ev_path + 'mod_sub-01_ses-002_task-relational_run-02_events.tsv'
#         trial_ev_file = pd.read_table(trial_ev_file)
#         print(trial_ev_file.trial_type.head(20))
