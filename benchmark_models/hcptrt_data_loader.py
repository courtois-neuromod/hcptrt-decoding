import numpy as np
import pandas as pd
import glob
import os
import pickle
# from load_confounds import Params9, Params24
# from nilearn.input_data import NiftiLabelsMasker, NiftiMasker, NiftiMapsMasker
from nilearn.maskers import NiftiLabelsMasker, NiftiMasker, NiftiMapsMasker
from termcolor import colored
import nilearn.datasets
from dypac.masker import LabelsMasker, MapsMasker
from nilearn.interfaces.fmriprep import load_confounds_strategy
# from nilearn.interfaces.fmriprep import load_confounds

"""
Utilities for first step of reading and processing hcptrt data.
We need to run it just once.
The outputs are medial_data that are saved as fMRI2 and events2.
"""

class DataLoader():
    
    def __init__(self, TR, modality, subject, 
                 bold_suffix, region_approach, resolution, 
                 fMRI2_out_path=None, events2_out_path=None, 
                 raw_data_path=None, pathevents=None, 
                 raw_atlas_dir=None):  #confounds,       
        
        """ 
        Initializer for DataLoader class.
                
        Parameters
        ----------
          TR: int
              Repetition Time
          confounds: str
              fMRI confounds generating strategy, 
              e.g. Params9()
          modality: str
              task, e.g. 'motor'
          subject: str
              subject ID, e.g. 'sub-01'
          bold_suffix: str
          region_approach: str, 
              parcelation method
              e.g.: 'MIST', 'dypac'
          resolution: int
              there are parcellations at different resolutions
              e.g.: 444, 256, 512, 1024
          fMRI2_out_path: str
              path to bold files w directory
          events2_out_path: str
              path to events files w directory
          raw_data_path: str
              path to HCPtrt dataset fMRI files
          pathevents: str
              path to HCPtrt dataset events files
          raw_atlas_path: str
              path to maskers atlas e.g. 'difumo_atlases'
        """
        
        self.TR = TR
#         self.confounds = confounds
        self.modality = modality
        self.subject = subject
        self.bold_suffix = bold_suffix
        self.region_approach = region_approach
        self.resolution = resolution
        self.fMRI2_out_path = fMRI2_out_path
        self.events2_out_path = events2_out_path
        self.raw_data_path = raw_data_path
        self.pathevents = pathevents
        self.raw_atlas_dir = raw_atlas_dir
        
        if not os.path.exists(self.fMRI2_out_path):
            os.makedirs(self.fMRI2_out_path)

        if not os.path.exists(self.events2_out_path):
            os.makedirs(self.events2_out_path)
            
        if not os.path.exists(self.raw_atlas_dir):
            os.makedirs(self.raw_atlas_dir)


    def _load_fmri_data(self): 
        
        """
        Out put is a list of preprocessed fMRI files using the
        given masker. (for each modality) 
        """

        data_path = sorted(glob.glob(self.raw_data_path+'{}/**/*{}*'
                                     .format(self.subject, self.modality)+self.bold_suffix, 
                                     recursive=True))
        
        print(data_path)        
        print(colored('{}, {}:'.format(self.subject, self.modality), attrs=['bold']))  
        
#         for i in range(0, len(data_path)):
#             print(data_path[i].split('func/', 1)[1])
        
        # make sure we don't exceed 15 runs.        
        if (len(data_path) > 15):
            data_extra_files = len(data_path) - 15 
            print(colored('Regressed out {} extra following fMRI file(s):'
                          .format(data_extra_files), 'red', attrs=['bold']))
            for i in range(15, len(data_path)):
                print(colored(data_path[i].split('func/', 1)[1], 'red'))
            for i in range(15, len(data_path)):
                data_path.pop()
                                   
        print('The number of bold files:', len(data_path))
 
        # generate masks
        if self.region_approach == 'MIST':
                                                                        
            masker = NiftiLabelsMasker(labels_img='{}_{}.nii.gz'.format(self.region_approach,
                                                                          self.resolution), 
                                       standardize=True, smoothing_fwhm=5)
            
            fmri_t = []
            for dpath in data_path:
                print('dpath: ', dpath)
                
#                 data_fmri = masker.fit_transform(dpath, confounds=self.confounds.load(dpath)) #old nilearn
                
                conf = load_confounds_strategy(dpath, denoise_strategy="simple",
                                               motion="basic", global_signal="basic") #new nilearn
                data_fmri = masker.fit_transform(dpath, confounds=conf[0]) #new nilearn
                fmri_t.append(data_fmri)
                
                print(dpath.split('func/', 1)[1])
                print(data_fmri)
                print('shape:', np.shape(data_fmri))
              
            
        elif self.region_approach == 'difumo':
            
#             num_parcels = int(self.region_approach.split("_", 1)[1])
            atlas = nilearn.datasets.fetch_atlas_difumo(data_dir=self.raw_atlas_dir, 
                                                        dimension=self.resolution)
            atlas_filename = atlas['maps']
            atlas_labels = atlas['labels']            
            masker = NiftiMapsMasker(maps_img=atlas['maps'], standardize=True, verbose=5)
            
            fmri_t = []
            for dpath in data_path:    
#                 data_fmri = masker.fit_transform(dpath, confounds=self.confounds.load(dpath)) #old nilearn
                
                conf = load_confounds_strategy(dpath, denoise_strategy="simple",
                                               motion="basic", global_signal="basic") #new nilearn
                data_fmri = masker.fit_transform(dpath, confounds=conf[0]) #new nilearn
                
                fmri_t.append(data_fmri)
        

        elif self.region_approach == 'dypac':
            
#             LOAD_CONFOUNDS_PARAMS = {
#                 "strategy": ["motion", "high_pass", "wm_csf", "global_signal"],
#                 "motion": "basic",
#                 "wm_csf": "basic",
#                 "global_signal": "basic",
#                 "demean": True
#             } # costume confounds
            
            path_dypac = '/data/cisl/pbellec/models'
            file_mask = os.path.join(path_dypac, 
                                     '{}_space-MNI152NLin2009cAsym_label-GM_mask.nii.gz'.format(self.subject))
            file_dypac = os.path.join(path_dypac,
                                      '{}_space-MNI152NLin2009cAsym_desc-dypac{}_components.nii.gz'.format(
                                          self.subject, self.resolution))
            print('file_mask: ', file_mask)
            print('file_dypac: ', file_dypac, '\n')
            masker = NiftiMasker(standardize=True, detrend=False, smoothing_fwhm=5, mask_img=file_mask)
            
            fmri_t = []
            for dpath in data_path:
                
                conf = load_confounds_strategy(dpath, denoise_strategy='simple', global_signal='basic')
#                 conf = load_confounds(dpath, strategy=**LOAD_CONFOUNDS_PARAMS) # costume confounds      
    
                masker.fit(dpath)
                maps_masker = MapsMasker(masker=masker, maps_img=file_dypac)
                data_fmri = maps_masker.transform(img=dpath, confound=conf[0])
                fmri_t.append(data_fmri)
                
                print('fMRI file:' ,dpath.split('func/', 1)[1])
                print('shape:', np.shape(data_fmri), '\n')
                print(data_fmri)
                print('\n')
            
        else:
            masker = NiftiMasker(standardize=True)                   

            fmri_t = []
            for dpath in data_path:    
#                 data_fmri = masker.fit_transform(dpath, confounds=self.confounds.load(dpath)) #old nilearn
                
                conf = load_confounds_strategy(dpath, denoise_strategy="simple",
                                               motion="basic", global_signal="basic") #new nilearn
                data_fmri = masker.fit_transform(dpath, confounds=conf[0]) #new nilearn
                
                
                
                fmri_t.append(data_fmri)

        print('### Reading Nifiti files is done!')
        print('-----------------------------------------------')

        return fmri_t, masker, data_path
   
    
    
    def _load_events_files(self):
        
        """
        Output is a list of relabeled events file (15 for each modality)
        """

        events_path = sorted(glob.glob(self.pathevents + '{}/**/func/*{}*_events.tsv'
                                       .format(self.subject, self.modality), 
                                       recursive=True))
        
        # make sure we don't exceed 15 runs.
        if (len(events_path) > 15):
            events_extra_files = len(events_path) - 15
            print(colored('Regressed out {} extra following events file(s):'
                          .format(events_extra_files), 'red', attrs=['bold'])) 
            for i in range(15, len(events_path)):
                print(colored(events_path[i].split('func/', 1)[1], 'red'))
            for i in range(15, len(events_path)):
                events_path.pop()            

        print('The number of events files:', len(events_path))
        
        # Labeling the conditions
        events_files = []
        for epath in events_path: 
            event = pd.read_csv(epath, sep="\t", encoding="utf8")
            print(colored(epath.split('func/', 1)[1], 'blue', attrs=['bold']))
            print(np.shape(event))
            print(np.unique(event.trial_type))
            
            if self.modality == 'emotion': 
                event.trial_type = event['trial_type'].replace(['response_face',
                                                                'response_shape'],
                                                               ['fear','shape'])
                    
            if self.modality == 'language':
                event.trial_type = event['trial_type'].replace(['presentation_story',
                                                                'question_story',
                                                                'response_story',
                                                                'presentation_math',
                                                                'question_math',
                                                                'response_math'],
                                                               ['story','story','story',
                                                                'math','math','math']) 

            if self.modality == 'motor':             
                event.trial_type = event['trial_type'].replace(['response_left_foot',
                                                                'response_left_hand',
                                                                'response_right_foot',
                                                                'response_right_hand',
                                                                'response_tongue'],
                                                               ['footL','handL','footR',
                                                                'handR','tongue']) 

            if self.modality == 'relational':                   
                event.trial_type = event['trial_type'].replace(['Control','Relational'],
                                                               ['match','relational']) 
                
                
            if self.modality == 'wm':                    
                event.trial_type = event.stim_type.astype(str) + '_' + \
                event.trial_type.astype(str)
                event.trial_type = event['trial_type'].replace(['Body_0-Back','Body_2-Back',
                                                                'Face_0-Back','Face_2-Back',
                                                                'Place_0-Back','Place_2-Back',
                                                                'Tools_0-Back','Tools_2-Back'],
                                                               ['zbody0b','body2b','face0b',
                                                                'face2b','place0b','place2b',
                                                                'tool0b','tool2b'])
            
            print(colored('After relabeling:', attrs=['bold']))
            print(np.unique(event.trial_type), '\n')
            print(event.trial_type.head(20))

            events_files.append(event)
        
        print('### Reading events files is done!')
        print('-----------------------------------------------')

        return events_files

        
############################################## Shima local ###########################################   
# def reading_events2(subject, modality, events2_out_path, region_approach):
    
#     events_outname = events2_out_path + subject + '_' +  modality + '_events2'
#     pickle_in = open(events_outname, "rb")
#     events_files = pd.read_pickle(events_outname)
    
#     return events_files      
######################################################################################################      
    
    
def _check_input(fmri_t, events_files):

    """
    - Remove bold files extra ending volumes if there exist.
    - Check events and fMRI files consistency.
        
    Parameters
    ----------
    fmri_t: list
        output of load_fmri_data function
    events_files: list
        output of load_events_files function
    """

    data_lenght = len(fmri_t)
    data_lenght = int (data_lenght or 0)

    # Removing extra volumes
    for i in range(0, data_lenght-1):
        if fmri_t[i].shape != fmri_t[i+1].shape:
            print('There is mismatch in BOLD file size:')

            if fmri_t[i].shape > fmri_t[i+1].shape:         
                a = np.shape(fmri_t[i])[0] - np.shape(fmri_t[i+1])[0]        
                fmri_t[i] = fmri_t[i][0:-a, 0:]
                print('The', a,'extra volumes of bold file number', i,'is removed.')
            else:
                b = np.shape(fmri_t[i+1])[0] - np.shape(fmri_t[i])[0]        
                fmri_t[i+1] = fmri_t[i+1][0:-b, 0:]
                print('The', b,'extra volumes of bold file number', i+1,'is removed.')

    if len(events_files) != len(fmri_t):
        print('Miss-matching between events and fmri files')
        print('Number of Nifti files:' ,len(fmri_t))
        print('Number of events files:' ,len(events_files))
    else:
        print('Events and fMRI files are Consistent.')

    print('### Cheking data is done!')
    print('-----------------------------------------------')


    
def _save_files(fmri_t, events_files, subject, modality, 
                fMRI2_out_path, events2_out_path):
    
# def _save_files(events_files, subject, modality, 
#                 events2_out_path, region_approach): #واسه وقتی فقط میخوام ایونت رو تغییر بدم
    
    """
    - Save a matrix of preprocessed fMRI file for each task.
    - Save an events file for each modality as a pickle file.
    """
    
    # fMRI
    bold_outname = fMRI2_out_path + subject + '_' + modality + '_fMRI2.npy'
    np.save(bold_outname, fmri_t)

    temp = np.load(bold_outname, allow_pickle=True)
    fmri_t = temp
    
    print('Bold file:', bold_outname)
    print('### Saving Nifiti files as matrices is done!')
    print('-----------------------------------------------')
    
    # events
    events_outname = events2_out_path + subject + '_' + modality + '_events2'
    events_dict = events_files
    pickle_out = open(events_outname,"wb")
    pickle.dump(events_dict, pickle_out)
    pickle_out.close()

    print('Events pickle file:', events_outname)
    print('### Saving events pickle files is done!')
    print('-----------------------------------------------')
    

    
def postproc_data_loader(subject, modalities, region_approach, resolution): # confounds, 
    
    TR = 1.49

#    ##### Elm #####
#    pathevents = '/data/neuromod/projects/ml_models_tutorial/data/hcptrt/HCPtrt_events_DATA/'
#    raw_data_path = '/data/neuromod/DATA/cneuromod/hcptrt/derivatives/fmriprep-20.2lts/fmriprep/'
#    proc_data_path = '/home/SRastegarnia/hcptrt_decoding_Shima/data/'


    ##### CC #####
    pathevents = '/home/rastegar/projects/def-pbellec/rastegar/Beluga_data/rastegar_hcptrt_decoding/data/hcptrt/'
#    raw_data_path = pathevents + 'derivatives/fmriprep-20.2lts/fmriprep/'
    raw_data_path = '/home/rastegar/scratch/hcptrt/derivatives/fmriprep-20.2lts/fmriprep'
    proc_data_path = '/home/rastegar/projects/def-pbellec/rastegar/hcptrt_decoding_shima/data/'


    bold_suffix = '_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'

    raw_atlas_dir = os.path.join(proc_data_path, "raw_atlas_dir") 

    fMRI2_out_path = proc_data_path + 'medial_data/fMRI2/{}/{}/{}/'.format(region_approach,
                                                                           resolution, subject)        
    events2_out_path = proc_data_path + 'medial_data/events2/{}/{}/{}/'.format(region_approach,
                                                                               resolution, subject)

    for modality in modalities:
        print(colored(modality,'red', attrs=['bold']))

        load_data = DataLoader(TR = TR,  
                               modality = modality, subject = subject, 
                               bold_suffix = bold_suffix,
                               region_approach = region_approach,
                               resolution = resolution,
                               fMRI2_out_path = fMRI2_out_path, 
                               events2_out_path = events2_out_path, 
                               raw_data_path = raw_data_path, 
                               pathevents = pathevents, 
                               raw_atlas_dir = raw_atlas_dir) #confounds = confounds,

        fmri_t, masker, data_path  = load_data._load_fmri_data()

        events_files = load_data._load_events_files()
#             events_files = _reading_events2(subject, modality, events2_out_path, region_approach) # Shima local

        _check_input(fmri_t, events_files)

        _save_files(fmri_t, events_files, subject, modality,  
                    fMRI2_out_path, events2_out_path)

#             _save_files(events_files, subject, modality,  
#                         events2_out_path, region_approach) #واسه وقتی فقط میخوام ایونت رو تغییر بدم 

        print('#############################################################')
    

