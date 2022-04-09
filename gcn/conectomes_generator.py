import numpy as np
import glob
import os
import nilearn.connectome
from nilearn.input_data import NiftiLabelsMasker
from nilearn.interfaces.fmriprep import load_confounds_strategy

"""
Utility for generating connectomes from a seession with 
one run of all hcptrt tasks using a weighted average.
"""

def _generate_conn_file(subject, region_approach, resolution, files_epi, conn_dir):

    masker = NiftiLabelsMasker(labels_img = '../{}_{}.nii.gz'.format(region_approach,resolution), 
                               standardize=True, smoothing_fwhm = 5, verbose=5)

    connectom_idx = []
    for file_epi in files_epi:
        print('\n', file_epi)
              
        conf = load_confounds_strategy(file_epi, denoise_strategy="simple",
                                       motion="basic", global_signal="basic") #new nilearn
        sample_ts = masker.fit_transform(file_epi, confounds=conf[0]) #new nilearn      

        corr_measure = nilearn.connectome.ConnectivityMeasure(kind="correlation")
        conn = corr_measure.fit_transform([sample_ts])[0]
        connectom_idx.append(conn)

    wts = np.array([.6,1.2,1,.8,1,1.4])
    np.save(os.path.join(conn_dir, 'conn_wavg_hcptrt_{}_{}{}'\
                         '.npy'.format(subject,region_approach,resolution)), 
            np.average(connectom_idx, axis=0, weights = wts))


def postproc_conectomes_generator(subject, region_approach, resolution):
    
     ##### Elm #####
    raw_data_path = '/data/neuromod/projects/ml_models_tutorial/data/hcptrt/temp/sub-01/ses-003/'
    files_epi = sorted(glob.glob(raw_data_path + '{}_*_run-1_space-MNI152NLin2009cAsym'\
                                 '_desc-preproc_bold.nii.gz'.format(subject)))
    proc_data_path = '/home/srastegarnia/hcptrt_decoding_Shima/data'
    
    print('\n', 'lenght files_epi: ', len(files_epi))
              
#     ##### CC #####
#     pathevents = '/home/rastegar/scratch/hcptrt/'
#     raw_data_path = '/home/rastegar/scratch/hcptrt/derivatives/fmriprep-20.2lts/fmriprep/'
#     proc_data_path = '/home/rastegar/projects/def-pbellec/rastegar/hcptrt_decoding_shima/data/'
     
    conn_dir = os.path.join(proc_data_path, 'connectomes')
    _generate_conn_file(subject, region_approach, resolution, 
                        files_epi, conn_dir)
                  
    conn_files = sorted(glob.glob(conn_dir + '/conn_wavg_hcptrt_{}_{}{}'\
                                  '.npy'.format(subject,region_approach,resolution)))
              
    print('conn_files', conn_files)
    a = np.load(conn_files[0])
    print('connectom file shape:', np.shape(a))
    
         