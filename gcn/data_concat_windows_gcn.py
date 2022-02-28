import numpy as np
import pandas as pd
import os
import glob
import sys
import pathlib
import csv
from termcolor import colored

sys.path.append(os.path.join("../"))
import utils

def ـconcat_labels(final_volume_labels):
    
    """ 
    Concatenating labels with the same conditions.
    
    Parameter
    ----------
    final_volume_labels: .csv file
        output of the 'hcptrt_data_prep.py' script.
    """ 
    
    # converting the final_volume_labels to a class 'numpy.ndarray'
    final_volume_labels = np.asarray(final_volume_labels)
    final_volume_labels = np.reshape(final_volume_labels, (len(final_volume_labels), 1))
    
    concat_volume_labels = []
    concat_volume_labels = np.append(concat_volume_labels, final_volume_labels[0, 0])
    concat_volume_num = []
    num = 1
    l = len(final_volume_labels[:, 0])
    
    for c in range(1, l):  
        if (final_volume_labels[c] != final_volume_labels[c - 1]):
            concat_volume_labels = np.append(concat_volume_labels, final_volume_labels[c])
            concat_volume_num = np.append(concat_volume_num, num)
            num = 1
        else:
            num += 1
    
    concat_volume_num = np.append(concat_volume_num, num)
    
    ###############################################################################
    print('lenght final_volume_labels:', l)
    print('type concat_volume_num:', type(concat_volume_num))
    print('shape concat_volume_num:', np.shape(concat_volume_num))
    print(concat_volume_num)
    
    print('type concat_volume_labels:', type(concat_volume_labels))
    print('shape type concat_volume_labels:', np.shape(concat_volume_labels))
    print(concat_volume_labels)
    ###############################################################################
    
    return concat_volume_num, concat_volume_labels



def _concat_files(subject, modality, HRFlag_process, concat_out_path, 
                  concat_volume_num, concat_volume_labels,
                  final_volume_labels, final_bold_file):
    
    """ 
    Generate & save the concatenated file of bold data with the same labels.
    
    Parameter
    ----------
    concat_out_path: str
        path for saving concatenated outputs.
    """ 
        
    concat_volume_num = np.array(concat_volume_num, dtype = np.int)
    i = 0

    if (len(final_volume_labels) == len(final_bold_file)):
        if (len(concat_volume_labels)==len(concat_volume_num)):
            if (len(final_volume_labels) == int(sum(concat_volume_num))):

                for j in range(0, len(concat_volume_num)):
                    condition = concat_volume_labels[j]
                    concat_bold_files = final_bold_file[i:i+1]
                    i += 1

                    for jj in range(1, concat_volume_num[j]):
                        concat_bold_files = np.concatenate((concat_bold_files, 
                                                            final_bold_file[i:i+1]), 
                                                           axis = 0)
                        i += 1

                    concat_file_name = concat_out_path + subject + '_' + condition + \
                                       '_' + HRFlag_process + '_concat_fMRI.npy'
                    file = pathlib.Path(concat_file_name)

                    if file.exists ():
                        concat_file = np.load(concat_file_name, allow_pickle=True)
                        concat_file = np.concatenate((concat_file, 
                                                      concat_bold_files), 
                                                     axis = 0)
                        np.save(concat_file_name, concat_file)
                    else:
                        np.save(concat_file_name, concat_bold_files)

                print('concat_file_name:', concat_file_name)
                print('---------------------------------------------------------------------','\n')
                
            else:
                print('final_volume_labels is not equal to sum of concat_volume_num')                                        
        else:
            print('concat_volume_labels and concat_volume_num do not have the same lenght')                    
    else:
        print('final_volume_labels and final_bold_files do not have the same lenght')
                    
     
    return concat_file_name

        
        
def _generate_phenotypic_data(concat_out_path,subject,HRFlag_process):
    
    conditions = []
    with open(concat_out_path + 'phenotypic_data.tsv','wt') as out_file:
        
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(['ID', 'Subject Type'])

        for file in glob.glob(concat_out_path + '*.npy'):

            cond = utils.between(file, "{}_".format(subject), "_{}_concat_fMRI.npy".format(HRFlag_process))
            conditions = np.append(conditions, cond)
            tsv_writer.writerow(["{}_{}".format(subject,cond), cond])

        return (out_file)
    

def _remove_extra_volumes():
    
    """ 
    Delete the extra volumes that some modality have more that min duration.
    
    Parameter
    ----------
    final_volume_labels: .csv file
        output of the 'hcptrt_data_prep.py' script.
    """ 
    
        
def postproc_time_windows(subject, region_approach, modalities, HRFlag_process, resolution): 
    
#     proc_data_path = os.path.join('..','data','processed_data')
#     concat_data_path = os.path.join('..','data','concat_data')
    
    proc_data_path = '/home/SRastegarnia/hcptrt_decoding_Shima/data/'
    
    final_bold_out_path = proc_data_path + 'processed_data/proc_fMRI/{}/{}/{}/'.format(region_approach,
                                                                                    resolution, subject)       
    final_labels_out_path = proc_data_path + 'processed_data/proc_events/{}/{}/{}/'.format(region_approach,
                                                                                        resolution, subject)    
    concat_out_path = proc_data_path + 'concat_data/{}/{}/{}/'.format(region_approach, 
                                                                      resolution, subject) 
    
    print(colored('{}, {}, {}, res={}:'.format(subject, region_approach,
                                               HRFlag_process, resolution), attrs=['bold']))  
    
    if not os.path.exists(concat_out_path):
        os.makedirs(concat_out_path)

    # delete the old contents to avoid concatenating files multiple times
    old_files = glob.glob(concat_out_path + '*')

    for f in old_files:
        os.remove(f)

    old_dirContents = os.listdir(concat_out_path)
    print(concat_out_path)
    print('old concat dir contents:', old_dirContents)

    for modality in modalities:
        print(colored((subject, modality), attrs=['bold']), '\n')

        final_bold_name = final_bold_out_path + '{}_{}_{}_final_fMRI.npy'.format(subject, modality, 
                                                                                 HRFlag_process)
        final_bold_file = np.load(final_bold_name)

        final_labels_name = final_labels_out_path + '{}_{}_{}_final_labels.csv'.format(subject, modality, 
                                                                                HRFlag_process)
        final_volume_labels = pd.read_csv(final_labels_name, sep='\t', 
                                          encoding="utf8", header=None)

        concat_volume_num, concat_volume_labels = ـconcat_labels(final_volume_labels)

        if (len(old_dirContents) == 0 or len(old_dirContents) == 1):
            concat_file_name = _concat_files(subject, modality, HRFlag_process, concat_out_path,
                                             concat_volume_num, concat_volume_labels,
                                             final_volume_labels, final_bold_file)

        else:
            print('concat data path is not empty')

    _generate_phenotypic_data(concat_out_path, subject, HRFlag_process)    
                        
                        
                                                                                              