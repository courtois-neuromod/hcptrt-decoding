import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, cross_val_score, train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import classification_report, confusion_matrix
from nilearn.plotting import plot_matrix
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split

import warnings

import visualization


#################### Global variables ####################

bold_suffix = 'space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
pathdata = '/data/neuromod/DATA/cneuromod/hcptrt/derivatives/fmriprep-20.2lts/fmriprep/'
pathevents = '/data/neuromod/DATA/cneuromod/hcptrt/'

processed_data_path = '/home/SRastegarnia/hcptrt_decoding_Shima/hcptrt_decoding/'\
            'timeseries_benchmark/outputs/'


##################### Task Dictionary #####################

def bulid_dict_task_modularity(modality):
    
    """
    Building a dictionary for different conditions 
    of events under different modalities.
    
    Parameters
    ----------
    modality: str
        e.g. 'motor'
    """

    motor_task_con = {"response_left_foot": "Lfoot_mot",
                      "response_left_hand": "Lhand_mot",
                      "response_right_foot": "Rfoot_mot",
                      "response_right_hand": "Rhand_mot",
                      "response_tongue": "tongue_mot"}

    wm_task_con   =  {"0-Back_Body":   "0Bbody_wm",
                      "0-Back_Face":  "0Bface_wm",
                      "0-Back_Place": "0Bplace_wm",
                      "0-Back_Tools":  "0Btools_wm",
                      "2-Back_Body":   "2Bbody_wm",
                      "2-Back_Face":  "2Bface_wm",
                      "2-Back_Place": "2Bplace_wm",
                      "2-Back_Tools":  "2Btools_wm"}

    lang_task_con =  {"presentation_math":  "math_lang",
                      "presentation_story": "story_lang"}
    
    emotion_task_con={"response_shape": "shape_emo",
                      "response_face": "face_emo"}
    
    gambl_task_con = {"Reward":  "win_gamb",
                      "Punishment": "loss_gamb"}
    
    relation_task_con = {"match":    "match_reson",
                      "relation": "relat_reson"}
    
    social_task_con ={"mental": "mental_soc",
                      "rnd":  "random_soc"}
    

    dicts = [motor_task_con, lang_task_con, emotion_task_con, 
             gambl_task_con, relation_task_con, social_task_con, wm_task_con]
    
    from collections import defaultdict
    all_task_con = defaultdict(list)
    for d in dicts:
        for k, v in d.items():
            all_task_con[k].append(v)
            
#     print('all tasks conditions:')
#     all_task_con.items()

    mod_chosen = modality[:3].lower().strip()
    mod_choices = {'mot': 'MOTOR',
                   'lan': 'LANGUAGE',
                   'emo': 'EMOTION',
                   'gam': 'GAMBLING',
                   'rel': 'RELATIONAL',
                   'soc': 'SOCIAL',
                   'wm': 'WM',
                   'all': 'ALLTasks'}
    task_choices = {'mot': motor_task_con,
                    'lan': lang_task_con,
                    'emo': emotion_task_con,
                    'gam': gambl_task_con,
                    'rel': relation_task_con,
                    'soc': social_task_con,
                    'wm': wm_task_con,
                    'all': all_task_con}

    modality = mod_choices.get(mod_chosen, 'default')
    task_contrasts = task_choices.get(mod_chosen, 'default')
    return task_contrasts, modality


######################### Reading Data #########################

def reading_data(subject, modality): 
    sample_events = pathevents + subject +  '/ses-001/func/' + subject + \
    '_ses-001_task-' + modality + '_run-01' + '_events.tsv'
    sample_events_file = pd.read_csv(sample_events, sep = "\t", encoding = "utf8", header = 0)

    fMRIs =  processed_data_path + modality + '_final_fMRI.npy'
    final_bold_files = np.load(fMRIs, allow_pickle=True)

    labels = processed_data_path + modality + '_final_labels.csv'
    volume_labels = pd.read_csv(labels, sep = "\t", encoding = "utf8", header = None)

    categories = list(sample_events_file.trial_type)
    unwanted = {'countdown','cross_fixation','Cue','new_bloc_right_hand', 
                'new_bloc_right_foot','new_bloc_left_foot','new_bloc_tongue', 
                'new_bloc_left_hand','new_bloc_control','new_bloc_relational',
                'new_bloc_shape','new_bloc_face', 'countdown_nan',
                'Cue_nan', 'HRF_lag', 'null'
                }
    categories = [c for c in categories if c not in unwanted]
    conditions = list(set(categories))
    num_cond = len(set(categories))
    
    # Reading the csv labels file & converting to list
    list_of_rows = [list(row) for row in volume_labels.values]
    list_of_rows.insert(0, volume_labels.columns.to_list())
    flat_labels_files = [item for sublist in list_of_rows for item in sublist]
    flat_volume_labels = np.array(flat_labels_files)    
    flat_volume_labels = np.delete(flat_volume_labels, 0)
    final_volume_labels = list(flat_volume_labels)
#     final_volume_labels = flat_volume_labels
   
    return final_volume_labels, final_bold_files, conditions, num_cond 


######################### Decoding #########################

def bench_perceptron(final_bold_files, final_volume_labels, num_cond, epochs):
    
    """
    Multi Layer Perceptron Neural Networks 
    Decoder, with two dense layers.
    
    Parameters
    ----------
    final_bold_files: 'numpy.ndarray' file
       Generated & saved in output path 
       using test_data_preparation script
     
    final_volume_labels: .csv file
       Generated & saved in output path 
       using test_data_preparation script
    """
    
    X = final_bold_files
    y = final_volume_labels

    categories = np.unique(y)
    unique_conditions, order = np.unique(categories, return_index=True)
    unique_conditions = unique_conditions[np.argsort(order)]

    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y)
    temp = np.reshape(y, (len(final_volume_labels),1))
    y = temp

    enc = OneHotEncoder(handle_unknown='ignore')
    y = pd.DataFrame(enc.fit_transform(y).toarray())
    encoded_values = print('label encoded values:', y, "\n")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test) 
    
    
    
#     a = np.shape(y)
#     return(a)
    
    # Initializing
    warnings.filterwarnings('ignore')
    classifier = Sequential()

    classifier.add(Dense(222 , input_dim = 444, activation = 'relu'))
    classifier.add(Dense(111, activation = 'relu'))
    classifier.add(Dense(55, activation = 'relu'))
    classifier.add(Dense(num_cond, activation = 'softmax'))
    summary = classifier.summary()
    
    
    # Compiling
    classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
    
    
#     history = classifier.fit(X_train, y_train, batch_size = 5, epochs = epochs, validation_split = 0.1)
    
    classifier.fit(X_train, y_train, batch_size = 5, epochs = epochs, validation_split = 0.1)
    return X_test, y_test, X_train, y_train
    
    plot_history = visualization.classifier_history (history)
    
    # Making the predictions and evaluating the model
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)

    # Confusion matrix
    cm_ann = confusion_matrix(y_test.values.argmax(axis = 1), y_pred.argmax(axis=1))
    model_conf_matrix = cm_ann.astype('float') / cm_ann.sum(axis = 1)[:, np.newaxis]
    title = 'Artificial Neural Networks confusion matrix'
    visualization.conf_matrix(model_conf_matrix, unique_conditions, title)
    


'--------------------------------------------------------------------------------'

def bench_svm(final_bold_files, final_volume_labels):
    
    """
    Support Vector Machine classifier.
    
    Parameters
    ----------
    final_bold_files: 'numpy.ndarray' file
       Generated & saved in output path 
       using test_data_preparation script
     
    final_volume_labels: list
        output of reading_data function
    """

    X = final_bold_files
    y = final_volume_labels
    
    categories = np.unique(y)
    unique_conditions, order = np.unique(categories, return_index=True)
    unique_conditions = unique_conditions[np.argsort(order)]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initializing
    model_svm = SVC(C = 1.0, cache_size = 200, class_weight = None, coef0 = 0.0,
        decision_function_shape = 'ovo', degree = 3, gamma = 'scale', kernel = 'linear',
        max_iter = -1, probability = False, random_state = None, shrinking = True,
        tol = 0.001, verbose = False)

    model_svm.fit(X_train, y_train)
    score = model_svm.score(X_test, y_test)

    # classification report
    svm = model_svm.predict(X_test)
    report = classification_report(y_test, svm)
    print(report)
    print("Test score with L1 penalty: %.4f" % score)
    
    # Cross validation
    cv_scores_svm  = cross_val_score(model_svm , X_train, y_train, cv=5) 
    print(cv_scores_svm)

    # Prediction accuracy
    classification_accuracy_svm  = np.mean(cv_scores_svm)
    classification_accuracy_svm
    
    # Confusion matrix
    cm_svm = confusion_matrix(y_test, svm)
    model_conf_matrix = cm_svm.astype('float') / cm_svm.sum(axis = 1)[:, np.newaxis]
    title = 'Support Vector Machine confusion matrix'
    visualization.conf_matrix(model_conf_matrix, unique_conditions, title)
    
    
'--------------------------------------------------------------------------------'

























