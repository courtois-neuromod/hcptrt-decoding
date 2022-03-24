import numpy as np
import pandas as pd
import glob
import os
import sys
import warnings
import math
import matplotlib.pyplot as plt
#from nilearn.input_data import NiftiMasker
from nilearn.maskers import NiftiLabelsMasker, NiftiMasker, NiftiMapsMasker
from nilearn.plotting import plot_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense
from termcolor import colored
import random
np.random.seed(0)

sys.path.append(os.path.join(".."))
import visualization

"""
Script for running benchmark models.
This script inputs are outputs files of hcptrt_data_prep.py.
"""


def _generate_all_modality_files(subject, modalities, region_approach, 
                                  HRFlag_process, proc_data_path, resolution):                    
    
    all_modality_concat_bold = []
    all_modality_concat_labels = []
    parcels_no = []
    
    for modality in modalities:
        
        final_bold_outpath = glob.glob(proc_data_path + 'processed_data/proc_fMRI/{}/{}/{}/' \
                                 '{}*{}*{}*.npy'.format(region_approach, resolution, subject,  
                                                        subject, modality, HRFlag_process))
        
        print(final_bold_outpath)
        final_labels_outpath = glob.glob(proc_data_path + 'processed_data/proc_events/{}/{}/{}/' \
                                   '{}*{}*{}*.csv'.format(region_approach, resolution, subject, 
                                                          subject, modality, HRFlag_process))
        
        
        for b_outpath in final_bold_outpath:       
            bold_file = np.load(b_outpath)
            all_modality_concat_bold.append(bold_file)
        
        for l_outpath in final_labels_outpath: 
            labels_file = pd.read_csv(l_outpath, sep='\t', encoding = "utf8", header=None)
            lable_arr = np.array(labels_file[0], dtype=object)
            labels_file = lable_arr
            all_modality_concat_labels.append(labels_file)
    
    flat_bold = [val for sublist in all_modality_concat_bold for val in sublist]
    all_modality_concat_bold = flat_bold
    
    flat_labels = [item for sublist in all_modality_concat_labels for item in sublist]
    all_modality_concat_labels = flat_labels
    
    return(all_modality_concat_bold, all_modality_concat_labels)



def _grid_svm_decoder(all_modality_concat_bold, all_modality_concat_labels, 
                 subject, region_approach, HRFlag_process, results_outpath, resolution):
    
    """
    Support Vector Machine classifier.
    """

    title = '{} Support Vector Machine using {}{}, {} HRFlag'.format(subject, region_approach, 
                                                                   resolution, HRFlag_process) 
    
    output_file_name = '{}_SVM_{}{}_{}_HRFlag'.format(subject, region_approach,
                                                      resolution, HRFlag_process)      
    
    X = all_modality_concat_bold
    y = all_modality_concat_labels
    
    categories = np.unique(y)
    unique_conditions, order = np.unique(categories, return_index=True)
    unique_conditions = unique_conditions[np.argsort(order)]
    
    # Encoding the string to numerical values
    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y)
    y = y.ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)   
    
    # defining parameter range
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf', 'linear']}
    
    grid = GridSearchCV(SVC(max_iter=300, random_state=0), param_grid, refit=True, verbose=3) #SVC(random_state=0)
    
    # fitting the model for grid search
    grid.fit(X_train, y_train)
    
    # print best parameter after tuning
    print(grid.best_params_)
 
    # print how our model looks after hyper-parameter tuning
    print('Best parameters found:\n', grid.best_estimator_)
    
    grid_predictions = grid.predict(X_test)

    # print classification report
    print(classification_report(y_test, grid_predictions))
    
    # confusion matrix
    cm_svm = confusion_matrix(y_test, y_test_pred)
    model_conf_matrix = cm_svm.astype('float') / cm_svm.sum(axis=1)[:, np.newaxis]
        
    visualization.conf_matrix(model_conf_matrix, unique_conditions, 
                              title, results_outpath, output_file_name)
                              


def _svm_decoder(all_modality_concat_bold, all_modality_concat_labels, 
                 subject, region_approach, HRFlag_process, results_outpath, resolution):
    
    """
    Support Vector Machine classifier.
    """

    title = '{} Support Vector Machine using {}{}, {} HRFlag'.format(subject, region_approach, 
                                                                   resolution, HRFlag_process) 
    
    output_file_name = '{}_SVM_{}{}_{}_HRFlag'.format(subject, region_approach,
                                                      resolution, HRFlag_process)      
    
    X = all_modality_concat_bold
    y = all_modality_concat_labels
    
    categories = np.unique(y)
    unique_conditions, order = np.unique(categories, return_index=True)
    unique_conditions = unique_conditions[np.argsort(order)]
    
    # Encoding the string to numerical values
    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y)
    y = y.ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)   
       
    model_svm = SVC(kernel='rbf', random_state=0, C=1)
    model_svm.fit(X_train, y_train)

    # evaluate model
#     print('Accuracy of the prediction on the training set:')
#     y_train_pred = model_svm.predict(X_train)
#     print(classification_report(y_train, y_train_pred))
    
    print('Accuracy of the prediction on the test set:')
    y_test_pred = model_svm.predict(X_test)
    print(classification_report(y_test, y_test_pred))
    
    # confusion matrix
    cm_svm = confusion_matrix(y_test, y_test_pred)
    model_conf_matrix = cm_svm.astype('float') / cm_svm.sum(axis = 1)[:, np.newaxis]
        
    visualization.conf_matrix(model_conf_matrix, unique_conditions, 
                              title, results_outpath, output_file_name)
                                                              
                           

    
def _grid_mlp_decoder(all_modality_concat_bold, all_modality_concat_labels,  
                      subject, region_approach, HRFlag_process, results_outpath, 
                      resolution, parcel_no):
    
    """
    Multi Layer Perceptron Neural Networks Decoder, with two dense layers,  
    using sklearn mlp model with grid search for tuning the hyper parameters.
    """
    
    title = '{} Scikit-Learn’s MLPClassifier using {}{}, {} HRFlag'.format(subject, region_approach,
                                                                           resolution, HRFlag_process) 
    
    output_file_name = '{}_skl_mlp_{}{}_{}_HRFlag'.format(subject, region_approach,
                                                      resolution, HRFlag_process)
    
    X = all_modality_concat_bold
    y = all_modality_concat_labels
    
    categories = np.unique(y)
    unique_conditions, order = np.unique(categories, return_index=True)
    num_cond = len(set(categories))
    unique_conditions = unique_conditions[np.argsort(order)]
    
    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y)
    temp = np.reshape(y, (len(all_modality_concat_labels),1))
    y = temp
    
    enc = OneHotEncoder(handle_unknown='ignore')
    y_onehot = enc.fit_transform(np.array(y).reshape(-1, 1))
    y = pd.DataFrame(y_onehot.toarray())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    param_grid = {
        'hidden_layer_sizes': [(int(resolution/(math.pow(2,1))),int(resolution/(math.pow(2,2))),
                                int(resolution/(math.pow(2,3)))), 
                               (int(resolution/(math.pow(2,1))),int(resolution/(math.pow(2,2)))), 
                               (int(resolution/(math.pow(2,1))))],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant','adaptive']}
    
    
    grid = GridSearchCV(MLPClassifier(random_state=0), param_grid, 
                        n_jobs=-1, cv=5, verbose=3) #MLPClassifier(max_iter=300, random_state=0)
    grid.fit(X_train, y_train)

    # Best paramete set
    print(colored(('Best parameters found:\n'), 'red', attrs=['bold']),
          grid.best_params_)
    
    # All results
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
 
    # print how our model looks after hyper-parameter tuning
    print(colored(('Best model estimation after hyper-parameter tuning:\n'),
                  'red', attrs=['bold']), 
          grid.best_estimator_)
    
    grid_predictions = grid.predict(X_test)

    # print classification report
    print('Results on the test set:')
    print(classification_report(y_test, grid_predictions))
    
    # Confusion matrix
    cm_ann = confusion_matrix(y_test.values.argmax(axis = 1), grid_predictions.argmax(axis=1))
    model_conf_matrix = cm_ann.astype('float') / cm_ann.sum(axis = 1)[:, np.newaxis]
    
    visualization.conf_matrix(model_conf_matrix, unique_conditions, 
                              title, results_outpath, output_file_name)
    
    
    
    
def _mlp_decoder(all_modality_concat_bold, all_modality_concat_labels,  
                 subject, region_approach, HRFlag_process, results_outpath, 
                 resolution, parcel_no):
    
    """
    Multi Layer Perceptron Neural Networks 
    Decoder, with two dense layers.
    """
    
    title = '{} Multi Layer Perceptron Neural Networks using {}{}, {} HRFlag'.format(subject, region_approach,
                                                                                     resolution, HRFlag_process) 

    output_file_name = '{}_mlp_{}{}_{}_HRFlag'.format(subject, region_approach,
                                                      resolution, HRFlag_process) 
  
    X = all_modality_concat_bold
    y = all_modality_concat_labels

    categories = np.unique(y)
    unique_conditions, order = np.unique(categories, return_index=True)
    num_cond = len(set(categories))
    unique_conditions = unique_conditions[np.argsort(order)]

    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y)
    temp = np.reshape(y, (len(all_modality_concat_labels),1))
    y = temp

    enc = OneHotEncoder(handle_unknown='ignore')
    y_onehot = enc.fit_transform(np.array(y).reshape(-1, 1))
    y = pd.DataFrame(y_onehot.toarray())
#     y = pd.DataFrame(enc.fit_transform(y).toarray())
    encoded_values = print('label encoded values:', y, "\n")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test) 
        
    # Initializing
    warnings.filterwarnings('ignore')
    model_mlp = Sequential()
     
    model_mlp.add(Dense(int(resolution/(math.pow(2,1))) , input_dim=parcel_no,
                        kernel_initializer="uniform", activation='relu'))

    model_mlp.add(Dense(int(resolution/(math.pow(2,2))), kernel_initializer="uniform",
                        activation='relu'))

    model_mlp.add(Dense(int(resolution/(math.pow(2,3))), kernel_initializer="uniform",
                       activation='relu'))

    model_mlp.add(Dense(num_cond, activation='softmax'))

    summary = model_mlp.summary()
            
    model_mlp.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])        
    history = model_mlp.fit(X_train, y_train, batch_size=10, epochs=5, validation_split=0.2) 
    
    plot_history = visualization.classifier_history (history, title, results_outpath, 
                                                     output_file_name)
    
    # Making the predictions and evaluating the model
    y_test_pred = model_mlp.predict(X_test)
#     y_test_pred = (y_pred > 0.5)
    print(classification_report(y_test.values.argmax(axis=1), y_test_pred.argmax(axis=1)))
    
#     print ('mean accuracy score:', np.round(accuracy_score(y_test.values.argmax(axis = 1), 
#                                                            y_pred.argmax(axis=1), normalize = True, 
#                                                            sample_weight=None),2))

    # Confusion matrix
    cm_ann = confusion_matrix(y_test.values.argmax(axis = 1), y_test_pred.argmax(axis=1))
    model_conf_matrix = cm_ann.astype('float') / cm_ann.sum(axis = 1)[:, np.newaxis]
    
    visualization.conf_matrix(model_conf_matrix, unique_conditions, 
                              title, results_outpath, output_file_name)
    


    
def _knn_decoder(all_modality_concat_bold, all_modality_concat_labels,
                 subject, region_approach, HRFlag_process, results_outpath, resolution):
            
    """
    k-nearest neighbors classifier. (k=8)
    """
    
    title = '{} KNN using {}{}, {} HRFlag'.format(subject, region_approach, 
                                                  resolution, HRFlag_process) 
    
    output_file_name = '{}_KNN_{}{}_{}_HRFlag'.format(subject, region_approach,
                                                      resolution, HRFlag_process)   
    
    X = all_modality_concat_bold
    y = all_modality_concat_labels
    
    categories = np.unique(y)
    unique_conditions, order = np.unique(categories, return_index=True)
    unique_conditions = unique_conditions[np.argsort(order)]
        
    # Encoding the string to numerical values
    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y)
    y = y.ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)   
    
    # prepare the cross-validation procedure
    cv = KFold(n_splits=10, random_state=0, shuffle=True)

    # Initializing the KNN model
    model_knn = KNeighborsClassifier(n_neighbors=8, algorithm='kd_tree') 
    # model_knn.fit(X_train, y_train)

    # evaluate model
    scores = cross_val_score(model_knn, X_test, y_test, 
                             scoring='accuracy', cv=cv, n_jobs=-1)    
    y_pred = cross_val_predict(model_knn, X_test, y_test, cv=cv)
    report = classification_report(y_test, y_pred)
    
    print(report)
    print(scores)
    print('mean accuracy:%.4f' % np.mean(scores))

    # confusion matrix
    cm_knn = confusion_matrix(y_test, y_pred)
    model_conf_matrix = cm_knn.astype('float') / cm_knn.sum(axis=1)[:, np.newaxis] 
    
    visualization.conf_matrix(model_conf_matrix, 
                              unique_conditions, 
                              title, 
                              results_outpath, 
                              output_file_name)
    
        

        
def _random_forest_decoder(all_modality_concat_bold, all_modality_concat_labels,
                           subject, region_approach, HRFlag_process, results_outpath, resolution):

    title = '{} Random Forest using {}{}, {} HRFlag'.format(subject, region_approach,
                                                            resolution, HRFlag_process) 
    
    output_file_name = '{}_RF_{}_{}_HRFlag'.format(subject, region_approach,
                                                   resolution, HRFlag_process)      
         
    X = all_modality_concat_bold
    y = all_modality_concat_labels

    categories = np.unique(y)
    unique_conditions, order = np.unique(categories, return_index=True)
    unique_conditions = unique_conditions[np.argsort(order)]
        
    # Encoding the string to numerical values
    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y)
    y = y.ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)   
    
    # prepare the cross-validation procedure
    cv = KFold(n_splits=10, random_state=0, shuffle=True)

    # Initializing
    model_rfc = RandomForestClassifier(max_depth=2, random_state=0) 
    # model_rfc.fit(X_train, y_train)

    # evaluate model
    scores = cross_val_score(model_rfc, X_test, y_test, 
                             scoring='accuracy', cv=cv, n_jobs=-1)    
    y_pred = cross_val_predict(model_rfc, X_test, y_test, cv=cv)
    report = classification_report(y_test, y_pred)
    
    print(report)
    print(scores)
    print('mean accuracy:%.4f' % np.mean(scores))

    # confusion matrix
    cm_rfc = confusion_matrix(y_test, y_pred)
    model_conf_matrix = cm_rfc.astype('float') / cm_rfc.sum(axis=1)[:, np.newaxis] 
              
        
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)
    
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)
    
#     # Initializing
#     model_rfc = RandomForestClassifier(max_depth=2, random_state = 0)
#     model_rfc.fit(X_train, y_train)      

#     # classification report
#     score = model_rfc.score(X_test, y_test)
#     rfc = model_rfc.predict(X_test)
#     report = classification_report(y_test, rfc)
#     print(report)
#     print("Average acc without cv: %.4f" % score)
    
#     # Cross validation
#     cv_scores_rfc  = cross_val_score(model_rfc , X_train, y_train, cv = 10)
#     print('10-Fold Cross-Validation:')
#     print(cv_scores_rfc)

#     # The mean prediction accuracy
#     classification_accuracy_rfc  = np.mean(cv_scores_rfc)
#     print('mean accuracy:', classification_accuracy_rfc)
    
#     # Confusion matrix
#     cm_rfc = confusion_matrix(y_test, rfc)
#     model_conf_matrix = cm_rfc.astype('float') / cm_rfc.sum(axis = 1)[:, np.newaxis]
    
    visualization.conf_matrix(model_conf_matrix, 
                              unique_conditions, 
                              title, 
                              results_outpath, 
                              output_file_name)    
    
    
        
def postproc_benchmark_decoder(subject, modalities, decoders, region_approach, 
                               HRFlag_process, resolution): 
    
    home_dir = '/home/SRastegarnia/hcptrt_decoding_Shima/'
    proc_data_path = home_dir + 'data/'
    results_outpath = home_dir + 'benchmark_models/' \
                     'results/{}_{}/{}/{}/'.format(region_approach,
                                                resolution, subject, 
                                                HRFlag_process)

    if not os.path.exists(results_outpath):
        os.makedirs(results_outpath)

    print('\n')
    print(colored((subject, region_approach, resolution, HRFlag_process),
                  'red',attrs=['bold']))
    print('------------------------------------------------------------------------------------------------')

    all_modality_concat_bold, all_modality_concat_labels = _generate_all_modality_files(subject, 
                                                                                        modalities,
                                                                                        region_approach,
                                                                                        HRFlag_process,
                                                                                        proc_data_path,
                                                                                        resolution)
    
    # getting the number of parcels, useful for soft parcellatin approaches like dypac        
    df_path = proc_data_path + '/medial_data/fMRI2/{}/{}/{}/{}_wm_fMRI2.npy'. format(region_approach, 
                                                                                     resolution, 
                                                                                     subject, subject) 
    df = np.load(df_path)
    parcel_no = int(len(df[0][:][1]))
    
    print('Generating concatenated bold and labels files is done, for',
          subject,'with',HRFlag_process,'HRFlag method,','\n',
          'and',region_approach,resolution,'parcelation approach.','\n')                
    print('all_modality_concat_bold shape', np.shape(all_modality_concat_bold))
    print('all_modality_concat_labels shape', np.shape(all_modality_concat_labels),'\n')
    

    for decoder in decoders:

        if decoder == 'svm':
            print(colored(('Support Vector Machine classifier:'),attrs=['bold']))
            _svm_decoder(all_modality_concat_bold = all_modality_concat_bold, 
                         all_modality_concat_labels = all_modality_concat_labels,
                         subject = subject, 
                         region_approach = region_approach, 
                         HRFlag_process = HRFlag_process, 
                         results_outpath = results_outpath,
                         resolution = resolution)
            

        elif decoder == 'mlp': 
            print(colored(('Multi Layer Perceptron Neural Networks classifier'\
                           '(two dense layers):'), attrs=['bold']))
            _mlp_decoder(all_modality_concat_bold = all_modality_concat_bold, 
                         all_modality_concat_labels = all_modality_concat_labels,
                         subject = subject, region_approach = region_approach,
                         HRFlag_process = HRFlag_process,
                         results_outpath = results_outpath,
                         resolution = resolution, 
                         parcel_no = parcel_no)
            

        elif decoder == 'knn': 
            print(colored(('K-Nearest Neighbours classifier'), attrs=['bold']))
            _knn_decoder(all_modality_concat_bold = all_modality_concat_bold, 
                         all_modality_concat_labels = all_modality_concat_labels,
                         subject = subject, 
                         region_approach = region_approach,
                         HRFlag_process = HRFlag_process,
                         results_outpath = results_outpath,
                         resolution = resolution)
            

        elif decoder == 'random_forest': 
            print(colored(('Random Forest classifier'), attrs=['bold']))
            _random_forest_decoder(all_modality_concat_bold = all_modality_concat_bold, 
                                   all_modality_concat_labels = all_modality_concat_labels,
                                   subject = subject, 
                                   region_approach = region_approach, 
                                   HRFlag_process = HRFlag_process,
                                   results_outpath = results_outpath,
                                   resolution = resolution)
            
            
        elif decoder == 'grid_svm': 
            print(colored(('Support Vector Machine classifier with grid search'), attrs=['bold']))
            _grid_svm_decoder(all_modality_concat_bold = all_modality_concat_bold, 
                              all_modality_concat_labels = all_modality_concat_labels,
                              subject = subject, 
                              region_approach = region_approach, 
                              HRFlag_process = HRFlag_process,
                              results_outpath = results_outpath,
                              resolution = resolution) 
            
            
        elif decoder == 'grid_mlp': 
            print(colored(('Multi Layer Perceptron Neural Networks classifier with grid search'), attrs=['bold']))
            _grid_mlp_decoder(all_modality_concat_bold = all_modality_concat_bold, 
                              all_modality_concat_labels = all_modality_concat_labels,
                              subject = subject, 
                              region_approach = region_approach, 
                              HRFlag_process = HRFlag_process,
                              results_outpath = results_outpath,
                              resolution = resolution,
                              parcel_no = parcel_no) 

        else:
            print('The model is not defined')
                        
                        
                        
                          
