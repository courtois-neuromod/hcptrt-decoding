import numpy as np
import pandas as pd
import glob
import os
import sys
import warnings
import math
import csv
from time import time
import matplotlib.pyplot as plt
from nilearn.maskers import NiftiLabelsMasker, NiftiMasker, NiftiMapsMasker
from nilearn.plotting import plot_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold, KFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB, ComplementNB, BernoulliNB 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from keras.models import Sequential
from keras.layers import Dense
from termcolor import colored
import random
np.random.seed(0)

# sys.path.append(os.path.join(".."))
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
        
#         print(final_bold_outpath)
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
                      subject, decoder, region_approach, HRFlag_process, 
                      results_outpath, cm_results_outpath, resolution):
    
    """
    Support Vector Machine classifier with GridSearchCV.
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
    
    # defining parameter range
    param_grid = {'C': [10], #[0.1, 1, 10, 100, 1000]
                  'gamma': [0.001], #[1, 0.1, 0.01, 0.001, 0.0001]
                  'kernel': ['rbf']} # ['rbf', 'linear']
    
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=0)
    
    t0 = time()
    grid = GridSearchCV(SVC(random_state=0), param_grid, refit=True, 
                        n_jobs=-1, cv=cv, verbose=3)

    grid.fit(X_train, y_train)
    
    # how our model looks after hyper-parameter tuning
    print(colored(('Model best parameters found:\n'),'red',attrs=['bold']),grid.best_params_)
    
    # classification report
    grid_predictions = grid.predict(X_test)
    print(classification_report(y_test, grid_predictions))
    print("SVM decoding time:", round(time()-t0, 3), "s")
    
    # confusion matrix
    cm_svm = confusion_matrix(y_test, grid_predictions) #,normalize='true'
    model_cm = np.round(cm_svm.astype('float') / cm_svm.sum(axis=1)[:, np.newaxis], 2)
    
    visualization.conf_matrix(model_cm, unique_conditions, title, cm_results_outpath, 
                              output_file_name, decoder, subject, region_approach, 
                              resolution, HRFlag_process)
                              

        
def _svm_decoder(all_modality_concat_bold, all_modality_concat_labels, 
                 subject, decoder, region_approach, HRFlag_process, 
                 results_outpath, cm_results_outpath, resolution):
    
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
    model_cm = cm_svm.astype('float') / cm_svm.sum(axis = 1)[:, np.newaxis]
        
    visualization.conf_matrix(model_cm, unique_conditions, title, cm_results_outpath, 
                              output_file_name, decoder, subject, region_approach, 
                              resolution, HRFlag_process)
                                                              
                           
    
def _grid_mlp_decoder(all_modality_concat_bold, all_modality_concat_labels,  
                      subject, decoder, region_approach, HRFlag_process,  
                      results_outpath, cm_results_outpath, resolution):
    
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
    num_cond = len(set(categories))
    unique_conditions, order = np.unique(categories, return_index=True)    
    unique_conditions = unique_conditions[np.argsort(order)]
    
    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y)
    temp = np.reshape(y, (len(all_modality_concat_labels),1))
    y = temp
    
    enc = OneHotEncoder(handle_unknown='ignore')
    y_onehot = enc.fit_transform(np.array(y).reshape(-1, 1))
    y = pd.DataFrame(y_onehot.toarray())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # defining parameter range
    param_grid = {
        'hidden_layer_sizes': [
#             (int(resolution/(math.pow(2,1))),int(resolution/(math.pow(2,2))),
#                                 int(resolution/(math.pow(2,3)))), 
                               (int(resolution/(math.pow(2,1))),int(resolution/(math.pow(2,2)))), 
#                                (int(resolution/(math.pow(2,1))))
                              ],
        'activation': ['relu'], #['tanh', 'relu']
        'solver': ['adam'], #['sgd','adam']
        'alpha': [0.05], #[0.0001, 0.05, 0.1]
        'learning_rate': ['constant']} #['constant','adaptive']
    
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=0)
    grid = GridSearchCV(MLPClassifier(random_state=0), param_grid, refit=True, 
                        n_jobs=-1, cv=cv, verbose=3) #MLPClassifier(max_iter=300, random_state=0) # cv=10,cv=5
    
    # fitting the model for grid search
    grid.fit(X_train, y_train)

    # print best parameter after tuning
    print(colored(('Best parameters found:\n'),'red',attrs=['bold']),grid.best_params_)
    
    # print how our model looks after hyper-parameter tuning
    print(colored(('Best model estimation after hyper-parameter tuning:\n'),
                  'red', attrs=['bold']), grid.best_estimator_)
    
    # All results
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    
    grid_predictions = grid.predict(X_test)

    # print classification report
    print('Results on the test set:')
    print(classification_report(y_test, grid_predictions))
    
    # Confusion matrix
    cm_ann = confusion_matrix(y_test.values.argmax(axis = 1), grid_predictions.argmax(axis=1))
#     cm_ann = confusion_matrix(y_test, grid_predictions)
    model_cm = cm_ann.astype('float') / cm_ann.sum(axis = 1)[:, np.newaxis]
    
    visualization.conf_matrix(model_cm, unique_conditions, title, cm_results_outpath, 
                              output_file_name, decoder, subject, region_approach, 
                              resolution, HRFlag_process)
    
        
    
def _mlp_decoder(all_modality_concat_bold, all_modality_concat_labels,  
                 subject, decoder, region_approach, HRFlag_process,  
                 results_outpath, cm_results_outpath, resolution):
    
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
    
    t0 = time()
    model_mlp = Sequential()
     
    model_mlp.add(Dense(int(resolution/(math.pow(2,1))) , input_dim=resolution,
                        kernel_initializer='uniform', activation='relu', use_bias=True,
                        bias_initializer='zeros'))

    model_mlp.add(Dense(int(resolution/(math.pow(2,2))), kernel_initializer='uniform',
                        activation='relu',use_bias=True, bias_initializer='zeros'))

    model_mlp.add(Dense(num_cond, activation='softmax'))

    summary = model_mlp.summary()     
    model_mlp.compile(optimizer='adamax', loss='categorical_crossentropy', metrics=['accuracy'])        
    history = model_mlp.fit(X_train, y_train, batch_size=10, epochs=10, validation_split=0.1) 
    
    plot_history = visualization.classifier_history (history, title, results_outpath, 
                                                     output_file_name)
    
    # Making the predictions and evaluating the model
    y_test_pred = model_mlp.predict(X_test)
    
    print(classification_report(y_test.values.argmax(axis=1), y_test_pred.argmax(axis=1)))
    
    print ('mean accuracy score:', np.round(accuracy_score(y_test.values.argmax(axis = 1), 
                                                           y_test_pred.argmax(axis=1), 
                                                           normalize = True, 
                                                           sample_weight=None),2))
    
    print("MLP decoding time:", round(time()-t0, 3), "s")

    # Confusion matrix
    cm_ann = confusion_matrix(y_test.values.argmax(axis = 1), y_test_pred.argmax(axis=1))
    model_cm = np.round(cm_ann.astype('float') / cm_ann.sum(axis = 1)[:, np.newaxis], 2) 
    
    visualization.conf_matrix(model_cm, unique_conditions, title, cm_results_outpath, 
                              output_file_name, decoder, subject, region_approach, 
                              resolution, HRFlag_process)

     

def _grid_knn_decoder(all_modality_concat_bold, all_modality_concat_labels, 
                      subject, decoder, region_approach, HRFlag_process, 
                      results_outpath, cm_results_outpath, resolution):
    
    """
    k-nearest neighbors classifier with GridSearchCV.
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
    
    # defining parameter range
    param_grid = {'n_neighbors': [4], # [2,4,8,10,16]
                  'leaf_size': [1], # [1,10,20,30]
                  'p': [1], # [1,2]
                  'weights' : ['distance'], # ['uniform','distance']
                  'metric' : ['minkowski'], # ['minkowski','euclidean','manhattan']
                  'algorithm': ['auto']} # ['auto','kd_tree']
    
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=0)
    
    t0 = time()
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, refit=True, 
                        n_jobs=-1, cv=cv, verbose=3)

    grid.fit(X_train, y_train)
    
    # how our model looks after hyper-parameter tuning
    print(colored(('Model best parameters found:\n'),'red',attrs=['bold']),grid.best_params_)
    
    # classification report
    grid_predictions = grid.predict(X_test)
    print(classification_report(y_test, grid_predictions))
    print("KNN decoding time:", round(time()-t0, 3), "s")
    
    # confusion matrix
    cm_knn = confusion_matrix(y_test, grid_predictions)
    model_cm = np.round(cm_knn.astype('float') / cm_knn.sum(axis=1)[:, np.newaxis], 2) 
        
    visualization.conf_matrix(model_cm, unique_conditions, title, cm_results_outpath, 
                              output_file_name, decoder, subject, region_approach, 
                              resolution, HRFlag_process)

         
    
def _knn_decoder(all_modality_concat_bold, all_modality_concat_labels,
                 subject, decoder, region_approach, HRFlag_process, 
                 results_outpath, cm_results_outpath, resolution): 
    
            
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
#    cm_knn = confusion_matrix(y_test, y_pred) #Apr10
    cm_knn = confusion_matrix(y_test, y_test_pred)
    model_cm = np.round(cm_knn.astype('float') / cm_knn.sum(axis=1)[:, np.newaxis], 2)  
    
    visualization.conf_matrix(model_cm, unique_conditions, title, cm_results_outpath, 
                              output_file_name, decoder, subject, region_approach, 
                              resolution, HRFlag_process)

    

def _grid_random_forest_decoder(all_modality_concat_bold, all_modality_concat_labels, 
                                subject, decoder, region_approach, HRFlag_process, 
                                results_outpath, cm_results_outpath, resolution):
    
    """
    Random Forest classifier with GridSearchCV.
    """

    title = '{} Random Forest using {}{}, {} HRFlag'.format(subject, region_approach, 
                                                  resolution, HRFlag_process) 
    
    output_file_name = '{}_RandomForest_{}{}_{}_HRFlag'.format(subject, region_approach,
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
    
    # defining parameter range
    param_grid = {#'n_estimators': [10,50,200],
                  'n_estimators': [200], #[int(x) for x in np.linspace(100,1000,num = 10)], or 200?
                  'max_features': [21], #[2,5,10,21,25,50,'auto','sqrt']
                  'max_depth': [20],#[10,20,50, None]
                  'min_samples_split': [2], #[1,2,5,8]
                  'min_samples_leaf': [1],#[1,2,4,8]
                  'bootstrap': [False]#[True, False]
                  }
    
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=0)
    
    t0 = time()
    grid = GridSearchCV(RandomForestClassifier(), param_grid, refit=True, 
                        n_jobs=-1, cv=cv, verbose=3)

    grid.fit(X_train, y_train)
    
    # how our model looks after hyper-parameter tuning
    print(colored(('Model best parameters found:\n'),'red',attrs=['bold']),grid.best_params_)
    
    # classification report
    grid_predictions = grid.predict(X_test)
    print(classification_report(y_test, grid_predictions))
    print("RandomForest decoding time:", round(time()-t0, 3), "s")
    
    # confusion matrix
    cm_rf = confusion_matrix(y_test, grid_predictions)
    model_cm = np.round(cm_rf.astype('float') / cm_rf.sum(axis=1)[:, np.newaxis], 2)
        
    visualization.conf_matrix(model_cm, unique_conditions, title, cm_results_outpath, 
                              output_file_name, decoder, subject, region_approach, 
                              resolution, HRFlag_process)



    
def _random_forest_decoder(all_modality_concat_bold, all_modality_concat_labels,
                           subject, decoder, region_approach, HRFlag_process, 
                           results_outpath, cm_results_outpath, resolution):

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
    model_cm = cm_rfc.astype('float') / cm_rfc.sum(axis=1)[:, np.newaxis] 
              
        
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=0)
    
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)
    
#     # Initializing
#     model_rfc = RandomForestClassifier(max_depth=2, random_state=0)
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
#     model_cm = cm_rfc.astype('float') / cm_rfc.sum(axis = 1)[:, np.newaxis]
    
    visualization.conf_matrix(model_cm, unique_conditions, title, cm_results_outpath, 
                              output_file_name, decoder, subject, region_approach, 
                              resolution, HRFlag_process)  
    


    
def _grid_logistic_regression_decoder(all_modality_concat_bold, all_modality_concat_labels,
                                      subject, decoder, region_approach, HRFlag_process,   
                                      results_outpath, cm_results_outpath, resolution):
    
    """
    Logistic Regression classifier with GridSearchCV.
    """

    title = '{} Logistic Regression using {}{}, {} HRFlag'.format(subject, region_approach, 
                                                                  resolution, HRFlag_process) 
    
    output_file_name = '{}_LogisticRegression_{}{}_{}_HRFlag'.format(subject, region_approach,
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)   
    
    # defining parameter range
    param_grid = {'solver': ['liblinear'], # ['newton-cg','lbfgs','liblinear','sag','saga']
                  'penalty': ['l1'], # ['none','l1','l2','elasticnet']
                  'C': [0.1], # [100,10,1.0,0.1,0.01]
                  'max_iter': [20], #[10,20,21,50,100,1000]
                  'class_weight': [None] #['balanced', None]
                  }
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=0)
    
    t0 = time()
    grid = GridSearchCV(LogisticRegression(), param_grid, refit=True, 
                        n_jobs=-1, cv=cv, verbose=3)
    
    grid.fit(X_train, y_train)
    
    # print best parameter after tuning
    print(colored(('Model best parameters found:\n'),'red',attrs=['bold']),grid.best_params_)
 
    # classification report
    grid_predictions = grid.predict(X_test)
    print(classification_report(y_test, grid_predictions))
    print("Logistic regression decoding time:", round(time()-t0, 3), "s")
    
    # confusion matrix
    cm_lr = confusion_matrix(y_test, grid_predictions)
    model_cm = np.round(cm_lr.astype('float') / cm_lr.sum(axis=1)[:, np.newaxis], 2)
        
    visualization.conf_matrix(model_cm, unique_conditions, title, cm_results_outpath, 
                              output_file_name, decoder, subject, region_approach, 
                              resolution, HRFlag_process)

    
        
    
def _grid_ridge_decoder(all_modality_concat_bold, all_modality_concat_labels,
                        subject, decoder, region_approach, HRFlag_process,   
                        results_outpath, cm_results_outpath, resolution):
    
    """
    Ridge Regression classifier with GridSearchCV.
    """

    title = '{} Ridge Regression using {}{}, {} HRFlag'.format(subject, region_approach, 
                                                                  resolution, HRFlag_process) 
    
    output_file_name = '{}_RidgeRegression_{}{}_{}_HRFlag'.format(subject, region_approach,
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
    
    # defining parameter range
    param_grid = {'alpha': [0.1], # [0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                  'normalize': [True], # [True, False]
                  'solver': ['lsqr'] # ['auto','svd','cholesky','lsqr','sparse_cg','sag','saga','lbfgs']
                  }
    
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=0)
    
    t0 = time()
    grid = GridSearchCV(RidgeClassifier(random_state=0), param_grid, refit=True, 
                        n_jobs=-1, cv=cv, verbose=3)

    grid.fit(X_train, y_train)
    
    # how our model looks after hyper-parameter tuning
    print(colored(('Model best parameters found:\n'),'red',attrs=['bold']),grid.best_params_)
 
    # classification report
    grid_predictions = grid.predict(X_test)
    print(classification_report(y_test, grid_predictions))
    print("Ridge decoding time:", round(time()-t0, 3), "s")
    
    # confusion matrix
    cm_ridge = confusion_matrix(y_test, grid_predictions)
    model_cm = np.round(cm_ridge.astype('float') / cm_ridge.sum(axis=1)[:, np.newaxis], 2)
        
    visualization.conf_matrix(model_cm, unique_conditions, title, cm_results_outpath, 
                              output_file_name, decoder, subject, region_approach, 
                              resolution, HRFlag_process)
    
    
    
    
def _grid_bagging_decoder(all_modality_concat_bold, all_modality_concat_labels,
                          subject, decoder, region_approach, HRFlag_process,   
                          results_outpath, cm_results_outpath, resolution):
    
    """
    Bagged Decision Trees (Bagging) classifier with GridSearchCV.
    """

    title = '{} Bagging using {}{}, {} HRFlag'.format(subject, region_approach, 
                                                      resolution, HRFlag_process) 
    
    output_file_name = '{}_Bagging_{}{}_{}_HRFlag'.format(subject, region_approach,
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
    
    # defining parameter range
    param_grid = {'n_estimators': [1000], #[10,100,200,800,1000,2000]
                  'bootstrap': [False],#[True, False]
                  'max_features': [100], #[2,5,10,21,25,50,100,500,'auto','sqrt']
                  'bootstrap_features': [False],
                  'oob_score': [False],#[False, 1,2,4,8,10]
                  'warm_start': [True]#[True, False]
                 }
    
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=0)
    
    t0 = time()
    grid = GridSearchCV(BaggingClassifier(random_state=0), param_grid, refit=True, 
                        n_jobs=-1, cv=cv, verbose=3)

    grid.fit(X_train, y_train)
    
    # how our model looks after hyper-parameter tuning
    print(colored(('Model best parameters found:\n'),'red',attrs=['bold']),grid.best_params_)
 
    # classification report
    grid_predictions = grid.predict(X_test)
    print(classification_report(y_test, grid_predictions))
    print("Bagging decoding time:", round(time()-t0, 3), "s")
    
    # confusion matrix
    cm_bagging = confusion_matrix(y_test, grid_predictions)
    model_cm = np.round(cm_bagging.astype('float') / cm_bagging.sum(axis=1)[:, np.newaxis], 2)
        
    visualization.conf_matrix(model_cm, unique_conditions, title, cm_results_outpath, 
                              output_file_name, decoder, subject, region_approach, 
                              resolution, HRFlag_process)


    
                
def _grid_gaussian_nb_decoder(all_modality_concat_bold, all_modality_concat_labels,
                               subject, decoder, region_approach, HRFlag_process,   
                               results_outpath, cm_results_outpath, resolution):    
    
    """
    Gaussian Naive Bayes classifier with GridSearchCV.
    """

    title = '{} Gaussian Naive Bayes using {}{}, {} HRFlag'.format(subject, region_approach, 
                                                                    resolution, HRFlag_process) 
    
    output_file_name = '{}_GaussianNB_{}{}_{}_HRFlag'.format(subject, region_approach,
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
    
    # defining parameter range
    param_grid = {'var_smoothing': [0.0001] #np.logspace(0,-14, num=15)
#                   'priors': [None]
                 }
    
    
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=0)
    
    t0 = time()
    grid = GridSearchCV(GaussianNB(), param_grid, refit=True, 
                        n_jobs=-1, cv=cv, verbose=3)
    
    grid.fit(X_train, y_train)
    
    # print how our model looks after hyper-parameter tuning
    print(colored(('Model best parameters found:\n'),'red',attrs=['bold']),grid.best_params_)
    
    # classification report
    grid_predictions = grid.predict(X_test)
    print(classification_report(y_test, grid_predictions))
    print("GaussianNB decoding time:", round(time()-t0, 3), "s")
    
    # confusion matrix
    cm_nb = confusion_matrix(y_test, grid_predictions)
    model_cm = np.round(cm_nb.astype('float') / cm_nb.sum(axis=1)[:, np.newaxis], 2) 
        
    visualization.conf_matrix(model_cm, unique_conditions, title, cm_results_outpath, 
                              output_file_name, decoder, subject, region_approach, 
                              resolution, HRFlag_process)
        
    
    

def postproc_benchmark_decoder(subjects, modalities, decoders, region_approach, 
                               HRFlag_process, resolution): 
    
#    home_dir = '/home/srastegarnia/hcptrt_decoding_Shima/' # elm
    home_dir = '/home/rastegar/projects/def-pbellec/rastegar/hcptrt_decoding_shima/' # CC

    proc_data_path = home_dir + 'data/'
    
    for subject in subjects:
        
        results_outpath = home_dir + 'hcptrt-decoding/results/' \
                         '{}/{}/{}/{}/'.format(region_approach,
                                               resolution, subject, 
                                               HRFlag_process)
        
        cm_results_outpath = results_outpath + 'cm_results/'
        
        # remove previous content
        if os.path.exists(cm_results_outpath):
            files = glob.glob(os.path.join(cm_results_outpath, "*"))
            for f in files:
                os.remove(f)        

        if not os.path.exists(results_outpath):
            os.makedirs(results_outpath)        
    
        if not os.path.exists(cm_results_outpath):
            os.makedirs(cm_results_outpath)
            
        # create a .csv file of decoders results summary
        header = ['subject','decoder','region_approach', 'resolution', 'HRFlag_process',
                  'body0b', 'body2b', 'face0b', 'face2b', 'fear', 'footL', 'footR', 'handL',
                  'handR', 'match', 'math', 'mental', 'place0b', 'place2b', 'random',
                  'relational', 'shape', 'story', 'tongue', 'tool0b', 'tool2b']
        
        df = pd.DataFrame(list())
        df.to_csv(cm_results_outpath + 'results_summary.csv')
        
        with open(cm_results_outpath + 'results_summary.csv', 'w', encoding='UTF8') as rslt_smry:
            writer = csv.writer(rslt_smry)
            writer.writerow(header)
                
        results_summary_file = cm_results_outpath + 'results_summary.csv' 
        
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
        df_path = proc_data_path + 'medial_data/fMRI2/{}/{}/{}/{}_wm_fMRI2.npy'. format(region_approach, 
                                                                                         resolution, 
                                                                                         subject, subject) 
        df = np.load(df_path)
    #     print(df)
        parcel_no = int(len(df[0][:][1]))

        print('all_modality_concat_bold shape', np.shape(all_modality_concat_bold))
        print('all_modality_concat_labels shape', np.shape(all_modality_concat_labels),'\n')
        

        for decoder in decoders:

            if decoder == 'svm_nogrid':
                print(colored(('Support Vector Machine classifier:'),attrs=['bold']))
                _svm_decoder(all_modality_concat_bold=all_modality_concat_bold, 
                             all_modality_concat_labels=all_modality_concat_labels,
                             subject=subject, decoder=decoder,
                             region_approach=region_approach, 
                             HRFlag_process=HRFlag_process, 
                             results_outpath=results_outpath,
                             cm_results_outpath=cm_results_outpath,
                             resolution=resolution)

                
            elif decoder == 'skl_mlp': 
                print(colored(('Multi Layer Perceptron Neural Networks classifier with grid search'), attrs=['bold']))
                _grid_mlp_decoder(all_modality_concat_bold=all_modality_concat_bold, 
                                  all_modality_concat_labels=all_modality_concat_labels,
                                  subject=subject, decoder=decoder,
                                  region_approach=region_approach, 
                                  HRFlag_process=HRFlag_process,
                                  results_outpath=results_outpath,
                                  cm_results_outpath=cm_results_outpath,
                                  resolution=resolution)  
                                

            elif decoder == 'knn_nogrid': 
                print(colored(('K-Nearest Neighbours classifier'), attrs=['bold']))
                _knn_decoder(all_modality_concat_bold=all_modality_concat_bold, 
                             all_modality_concat_labels=all_modality_concat_labels,
                             subject=subject, decoder=decoder,
                             region_approach=region_approach,
                             HRFlag_process=HRFlag_process,
                             results_outpath=results_outpath,
                             cm_results_outpath=cm_results_outpath,
                             resolution=resolution)


            elif decoder == 'random_forest_nogrid': 
                print(colored(('Random Forest classifier'), attrs=['bold']))
                _random_forest_decoder(all_modality_concat_bold=all_modality_concat_bold, 
                                       all_modality_concat_labels=all_modality_concat_labels,
                                       subject=subject, decoder=decoder,
                                       region_approach=region_approach, 
                                       HRFlag_process=HRFlag_process,
                                       results_outpath=results_outpath,
                                       cm_results_outpath=cm_results_outpath,
                                       resolution=resolution)


            elif decoder == 'svm': 
                print(colored(('Support Vector Machine classifier with grid search'), attrs=['bold']))
                _grid_svm_decoder(all_modality_concat_bold=all_modality_concat_bold, 
                                  all_modality_concat_labels=all_modality_concat_labels,
                                  subject=subject, decoder=decoder,
                                  region_approach=region_approach, 
                                  HRFlag_process=HRFlag_process,
                                  results_outpath=results_outpath,
                                  cm_results_outpath=cm_results_outpath,
                                  resolution=resolution) 
                           

            elif decoder == 'mlp': 
                print(colored(('Multi Layer Perceptron Neural Networks classifier'\
                               '(two dense layers):'), attrs=['bold']))
                _mlp_decoder(all_modality_concat_bold=all_modality_concat_bold, 
                             all_modality_concat_labels=all_modality_concat_labels,
                             subject=subject, decoder=decoder,
                             region_approach=region_approach,
                             HRFlag_process=HRFlag_process,
                             results_outpath=results_outpath,
                             cm_results_outpath=cm_results_outpath,
                             resolution=resolution)
      


            elif decoder == 'knn': 
                print(colored(('K-Nearest Neighbor classifier with grid search'), attrs=['bold']))
                _grid_knn_decoder(all_modality_concat_bold=all_modality_concat_bold, 
                                  all_modality_concat_labels=all_modality_concat_labels,
                                  subject=subject, decoder=decoder,
                                  region_approach=region_approach, 
                                  HRFlag_process=HRFlag_process,
                                  results_outpath=results_outpath,
                                  cm_results_outpath=cm_results_outpath,
                                  resolution=resolution) 


            elif decoder == 'random_forest': 
                print(colored(('Random Forest classifier with grid search'), attrs=['bold']))
                _grid_random_forest_decoder(all_modality_concat_bold=all_modality_concat_bold, 
                                            all_modality_concat_labels=all_modality_concat_labels,
                                            subject=subject, decoder=decoder,
                                            region_approach=region_approach, 
                                            HRFlag_process=HRFlag_process,
                                            results_outpath=results_outpath,
                                            cm_results_outpath=cm_results_outpath,
                                            resolution=resolution)


            elif decoder == 'logistic_regression': 
                print(colored(('Logistic Regression classifier with grid search'), attrs=['bold']))
                _grid_logistic_regression_decoder(all_modality_concat_bold=all_modality_concat_bold, 
                                                  all_modality_concat_labels=all_modality_concat_labels,
                                                  subject=subject, decoder=decoder,
                                                  region_approach=region_approach, 
                                                  HRFlag_process=HRFlag_process,
                                                  results_outpath=results_outpath,
                                                  cm_results_outpath=cm_results_outpath,
                                                  resolution=resolution)


            elif decoder == 'ridge': 
                print(colored(('Ridge Regression classifier with grid search'), attrs=['bold']))
                _grid_ridge_decoder(all_modality_concat_bold=all_modality_concat_bold, 
                                    all_modality_concat_labels=all_modality_concat_labels,
                                    subject=subject, decoder=decoder,
                                    region_approach=region_approach, 
                                    HRFlag_process=HRFlag_process,
                                    results_outpath=results_outpath,
                                    cm_results_outpath=cm_results_outpath,
                                    resolution=resolution)


            elif decoder == 'bagging': 
                print(colored(('Bagged Decision Trees classifier with grid search'), attrs=['bold']))
                _grid_bagging_decoder(all_modality_concat_bold=all_modality_concat_bold, 
                                      all_modality_concat_labels=all_modality_concat_labels,
                                      subject=subject, decoder=decoder,
                                      region_approach=region_approach, 
                                      HRFlag_process=HRFlag_process,
                                      results_outpath=results_outpath,
                                      cm_results_outpath=cm_results_outpath,
                                      resolution=resolution)


            elif decoder == 'gaussian_nb': 
                print(colored(('Gaussian Naive Bayes classifier with grid search'), attrs=['bold']))
                _grid_gaussian_nb_decoder(all_modality_concat_bold=all_modality_concat_bold, 
                                           all_modality_concat_labels=all_modality_concat_labels,
                                           subject=subject, decoder=decoder,
                                           region_approach=region_approach, 
                                           HRFlag_process=HRFlag_process,
                                           results_outpath=results_outpath,
                                           cm_results_outpath=cm_results_outpath,
                                           resolution=resolution)


            else:
                print('The model is not defined')
                        
                        
                        
                      
