import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix
from nilearn import plotting
from nilearn.plotting import plot_matrix
import seaborn as sn
#from csv import DictWriter
from csv import writer



def classifier_history(history, title, results_outpath, output_file_name):
    print(history.history.keys())
    
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(title + 'model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.savefig(results_outpath + output_file_name + '_modelـaccuracy.png', 
                dpi=300, bbox_inches='tight')
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(title + 'model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.savefig(results_outpath + output_file_name + '_modelـloss.png', 
                dpi=300, bbox_inches='tight')
    
    
    
def conf_matrix(model_cm, unique_conditions, title, cm_results_outpath, output_file_name, 
                decoder, subject, region_approach, resolution, HRFlag_process):
    
    df_cm = pd.DataFrame(model_cm, index=unique_conditions, 
                         columns=unique_conditions)
    
    # Adding decoding info to the summary results file
    cm_diag = np.diag(df_cm, k=0)
    cm_smry = np.append([subject, decoder, region_approach,
                         resolution, HRFlag_process], cm_diag)

    with open(cm_results_outpath + 'results_summary.csv', 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(cm_smry) 

    
    # Saving CM files
    df_cm.to_csv(cm_results_outpath + output_file_name + '.csv')
    plt.figure(figsize=(20,14))
    sn.heatmap(df_cm, annot=True, cmap='Blues', square = True)
    plt.xticks(rotation=45)
    plt.title(title , fontsize=15, fontweight='bold')
    plt.xlabel("true labels", fontsize=14, fontweight='bold')
    plt.ylabel("predicted labels", fontsize=14, fontweight='bold')
    plt.show()
     
    
    
    
def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):
    
    """
    Visualizing cross-validation behavior in scikit-learn. 
    Creates a sample plot for indices of a CV object.
    
    """
    
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



#------------------------------------------------------------------------------------------------------------    
def linear_decoder_weights(model_svm):
    coef_img = masker.inverse_transform(model_svm.coef_[0, :])
    plotting.view_img(coef_img, title="SVM weights map", dim=-1, 
                      resampling_interpolation='nearest') # bg_img=haxby_dataset.anat[0],




    
       