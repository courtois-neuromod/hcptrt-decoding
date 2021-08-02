import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix
from nilearn.plotting import plot_matrix
import seaborn as sn


def classifier_history(history):
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    
def conf_matrix(model_conf_matrix, unique_conditions, title):
    
    df_cm = pd.DataFrame(model_conf_matrix, index=unique_conditions, columns=unique_conditions)
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, cmap='Blues', square=True)
    plt.xticks(rotation = 45)
    plt.title(title, fontsize=15, fontweight='bold')
    plt.xlabel("true labels", fontsize= 12, fontweight='bold')
    plt.ylabel("predicted labels", fontsize=12, fontweight='bold')