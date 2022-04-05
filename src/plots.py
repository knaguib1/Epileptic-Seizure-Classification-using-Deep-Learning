import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

"""
    File name: plots.py
    Author: Kareem Naguib
    Date created: 12/01/2021
    Contact: knaguib1@gmail.com, knaguib3@gatech.edu
"""

def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
    xx_train = np.arange(len(train_losses))
    xx_valid = np.arange(len(valid_losses))
    
    fig, ax = plt.subplots(1, 2, figsize = (16, 6))
    
    # plot loss
    ax[0].plot(xx_train, train_losses, label='Train')
    ax[0].plot(xx_valid, valid_losses, label='Valid')
    ax[0].set(xlabel = 'epoch', ylabel = 'Loss')
    ax[0].legend(loc='best')
    ax[0].title.set_text('Loss Curve')
    
    # plot accuracy
    ax[1].plot(xx_train, train_accuracies, label='Train')
    ax[1].plot(xx_valid, valid_accuracies, label='Valid')
    ax[1].set(xlabel = 'epoch', ylabel = 'Accuracy')
    ax[1].legend(loc='best')
    ax[1].title.set_text('Accuracy Curve')
    
    plt.show()
    fig.savefig('Learning curves.png')


def plot_confusion_matrix(results, class_names):
    
    y_true = [y_true for y_true, _ in results]
    y_pred = [y_pred for _, y_pred in results]
    
    # confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    
    # display
    if len(class_names) > 0:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                        display_labels=class_names)
    else:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                   
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix')
    plt.savefig('Confusion Matrix.png')
    plt.show()
    
    
