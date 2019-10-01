# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 16:36:27 2019

@author: black

This is part of the original helper functions file used in other scripts as well, including only the parts used in the Titanic
dataset workflow, to avoid clutter. 
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_curve, precision_recall_curve as prc, log_loss 
from sklearn.metrics import accuracy_score as acc, classification_report as cl
from sklearn.model_selection import KFold

def to_categorical(data, verbose=0):
    """
    Returns all categorical dtypes from all object ones for efficiency reasons
    """
    categorize_label = lambda x: x.astype('category')
    categorical_feature_mask = data.dtypes == object
    categorical_columns = data.columns[categorical_feature_mask].tolist()
    LABELS = categorical_columns
    #Convert df[LABELS] to a categorical type
    data[LABELS] = data[LABELS].apply(categorize_label, axis=0)
    #print(data[LABELS].dtypes)
    if verbose == 1:
        print(data.info())
    else:
        pass
    return data

def plot_roc_auc(labels, label_probs):
    #ROC-AUC Curve => Best for balanced data
    fpr, tpr, thresholds = roc_curve(labels, label_probs)
    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('ROC curve for diabetes classifier')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.grid(True)
    plt.show()
    
def plot_precision_recall(labels, label_probs):
    #Recall-precision curve => Best for less balanced data
    precision, recall, thresholds = prc(labels, label_probs)
    plt.plot(recall, precision)
    plt.plot([0, 1], [0.5, 0.5], linestyle='--')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.grid(True)
    plt.show()
    
def classification_metrics(labels, preds):
    print('--'*30)
    print(cl(labels, preds))
    print('--'*30)
    print('Accuracy', round(acc(labels, preds) * 100,2), '%')
    print('Log Loss', round(log_loss(labels, preds) * 100,4), '%')

    
