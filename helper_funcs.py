# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 16:36:27 2019

@author: black
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

def forest_feature_importances(df, feat_imp, n_features=4):
    '''Plots static Random Forest feature importances'''
    feat_imp = dict(zip(df.columns, feat_imp.round(2)))
    feat_imp_df = pd.Series(data=feat_imp)
    feat_imp_df = feat_imp_df.sort_values(ascending=False).iloc[0:n_features]
    feat_imp_df.plot(kind='bar', title='Feature Importances: Top {}'.format(n_features)) 
    plt.show()
    return feat_imp_df

#rf_imp = rf_feature_importances(X_train, forest.feature_importances_)

def model_reduce(estimator, n_features, X,y, verbose=1):
    '''Single model RFE Fitting'''
    rfe = RFE(estimator=estimator, n_features_to_select=n_features)
    rfe.fit(X, y)
    rf_mask = rfe.support_
    if verbose == 1:
        rfe_best_features(estimator, X, rfe)
    else:
        pass
    return rf_mask

def rfe_best_features(model, data, rfe):
    '''Lower ranking= Better'''
    model_name = model.__class__.__name__
    rfe_order = pd.Series(dict(zip(data.columns, rfe.ranking_))).sort_values()
    rfe_order.rename_axis(model_name,inplace=True)
    print('\n', rfe_order)

    
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

    
def test_mean_target_encoding(train, test, target, categorical, alpha=5):
    # Calculate global mean on the train data
    global_mean = train[target].mean()
    
    # Group by the categorical feature and calculate its properties
    train_groups = train.groupby(categorical)
    category_sum = train_groups[target].sum()
    category_size = train_groups.size()
    
    # Calculate smoothed mean target statistics
    train_statistics = (category_sum + global_mean * alpha) / (category_size + alpha)
    
    # Apply statistics to the test data and fill new categories
    test_feature = test[categorical].map(train_statistics).fillna(global_mean)
    return test_feature.values

def train_mean_target_encoding(train, target, categorical, alpha=5):
    # Create 5-fold cross-validation
    kf = KFold(n_splits=5, random_state=123, shuffle=True)
    train_feature = pd.Series(index=train.index)
    
    # For each folds split
    for train_index, test_index in kf.split(train):
        cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]
      
        # Calculate out-of-fold statistics and apply to cv_test
        cv_test_feature = test_mean_target_encoding(cv_train, cv_test, target, categorical, alpha)
        
        # Save new feature for this particular fold
        train_feature.iloc[test_index] = cv_test_feature       
    return train_feature.values

def mean_target_encoding(train, test, target, categorical, alpha=5):
  
    # Get the train feature
    train_feature = train_mean_target_encoding(train, target, categorical, alpha)
  
    # Get the test feature
    test_feature = test_mean_target_encoding(train, test, target, categorical, alpha)
    
    # Return new features to add to the model
    return train_feature, test_feature

    
def plot_loss_metric(history):
    ''' Plots Loss and chosen metric(s) history for Neural Networks'''
    # plot loss during training
    plt.subplot(211)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    # plot accuracy during training
    plt.subplot(212)
    plt.title('Accuracy')
    plt.plot(history.history['acc'], label='train')
    plt.plot(history.history['val_acc'], label='test')
    plt.legend()
    plt.show()