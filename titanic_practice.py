# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 13:48:47 2019

@author: black
Legend:
-survived 0 died, 1 survived
-pclass = 1 upper economic class, 2 med, 3 low
-Embarkation C = Cherbourg, Q = Queenstown, S = Southampton 
-sibsp = #siblings on board, parch= # parents
"""
import pandas as pd
import matplotlib.pyplot as plt; import seaborn as sns
import copy
data = pd.read_csv('titanic_train.csv')
data_orig = copy.deepcopy(data)

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
    if verbose == 1:
        print(data.info())
    else:
        pass
    return data

print(data.isna().sum()) #check how many missing values we have, revealing problematic columns

data.set_index(['PassengerId'], inplace=True)
data = data.drop(columns=['Cabin', 'Name', 'Ticket' ])  #not informative in any way, plus Cabin misses a lot of values
data = to_categorical(data)

#Basic initial EDA
correl = data.corr()
data.Pclass = data.Pclass.map({1:'Upper', 2:'Medium', 3:'Poor'})  #rename Pclasses for EDA since numbers are not intuitive to me
sns.boxplot(x='Sex', y='Age', data=data,hue='Survived').set_title('Survivors by Age and Sex'); plt.show()
sns.heatmap(correl, annot=True); plt.tight_layout(); plt.show()

#Imputing NaN ages with the mean ages per Class, obtained visually and in table
sns.boxplot(x='Pclass', y='Age', data=data); plt.show()
print(data.groupby(by='Pclass').mean())

data.Age = data.apply(lambda row: 25 if row['Pclass'] == 'Poor' else (30 if 
                      row['Pclass'] == 'Medium' else 38), axis=1)

#Impute couple of missing values with most common Embarked station, won't change much either way
data.Embarked.fillna('C',inplace=True)
assert(data.isna().sum().sum() == 0)
 
#NEXT UP: More EDA and Pandas manipulation for answering quantitative Questions!


    
