# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 13:48:47 2019

@author: black
"""
# survived 0 died, 1 survived
#pclass = 1 upper economic class, 2 med, 3 low
#Embarkation C = Cherbourg, Q = Queenstown, S = Southampton 
#sibsp = #siblings on board, parch= # parents

import pandas as pd
import matplotlib.pyplot as plt; import seaborn as sns
import copy
data = pd.read_csv('titanic_train.csv')
data_orig = copy.deepcopy(data)
#y = data.Survived
#X = data.drop(columns=['Survived', 'Name'])
#X.set_index(['PassengerId'], inplace=True)

def to_categorical(data):
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
    print(data.info())
    return data

#print("% of Cabin non Null: {:.2f}".format((X.Cabin.count() / X.shape[0])*100))

data.set_index(['PassengerId'], inplace=True)
data = data.drop(columns=['Cabin', 'Name', 'Ticket' ])
data = to_categorical(data)
correl = data.corr()
data.Pclass = data.Pclass.map({1:'Upper', 2:'Medium', 3:'Poor'})
#sns.boxplot(x='Sex', y='Age', data=data,hue='Survived').set_title('Survivors by Age and Sex')
#plt.show()
#sns.heatmap(correl, annot=True); plt.tight_layout(); plt.show()

print(data.isnull().sum()) 
sns.boxplot(x='Pclass', y='Age', data=data); plt.show()
print(data.groupby(by='Pclass').mean())

data.Age = data.apply(lambda row: 25 if row['Pclass'] == 'Poor' else (30 if 
                      row['Pclass'] == 'Medium' else 38), axis=1)

data.Embarked.fillna('C',inplace=True)
assert(data.isna().sum().sum() == 0)
 
#EDA for older people
#old = data.loc[data.Age > 35]
#old.rename(columns={'Parch': 'Parents/Children', 'SibSp': 'Siblings/Spouses'},inplace=True) 
##sns.violinplot(x='Survived', y='Age', data=old, inner=None)#;plt.show()
##sns.swarmplot(x='Survived', y='Age', hue='Sex',data=old) ;plt.show()
##sns.jointplot(x='Survived', y='Age', kind='kde',data=old); plt.show()

#ALSO sns.countplot() to count categories 
#EDA for younger people
#young = data.loc[data.Age <= 35]


    