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
import numpy as np

data = pd.read_csv('titanic_train.csv')

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
data.Pclass = data.Pclass.map({1:'Upper', 2:'Medium', 3:'Poor'})  #rename Pclasses for EDA since numbers are not intuitive to me
sns.boxplot(x='Sex', y='Age', data=data,hue='Survived').set_title('Survivors by Age and Sex'); plt.show()
plt.legend(['Died', 'Survived'])
plt.show()

#Imputing NaN ages with the mean ages per Class, obtained visually and in table
sns.boxplot(x='Pclass', y='Age', data=data); plt.show()
print(data.groupby(by='Pclass').mean())
conditions = [data['Pclass'] == 'Upper', data['Pclass']=='Medium', data['Pclass'] == 'Poor']
values = [37, 29, 24]
data['Age'] = np.where(data['Age'].isnull(),
                              np.select(conditions, values),
                              data['Age'])


#Impute couple of missing values with most common Embarked station, won't change much either way
data.Embarked.fillna('C',inplace=True)
assert(data.isna().sum().sum() == 0)
data.rename(columns={'Parch': 'Parents/Children', 'SibSp': 'Siblings/Spouses'},inplace=True) 
data.Survived = data.Survived.map({1:'Survived', 0:'Died'})
hue_order = ['Died', 'Survived']

#It plots the ROWS as individual bars, hence cleaner that way
group0= data.groupby(by=['Sex', 'Survived']).Survived.count().unstack()
group0.plot(kind='bar', stacked=True).set_title('Survivors by Sex')
plt.ylabel('# of Passengers')
plt.show()

print('*'*10)
group = data.groupby(by=['Pclass', 'Survived']).Pclass.count().unstack()
group.plot(kind='bar', stacked=True).set_title('Survivors by Economic Status')
plt.show()

group = data.groupby(by=['Parents/Children', 'Survived'])['Parents/Children'].count().unstack()
group.plot(kind='bar', stacked=True).set_title('Survivors by # of Parents/Children')
plt.show()

group = data.groupby(by=['Siblings/Spouses', 'Survived'])['Siblings/Spouses'].count().unstack()
group.plot(kind='bar', stacked=True).set_title('Survivors by # of Siblings/Spouses')
plt.show()

group = data.groupby(by=['Embarked', 'Survived']).Embarked.count().unstack()
group.plot(kind='bar', stacked=True).set_title('Survivors by Embarking Station')
plt.show()

#NEXT UP: More EDA and Pandas manipulation for answering quantitative Questions!


    
