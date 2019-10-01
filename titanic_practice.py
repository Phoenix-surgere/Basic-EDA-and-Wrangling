# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 13:48:47 2019

@author: black

LEGEND:
Survived:  0 died, 1 survived
Pclass = 1 upper economic class, 2 med, 3 low
Embarkation C = Cherbourg, Q = Queenstown, S = Southampton 
Sibsp = #siblings/spouses on board, Parch= # parents/children on board
"""
import pandas as pd
import matplotlib.pyplot as plt; import seaborn as sns
import copy
from helper_funcs import to_categorical
data = pd.read_csv('titanic_train.csv')

print(data.isnull().sum()) #tests for NA entries
print("% of Cabin Null: {:.2f}".format((data.Cabin.count() / data.shape[0])*100)) #checks the % of NA values in this column

data.set_index(['PassengerId'], inplace=True)
data = data.drop(columns=['Cabin', 'Name', 'Ticket' ])
data = to_categorical(data)
data_numeric = copy.deepcopy(data)
data.Pclass = data.Pclass.map({1:'Upper', 2:'Medium', 3:'Poor'})
data.Survived = data.Survived.map({1:'Survived', 0:'Died'})

#Basic Histograms
sns.countplot(x='Sex', data=data).set_title('Passenger Sex Distribution'); plt.show()
sns.countplot(x='Pclass', data=data).set_title('Passenger Economic Class Distribution'); plt.show()
sns.countplot(x='Embarked', data=data).set_title('Passenger Embarking Station'); plt.show()
data.Fare.plot(kind='hist', bins=20, title='Fare Distribution', alpha=0.5);plt.show()
data.Parch.plot(kind='hist', bins=20, title='Parents/Children Distribution', alpha=0.5);plt.show()
data.SibSp.plot(kind='hist', bins=20, title='Siblings/Spouses Distribution', alpha=0.5);plt.show()

#Basic Descriptive boxplots statistics
sns.boxplot(x='Sex', y='Age', data=data,hue='Survived').set_title('Survivors by Age and Sex');plt.show()
sns.boxplot(x='Pclass', y='Age', data=data,hue='Survived').set_title('Survivors by Class and Age');plt.show()

sns.boxplot(x='Pclass', y='Age', data=data).set_title('Age by Economic Class'); plt.show()
sns.boxplot(x='Pclass', y='Fare', data=data).set_title('Fare by Economic Class'); plt.show()
print(data.groupby(by='Pclass').mean()) #displays average survivor age per class for imputation

conditions = [data['Pclass'] == 'Upper', data['Pclass']=='Medium', data['Pclass'] == 'Poor']
values = [37, 29, 24]
import numpy as np
data['Age'] = np.where(data['Age'].isnull(),
                              np.select(conditions, values),
                              data['Age'])

data.Embarked.fillna('C',inplace=True)
assert(data.isna().sum().sum() == 0)
data.rename(columns={'Parch': 'Parents/Children', 'SibSp': 'Siblings/Spouses'},inplace=True) 
hue_order = ['Died', 'Survived']

sns.set_style(style='dark')

group= data.groupby(by=['Sex', 'Survived']).Survived.count().unstack()
group.plot(kind='bar', stacked=True).set_title('Survivors by Sex')
plt.ylabel('# of Passengers')
plt.show()

print('*'*10)
group = data.groupby(by=['Pclass', 'Survived']).Pclass.count().unstack()
print(group)
group.plot(kind='bar', stacked=True,title='Survivors by Economic Status')
plt.show()

group = data.groupby(by=['Parents/Children', 'Survived'])['Parents/Children'].count().unstack()
print(group)
group.plot(kind='bar', stacked=True,title='Survivors by # of Parents/Children')
plt.show()

group = data.groupby(by=['Siblings/Spouses', 'Survived'])['Siblings/Spouses'].count().unstack()
print(group)
group.plot(kind='bar', stacked=True,title='Survivors by # of Siblings/Spouses')
plt.show()

group = data.groupby(by=['Embarked', 'Survived']).Embarked.count().unstack()
print(group)
group.plot(kind='bar', stacked=True,title='Survivors by Embarking Station')
plt.show()

#Mean of Age and Fare of Survivors
pivot = pd.pivot_table(data=data, index='Survived', values=['Age', 'Fare'])
print(pivot)

#Modelling
seed = 43
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RFC, GradientBoostingClassifier as GBC
from sklearn.linear_model import LogisticRegression as LR
from sklearn import metrics as m
import xgboost as xgb
from plot_confusion_matrix import plot_matrix
from helper_funcs import plot_roc_auc, plot_precision_recall
from sklearn.preprocessing import StandardScaler as SS

#Data preprocessing 
targets = data.Survived; targets = targets.map({'Died': 0, 'Survived':1})
data.drop(columns=['Survived'],inplace=True)
data.Pclass = data.Pclass.map({'Poor':1, 'Medium':2, 'Upper':3})
data_model = pd.get_dummies(data=data, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(data_model, targets,
                test_size=0.15, random_state=seed, stratify=targets)
#Model Instantiation
logit = LR(random_state=seed,solver='lbfgs',max_iter=300)
rf = RFC(n_estimators=250, random_state=seed)
gb = GBC(n_estimators=250, random_state=seed)
xgb = xgb.XGBClassifier(objective='reg:logistic', n_estimators=250, seed=42)
svm = SVC(random_state=seed,probability=True)

models = [logit, rf,gb,xgb, svm]
labels = ['Died', 'Survived']
scaler = SS(); X_train = scaler.fit_transform(X_train); X_test = scaler.transform(X_test)  #FOR SVM

def fit_metrics(model, Xtr, ytr, Xts, yts, labels):
    print(model.__class__.__name__ + ' Results:')
    model.fit(Xtr, ytr)
    cm = m.confusion_matrix(yts, model.predict(Xts))
    plot_matrix(cm, classes=labels, normalize=True,
    title='Confusion Matrix for Titanic Test Data'); plt.show()
    plot_roc_auc(yts, logit.predict_proba(Xts)[:,1])
    plot_precision_recall(yts, logit.predict_proba(Xts)[:,1])
    classification_metrics(yts,logit.predict(Xts))
    
#need to add Cross-Validation for more reliable results, here we get Bird's Eye view
for model in models:
    print('*'*25)
    fit_metrics(model, X_train, y_train, X_test, y_test, labels)
    print('*'*25)

 #fit_metrics(logit, X_train, y_train, X_test, y_test, labels)  #81.34% accuracy for logit

 dist_rf = {    "n_estimators"      : [100,250,500],
               "criterion"         : ["gini", "entropy"],
               "max_features"      : ['sqrt','log2',0.2,0.5,0.8],
               "max_depth"         : [3,4,6,10],
               "min_samples_split" : [2, 5, 20,50] }
grid = RSCV(rf, param_distributions=dist_rf,cv=5, scoring='accuracy', verbose=1, n_jobs=2, n_iter=200)
#fit_metrics(grid, X_train, y_train, X_test, y_test, labels)  #83.58% accuracy for logit

dist_gbc = {    "n_estimators"     : [100,250,500],
               "max_features"      : ['sqrt','log2',0.2,0.5,0.8],
               "max_depth"         : [3,4,6,10],
               "min_samples_split" : [2, 5, 20,50] } 
grid = RSCV(gb, param_distributions=dist_gbc, cv=5, scoring='accuracy', verbose=1, n_jobs=2, n_iter=200)
#fit_metrics(grid, X_train, y_train, X_test, y_test, labels)  #81.34% accuracy 

dist_xgb ={
         'n_estimators' : [100, 250, 500],
         'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
           }
grid = RSCV(xgb, param_distributions=dist_xgb, cv=5, scoring='accuracy', verbose=1, n_jobs=2, n_iter=200)
#fit_metrics(grid, X_train, y_train, X_test, y_test, labels)    #83.58% accuracy 
    
dist_svm = {
        'C':np.linspace(0.0001, 1, 100),
        'gamma': np.linspace(0.0001, 1,100) ,
        'kernel': ['linear', 'rbf']}        
grid_svm = RSCV(svm, param_distributions=dist_svm, cv=5, scoring='accuracy', verbose=1, n_jobs=2, n_iter=400)
#fit_metrics(grid_svm, X_train, y_train, X_test, y_test, labels)    #84.33% accuracy, best model

#No need to fit models prior to running the classifier, but since i did I gave more weight to SVM
votes = VC(estimators=[
('lr', logit), ('rf', grid_rf ) , ('gb', grid_gb), ('xgb', grid_xgb), ('svm', grid_svm) ],
voting='soft', weights=[1, 1, 1, 1, 1 ])
fit_metrics(votes, X_train, y_train, X_test, y_test, labels) #Accuracy 82.84 %
