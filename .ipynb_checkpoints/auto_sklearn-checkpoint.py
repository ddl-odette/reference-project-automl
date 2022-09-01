import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import pandas as pd
import numpy as np
import subprocess

from joblib import dump, load
import json


#load and clean the data----------------------

#column names
names = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang', \
         'oldpeak','slope','ca','thal','target']

#load data from Domino project directory
hd_data = pd.read_csv("./data/raw/heart.csv", header=None, names=names)

#in case some data comes in as string
#convert to numeric and coerce errors to NaN
for col in hd_data.columns:  # Iterate over chosen columns
    hd_data[col] = pd.to_numeric(hd_data[col], errors='coerce')
    
#drop nulls
hd_data.dropna(inplace=True)

#non-ohe data---------------------------------
   
#load the X and y set as a numpy array
X_hd = hd_data.drop('target', axis=1).values
y_hd = hd_data['target'].values

#build the train and test sets
X_hd_train, X_hd_test, y_hd_train, y_hd_test = \
    sklearn.model_selection.train_test_split(X_hd, y_hd, random_state=12)

#now do ohe-----------------------------------

#function to do one hot encoding for categorical columns
def create_dummies(data, cols, drop1st=True):
    for c in cols:
        dummies_df = pd.get_dummies(data[c], prefix=c, drop_first=drop1st)  
        data=pd.concat([data, dummies_df], axis=1)
        data = data.drop([c], axis=1)
    return data

cat_cols = ['cp', 'restecg', 'slope', 'ca', 'thal']
hd_data = create_dummies(hd_data, cat_cols)
    
#load the X and y set as a numpy array
X_hd_ohe = hd_data.drop('target', axis=1).values
y_hd_ohe = hd_data['target'].values

#build the train and test sets
X_hd_ohe_train, X_hd_ohe_test, y_hd_ohe_train, y_hd_ohe_test = \
    sklearn.model_selection.train_test_split(X_hd_ohe, y_hd_ohe, \
                                             random_state=12)

automl_hd_ohe_p = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=120,
    per_run_time_limit=60,
    disable_evaluator_output=False,
    resampling_strategy='holdout',
    resampling_strategy_arguments={'train_size': 0.67},
    
    #turn on parallelization
    n_jobs=4,
    seed=5,
)

#call it
automl_hd_ohe_p.fit(X_hd_ohe_train, y_hd_ohe_train, dataset_name='heart_disease')

#save the predicitons
predictions_hd_ohe_p = automl_hd_ohe_p.predict(X_hd_ohe_test)

print('Accuracy:')
print(sklearn.metrics.accuracy_score(y_hd_ohe_test, predictions_hd_ohe_p))
print(' ')
print('-----------------------------------------')
print(' ')
print('Sprint Stats:')
print(automl_hd_ohe_p.sprint_statistics())

dump(automl_hd_ohe_p, 'automl_hd_ohe_p.joblib')

hd_acc = sklearn.metrics.accuracy_score(y_hd_ohe_test, predictions_hd_ohe_p)

with open('dominostats.json', 'w') as f:
    f.write(json.dumps( {"HD_ACC": hd_acc, "BC_ACC": bc_acc}))