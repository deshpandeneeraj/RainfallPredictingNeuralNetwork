
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split
import os
##from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn import metrics

dataset = pd.read_csv('weatherAUS.csv')


dataset.drop(["Date", "Location", 'WindGustDir',	'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm'], axis = 1, inplace = True)
    #Cleaning Nan and Yes No collumns
yes_no_map = {'Yes': 1, 'No': 0, 0:0}
dataset['RainToday'] = dataset['RainToday'].fillna(0)
dataset['RainToday'] = dataset['RainToday'].map(yes_no_map).astype(int)
dataset['RainTomorrow'] = dataset['RainTomorrow'].map(yes_no_map)

dataset = dataset.fillna(0)

    #Min Max Normalizing Numeric Columns
dataset = (dataset-dataset.min())/(dataset.max()-dataset.min())

# print(dataset)

X = dataset.iloc[:,:15]
y = dataset["RainTomorrow"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=99)


model = load_model('model.h5')
print('Model Loaded')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print('Model Compiled')
print('\n\n\n')
print(X_test)

predictions = np.array(model.predict_classes(X_test))
print('\n\n\n PREDICTIONS MADE \n\n\n')
print(predictions)

##crosstable = pd.crosstab(y_test, predictions, margins=True)
conf_matrix = confusion_matrix(y_test, predictions)
##print('CROSSTABLE \n ', crosstable)
print('\n\nCONF MATRIX \n', conf_matrix) 
print(metrics.classification_report(y_test, predictions, labels=[0, 1]))
