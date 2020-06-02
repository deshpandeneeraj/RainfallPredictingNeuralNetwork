import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split
import os
from tensorflow.keras.models import model_from_json


dataset = pd.read_csv('weatherAUS.csv')


#Cleaning Data
    #Removing unnecessary and non-numeric colummns
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



np.save('X.npy', X)
np.save('y.npy', y)
