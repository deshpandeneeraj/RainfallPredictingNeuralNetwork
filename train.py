
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split
import os
from tensorflow.keras.models import model_from_json
import datetime

#Loading Clean Data
X = np.load('X.npy')
y = np.load('y.npy')

#Splitting into training and testing by 2:1 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=99)
#Designing CNN

batch_size = 128
epochs = 20

model = Sequential()

model.add(Dense(32, input_dim = 15, activation = 'relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Setting TensorBoard Directory
log_dir = "logs\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#Training Model
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[tensorboard_callback])
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))



model.save('model.h5')
model.save_weights('weights.h5')#

