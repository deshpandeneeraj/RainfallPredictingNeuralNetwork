# RainfallPredictingNeuralNetwork
Simple Neural Network Using Keras in Python to predict if it will rain the next day depending on today's factors


A simple implementation of a 2 layered Neural Network developed using Keras package that predicts the rainfall outcome of the next day.
Both layers are Dense layers from tf.keras.layers module

Architecture:
![Architecture](https://user-images.githubusercontent.com/45695989/83488953-50416200-a4cb-11ea-9de8-5b841aa313b7.png)

Dataset used:
  Weather in Australia Dataset was used for training the model.
  Link : https://www.kaggle.com/jsphyg/weather-dataset-rattle-package
  
  Data is cleaned by data.py script and cleaned np array files (.npy) are generated


Model was trained for 20 epochs over the dataset.
Accuracy and Loss scalars:
![Accuracy](https://user-images.githubusercontent.com/45695989/83488844-225c1d80-a4cb-11ea-976d-9a8e720a5eb7.png)
![Loss](https://user-images.githubusercontent.com/45695989/83488852-24be7780-a4cb-11ea-82ca-cace7562fe11.png)
