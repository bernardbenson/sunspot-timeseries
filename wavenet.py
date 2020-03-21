#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 18:30:06 2020

"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

#load data
df = pd.read_csv(r'/home/exx/data/Latest_SunspotNo.csv')

def univariate_data(dataset, target, start_index, end_index, history_size, target_size, step, single_step=False):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i, step)
    data.append(np.reshape(dataset[indices], (int(history_size/step), 1)))
    
    if single_step:
      labels.append(target[i+target_size])
    else:
      labels.append(target[i:i+target_size])

  return np.array(data), np.array(labels)

TRAIN_SPLIT = 3251-131 #3251 is total number of records in data. Can change according to dataset
BATCH_SIZE = 32
BUFFER_SIZE = 3251-131
tf.random.set_seed(13)
uni_data = df['Sunspot_No']
uni_data.index = df['Date']
uni_data.head()

uni_data.plot(subplots=True)
date = uni_data.index
uni_data = uni_data.values

#Normalize data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (0, 1))

uni_data = scaler.fit_transform(uni_data.reshape(len(uni_data), 1))

#528 input time steps
univariate_past_history = 528
STEP = 1
#forecast horizon
future_target = 132
x_train_uni_multi, y_train_uni_multi = univariate_data(uni_data, uni_data[:,], 0,
                                                 TRAIN_SPLIT, univariate_past_history,
                                                 future_target, STEP)
x_val_uni_multi, y_val_uni_multi = univariate_data(uni_data, uni_data[:,],
                                             TRAIN_SPLIT, None, univariate_past_history,
                                             future_target, STEP) 

#split data
x_train = x_train_uni_multi[:1085,:,:]
y_train = y_train_uni_multi[:1085,:,:]
x_val = x_train_uni_multi[1085:,:,:]
y_val = y_train_uni_multi[1085:,:,:]



def create_time_steps(length):
  time_steps = []
  for i in range(-length, 0, 1):
    time_steps.append(i)
  return time_steps

def show_plot(plot_data, delta, title):
  labels = ['History', 'True Future', 'Model Prediction']
  marker = ['.-', 'rx', 'go']
  time_steps = create_time_steps(plot_data[0].shape[0])
  if delta:
    future = delta
  else:
    future = 0

  plt.title(title)
  for i, x in enumerate(plot_data):
    if i:
      plt.plot(future, plot_data[i], marker[i], markersize=10,
               label=labels[i])
    else:
      plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
  plt.legend()
  plt.xlim([time_steps[0], (future+5)*2])
  plt.xlabel('Time-Step')
  return plt


def baseline(history):
  return np.mean(history)

def plot_train_history(history, title):
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(loss))

  plt.figure()

  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title(title)
  plt.legend()

  plt.show()


#tensorflow dataset format
train_data_uni_multi = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data_uni_multi = train_data_uni_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_uni_multi = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_data_uni_multi = val_data_uni_multi.batch(BATCH_SIZE).repeat()

def multi_step_plot(history, true_future, prediction):
  plt.figure(figsize=(12, 6))
  num_in = create_time_steps(len(history))
  num_out = len(true_future)

  plt.plot(num_in, np.array(history[:,]), label='History')
  plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
           label='True Future')
  if prediction.any():
    plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
             label='Predicted Future')
  plt.legend(loc='upper left')
  plt.show()
  
for x, y in train_data_uni_multi.take(1):
  multi_step_plot(x[0], y[0], np.array([0]))

    
EVALUATION_INTERVAL = 200
EPOCHS = 100

#WaveNet model
class GatedActivationUnit(tf.keras.layers.Layer):
    def __init__(self, activation="tanh", **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
    def call(self, inputs):
        n_filters = inputs.shape[-1] // 2
        linear_output = self.activation(inputs[..., :n_filters])
        gate = tf.keras.activations.sigmoid(inputs[..., n_filters:])
        return self.activation(linear_output) * gate
    
def wavenet_residual_block(inputs, n_filters, dilation_rate):
    z = tf.keras.layers.Conv1D(2 * n_filters, kernel_size=2, padding="causal",
                            dilation_rate=dilation_rate)(inputs)
    z = GatedActivationUnit()(z)
    z = tf.keras.layers.Conv1D(n_filters, kernel_size=1)(z)
    return tf.keras.layers.Add()([z, inputs]), z

n_layers_per_block = 10
n_blocks = 1
n_filters = 64
n_outputs = 132

inputs = tf.keras.layers.Input(shape=x_train_uni_multi.shape[-2:])
z=tf.keras.layers.Conv1D(n_filters, kernel_size=3, padding="causal")(inputs)
skip_to_last=[]
for dilation_rate in[2**i for i in range(n_layers_per_block)] * n_blocks:
    z, skip = wavenet_residual_block(z, n_filters, dilation_rate)
    skip_to_last.append(skip)
 
z = tf.keras.activations.relu(tf.keras.layers.Add()(skip_to_last))
z = tf.keras.layers.BatchNormalization()(z)
z = tf.keras.layers.Conv1D(n_filters, kernel_size=1, activation="relu")(z)
z = tf.keras.layers.Conv1D(n_outputs, kernel_size=1, activation="relu")(z)
Y_proba = tf.keras.layers.LSTM(132)(z)
model = tf.keras.models.Model(inputs=[inputs], outputs=[Y_proba])
model.summary()

model.compile(loss="mse", optimizer="adam", metrics=["mae"])
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, restore_best_weights=True)
#mc = tf.keras.callbacks.ModelCheckpoint('/home/exx/Documents/Bernard/LSTM/Best/best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)


#multi_step_history = model.fit(x_train, y_train, epochs=EPOCHS,
#                                          validation_data=(x_val, y_val), callbacks=[es])
  
multi_step_history = model.fit(x_train, y_train, epochs=EPOCHS)
  
#multi_step_history = multi_step_model.fit(train_data_uni_multi, epochs=EPOCHS,
#                                          steps_per_epoch=int(2591/32))

history = multi_step_history.history

#evaluation = model.evaluate(x_val, y_val)
#val_evl = scaler.inverse_transform(evaluation[0].reshape(1,-1))
#
#plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')

for x, y in train_data_uni_multi.take(10):
  multi_step_plot(x[0], y[0], model.predict(x)[0])
  
cycle_22_input = uni_data[-924:-396,]
prediction_0 = model.predict(np.reshape(cycle_22_input, (1, x_train_uni_multi.shape[1],1)))

prediction_0 = np.reshape(prediction_0, (prediction_0.shape[1],1))

#predicting future cycles
cycle_22_future = scaler.inverse_transform(prediction_0)
plt.plot(prediction_0, label="prediction")
plt.plot(uni_data[-396:-264,], label="actual")
plt.legend(loc="upper left")
plt.show()
  
cycle_23_input = uni_data[-792:-264,]
prediction_1 = model.predict(np.reshape(cycle_23_input, (1, x_train_uni_multi.shape[1],1)))

prediction_1 = np.reshape(prediction_1, (prediction_1.shape[1],1))

#future = (prediction * uni_train_std) + uni_train_mean
#future =  (prediction *(uni_data.max() - uni_data.min()) ) + uni_data.min()
cycle_23_future = scaler.inverse_transform(prediction_1)
plt.plot(prediction_1, label="prediction")
plt.plot(uni_data[-264:-132,], label="actual")
plt.legend(loc="upper left")
plt.show()



cycle_24_input = uni_data[-660:-132,]
prediction_2 = model.predict(np.reshape(cycle_24_input, (1, x_train_uni_multi.shape[1],1)))

prediction_2 = np.reshape(prediction_2, (prediction_2.shape[1],1))

#future = (prediction * uni_train_std) + uni_train_mean
#future =  (prediction *(uni_data.max() - uni_data.min()) ) + uni_data.min()
cycle_24_future = scaler.inverse_transform(prediction_2)
plt.plot(prediction_2, label="prediction")
plt.plot(uni_data[-132:,], label="actual")
plt.legend(loc="upper left")
plt.show()

cycle_25_input = uni_data[-528:,]
prediction = model.predict(np.reshape(cycle_25_input, (1, x_train_uni_multi.shape[1],1)))

prediction = np.reshape(prediction, (prediction.shape[1],1))
cycle_25_future = scaler.inverse_transform(prediction)
plt.plot(cycle_25_future, label="prediction")
plt.show()
