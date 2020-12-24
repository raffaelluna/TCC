"""
Created on Tue Sep 18 15:26:49 2018

@author: Raffael
"""
import numpy as np

from keras import optimizers
from keras import regularizers
from keras.models import Model, Sequential
from keras.layers.core import Lambda
from keras.layers.merge import Concatenate, concatenate
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input, BatchNormalization
from keras.optimizers import SGD, Adadelta, Adam, RMSprop

import tensorflow as tf
from keras import backend as k

root = './csiq_video_database/'
random.seed(123)

def min_max_pool2d(x, data_format):
    if data_format == 'channels_first':
        max_x =  k.pool2d(x, pool_size=(x.shape[2], x.shape[3]), data_format=data_format)
        min_x = -k.pool2d(-x, pool_size=(x.shape[2], x.shape[3]), data_format=data_format)
        concat = Concatenate(axis = 1) #CHANNELS FIRST
    elif data_format == 'channels_last':
        max_x =  k.pool2d(x, pool_size=(x.shape[1], x.shape[2]), data_format=data_format)
        min_x = -k.pool2d(-x, pool_size=(x.shape[1], x.shape[2]), data_format=data_format)
        concat = Concatenate(axis = -1) #CHANNELS FIRST

    return concat([max_x, min_x])  # concatenate on channel

def min_max_pool2d_output_shape(input_shape, data_format):
    if data_format == 'channels_first':
        shape = list(input_shape)
        shape[1] = shape[1]*2
        shape[2] = 1
        shape[3] = 1
    elif data_format == 'channels_last':
        shape = list(input_shape)
        shape[1] = 1
        shape[2] = 1
        shape[3] = shape[3]*2

    return tuple(shape)

def little_convnet(img_rows, img_cols, of_type, lrate, data_format):

    print('initializing model...')

    if data_format == 'channels_first':
        if of_type == 'mod':
            shape = (5, img_rows, img_cols)
        elif of_type == 'comp':
            shape = (10, img_rows, img_cols)
    elif data_format == 'channels_last':
        if of_type == 'mod':
            shape = (img_rows, img_cols, 5)
        elif of_type == 'comp':
            shape = (img_rows, img_cols, 10)

    #INPUTS: BATCH OF STACKS (channels, img_rows, img_cols)
    inputs = Input(shape = shape, name='Stacks_inputs')

    #CONVOLUTIONAL LAYER
    x = Conv2D(50, (7,7), data_format=data_format, name='Conv_1')(inputs)
    x = Activation('relu', name='ReLU_1')(x)
    # x = Conv2D(32, (7,7), data_format='channels_first')(x)
    # x = Activation('relu')(x)

    #POOLING LAYER
    # x = MaxPooling2D((2,2), strides=(2,2))(x)
    # x = Activation('linear')(x)
    x = Lambda(min_max_pool2d, output_shape=min_max_pool2d_output_shape, name='Min-max_pooling')(x)
    x = Activation('linear', name='Linear_1')(x)

    #FULL-CONNECTED: REGRESSION

    x = Flatten()(x)
    x = Dense(800, activation='relu', name='FC_1')(x)
    # x = Dropout(0.5, name='Dropout_1')(x)
    x = Dense(800, activation='relu', name='FC_2')(x)
    x = Dropout(0.5, name='Dropout_2')(x)

    #OUTPUT: PREDICTIONS
    x = Dense(1)(x)
    predictions = Activation('sigmoid', name='Output_Score')(x)

    #MODEL
    model = Model(input = inputs, output = predictions)
    sgd = SGD(lr=0.001, decay=0.0004, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='mse', metrics=['mae'])

    return model
