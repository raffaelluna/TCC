# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 15:26:49 2018

@author: Raffael
"""

import os
import math
import yaml

import scipy.io as sio
from math import floor, ceil

from model import little_convnet
from utils import create_lists, block_shuffle_mod, list_random, batch_mod_generator, get_data_mod

from argparse import ArgumentParser

from keras import callbacks
from keras.callbacks import LearningRateScheduler

import tensorflow as tf
from keras import backend as k

root = './csiq_video_database/'
random.seed(123)

###################################
# TensorFlow wizardry
config = tf.ConfigProto()

# Select which GPU will use # 0 = Quadro P6000 24Gb GPU, 1 = Gtx 1080 8Gb GPU
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True

# Only allow a total of half the GPU memory to be allocated
#config.gpu_options.per_process_gpu_memory_fraction = 0.45

# Create a session with the above options specified.
k.tensorflow_backend.set_session(tf.Session(config=config))
###################################

def step_decay(epoch):
	initial_lrate = 0.001
	drop = 0.5
	epochs_drop = 100.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

def batch_training_mod(nb_epochs, batch_size, txt_file, res, train_size, size, step, of_type, val_split, dataset_split, exp_name, data_format):

    dir = exp_name+"_"+batch_size+"_"+nb_epochs+"_"+res+"_"+of_type+"/"

    if not os.path.isdir(dir):
        os.mkdir(dir)

    if res == '832x480':
        img_rows = 480
        img_cols = 832

    if res == '640x360':
        img_rows = 360
        img_cols = 640

    train_mod, val_mod, test_mod, video_names, video_scores, video_names_train, video_names_val, video_names_test = create_lists(txt_file, 'mod', res, train_size, val_split, dataset_split)

    sio.savemat(dir+"trainvid_"+of_type+"_"+res+"_"+exp_name+".mat", {"Train Name":video_names_train})
    sio.savemat(dir+"valvid_"+of_type+"_"+res+"_"+exp_name+".mat", {"Val Name":video_names_val})
    sio.savemat(dir+"testvid_"+of_type+"_"+res+"_"+exp_name+".mat", {"Test Name":video_names_test})

    model = little_convnet(img_rows, img_cols, of_type, lrate = 0.1, data_format)
    #model.load_weights("weights30_"+res+".h5")
    print(model.summary())

    lrate = LearningRateScheduler(step_decay)
    callbacks_list = [lrate]

    results = []
    mse_val = []
    for epoch in range(1+20,nb_epochs+1+20):

        train_mod = block_shuffle_mod(train_mod, size, video_names_train)
        video_names_train = list_random(video_names_train)

        print('\n**************** Epoca %.0f/' %(epoch)+ '%.0f ****************\n' %(nb_epochs+20))
        gen_train = batch_mod_generator(train_mod, video_names, video_names_train, video_scores, size, step, img_cols, img_rows, batch_size, 'train')

        print('training...\n')
        for X, Y in gen_train:
            history = model.fit(X, Y,  epochs = 1, verbose = 1, batch_size=batch_size, callbacks=callbacks_list)
        results.append(history.history)

        gen_val = batch_mod_generator(val_mod, video_names, video_names_val, video_scores, size, step, img_cols, img_rows, batch_size, 'val')

        print('validating...')
        for X_val, Y_val in gen_val:
            mse_val.append(model.evaluate(X_val, Y_val))

        if (epoch % 5) == 0:
            print('saving weights...')
            name = "weights%d_"+res+"_"+of_type+"_"+exp_name+"".h5" % epoch
            model.save_weights(dir+name)

    print('testing...')
    gen_test = batch_mod_generator(test_mod, video_names, video_names_test,
                                    video_scores, size, step, img_cols, img_rows,
                                    batch_size, 'test')

    score_mse = []
    scores = []
    for X_test, Y_test in gen_test:
        score_mse.append(model.evaluate(X_test, Y_test))
        scores.append(model.predict(X_test))

    return results, mse_val, score_mse, scores

def general_training_mod(nb_epochs, batch_size, txt_file, res, train_size, size, step, of_type, val_split, dataset_split, data_format):

    dir = exp_name+"_"+batch_size+"_"+nb_epochs+"_"+res+"_"+of_type+"/"

    if not os.path.isdir(dir):
        os.mkdir(dir)

    if res == '832x480':
        img_rows = 480
        img_cols = 832

    if res == '640x360':
        img_rows = 360
        img_cols = 640

    train_mod, val_mod, test_mod, video_names, video_scores, video_names_train, video_names_val, video_names_test = create_lists(txt_file, 'mod', res, train_size)

    data_list_mod_train = []
    for filename in train_mod:
        data_list_mod_train.append(root+filename.split('_dst')[0]+'/'+filename.split('_mod')[0]+'/of_%dx%d_mod_f/' % (img_cols, img_rows) + filename)

    data_list_mod_val = []
    for filename in val_mod:
        data_list_mod_val.append(root+filename.split('_dst')[0]+'/'+filename.split('_mod')[0]+'/of_%dx%d_mod_f/' % (img_cols, img_rows) + filename)

    sio.savemat(dir+"trainvid_"+of_type+"_"+res+"_exp3new.mat", {"Train Name":video_names_train})
    sio.savemat(dir+"valvid_mse_"+of_type+"_"+res+"_exp3new.mat", {"Val Name":video_names_val})
    sio.savemat(dir+"testvid_val_"+of_type+"_"+res+"_exp3new.mat", {"Test Name":video_names_test})

    model = little_convnet(img_rows, img_cols, of_type, lrate = 0.1, data_format)
    print(model.summary())

    lrate = LearningRateScheduler(step_decay)
    callbacks_list = [lrate]

    # X,Y
    X_train, Y_train = get_data_mod(data_list_mod_train, video_names, video_scores, size, step)
    X_val, Y_val = get_data_mod(data_list_mod_val, video_names, video_scores, size, step)

    results = []
    history = model.fit(X_train, Y_train, validation_data = (X_val, Y_val), epochs = nb_epochs, shuffle = True, verbose = 1, batch_size = batch_size)
    results.append(history.history)

    # model.save('modelo_general_mod.h5')

    data_list_mod_test = []
    for filename in test_mod:
        data_list_mod_test.append(root+filename.split('_dst')[0]+'/'+filename.split('_mod')[0]+'/of_%dx%d_mod_f/' % (img_cols, img_rows) + filename)

    X_test, Y_test = get_data_mod(data_list_mod_test, video_names, video_scores, size, step)
    predictions = model.predict(X_test)

    return results, predictions

def run(nb_epochs, batch_size, txt_file, res, train_size, size, step, of_type, val_split, dataset_split, exp_name, data_format, training_type):

    dir = exp_name+"_"+batch_size+"_"+nb_epochs+"_"+res+"_"+of_type+"/"

    if not os.path.isdir(dir):
        os.mkdir(dir)

    if of_type == 'mod':
        if training_type == 'batch':
            results, mse_val, score_mse, scores = batch_training_mod(nb_epochs, batch_size, txt_file,
                                                                    res, train_size, size, step, of_type,
                                                                    val_split, dataset_split, exp_name, data_format)

            sio.savemat(dir+"results_"+of_type+"_"+res+"_"+exp_name+".mat", {"results":results})
            sio.savemat(dir+"score_mse_"+of_type+"_"+res+"_"+exp_name+".mat", {"Scores_mse":score_mse})
            sio.savemat(dir+"mse_val_"+of_type+"_"+res+"_"+exp_name+".mat", {"Scores_mse_val":mse_val})
            sio.savemat(dir+"score_predict_"+of_type+"_"+res+"_"+exp_name+".mat", {"Scores":scores})

        elif training_type == 'general':
            results, scores = general_training_mod(nb_epochs, batch_size, txt_file,
                                                                    res, train_size, size, step, of_type,
                                                                    val_split, dataset_split, exp_name, data_format)

            sio.savemat(dir+"results_"+of_type+"_"+res+"_"+exp_name+".mat", {"results":results})
            sio.savemat(dir+"score_predict_"+of_type+"_"+res+"_"+exp_name+".mat", {"Scores":scores})

    if of_type == 'comp':
        if training_type == 'batch':
            results, mse_val, score_mse, scores = batch_training_comp(nb_epochs, batch_size, txt_file,
                                                                    res, train_size, size, step, of_type,
                                                                    val_split, dataset_split, exp_name, data_format)

            sio.savemat(dir+"results_"+of_type+"_"+res+"_"+exp_name+".mat", {"results":results})
            sio.savemat(dir+"score_mse_"+of_type+"_"+res+"_"+exp_name+".mat", {"Scores_mse":score_mse})
            sio.savemat(dir+"mse_val_"+of_type+"_"+res+"_"+exp_name+".mat", {"Scores_mse_val":mse_val})
            sio.savemat(dir+"score_predict_"+of_type+"_"+res+"_"+exp_name+".mat", {"Scores":scores})

        elif training_type == 'general':
            results, scores = general_training_comp(nb_epochs, batch_size, txt_file,
                                                                    res, train_size, size, step, of_type,
                                                                    val_split, dataset_split, exp_name, data_format)

            sio.savemat(dir+"results_"+of_type+"_"+res+"_"+exp_name+".mat", {"results":results})
            sio.savemat(dir+"score_predict_"+of_type+"_"+res+"_"+exp_name+".mat", {"Scores":scores})

    return

if __name__ == "__main__":

    parser = ArgumentParser(description='Optical Flow CNNVQA')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--nb_epochs', type=int, default=30,
                        help='number of trainig epochs (default: 30)')
    parser.add_argument('--lrate', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--config', default='config.yaml', type=str,
                        help='config file path (default: config.yaml)')
    parser.add_argument('--exp_name', default='0', type=str,
                        help='Name of experiment (default: 0)')
    parser.add_argument('--size', default='5', type=int,
                        help='Size of stacks (default: 5)')
    parser.add_argument('--step', default='2', type=int,
                        help='Step between stacks (default: 2)')
    parser.add_argument('--res', default='832x480', type=str,
                        help='Resolution of video (default: 832x480)')
    parser.add_argument('--of_type', default='mod', type=str,
                        help='Optical flow type (default: module)')
    parser.add_argument('--training_type', default='general', type=str,
                        help='Batch or general training (default: general)')

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)

    print('Experiment Name: ' + args.exp_name)
    print('Batch Size: ' + str(args.batch_size))
    print('Number of epochs: ' + str(args.nb_epochs))
    print(args.of_type + ' Optical Flow with ' + args.res + ' resolution')
    print('Database: CSIQ')
    print('Model: Optical Flow CNNVQA')
    print('Training Type: '+ args.training_type)

    run(nb_epochs=args.nb_epochs, batch_size=args.batch_size, txt_file=config['txt_file'],
        res=args.res, train_size=config['train_size'], size=args.size, step=args.step,
        of_type=args.of_type, val_split=config['val_split'], dataset_split=config['dataset_split'],
        exp_name=args.exp_name, data_format=config['data_format'], training_type=args.training_type)
