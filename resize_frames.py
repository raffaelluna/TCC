# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 06:54:30 2018

@author: Raffael
"""

import cv2
import os
import numpy as np
import shutil

root = 'C:/Users/Raffael/Desktop/Tcc2/csiq_video_database/'

pastas_videos = [x for x in os.listdir(root)]

sub_pastas = []
for i in xrange(len(pastas_videos)):
    sub_pastas.append(os.listdir(root+pastas_videos[i]))

path_videos = []   
videos = []
for i in xrange(len(sub_pastas)):
    for j in xrange(len(sub_pastas[i])):
        videos.append(sub_pastas[i][j])
        path_videos.append(root+pastas_videos[i]+'/'+sub_pastas[i][j]+'/')       
  
#for i in range(len(path_videos)):    
#    if not os.path.isdir(path_videos[i]+'frames_640x360'):
#        os.mkdir(path_videos[i]+'frames_640x360')
            
video_file = []
frame_folder = [] 
of_folder_640x360 = []
frames_640x360 = []       
for i in xrange(len(videos)):
    video_file.append(path_videos[i]+videos[i]+'.yuv')
    frame_folder.append(path_videos[i]+'frames')
    of_folder_640x360.append(path_videos[i]+'of_output_640x360')
    frames_640x360.append(path_videos[i]+'frames_640x360')

for i in xrange(len(frame_folder)):
    imagens = [y for y in os.listdir(frame_folder[i])]
    
    for j in xrange(len(imagens)):
        imagem = cv2.imread(frame_folder[i]+'/'+imagens[j])
        
        dsize = (640,360)
        dst = cv2.resize(imagem, dsize, 0, 0, cv2.INTER_AREA)

        cv2.imwrite(frames_640x360[i]+'/'+'frame_640x360_%d.png' % (j+1), dst)
