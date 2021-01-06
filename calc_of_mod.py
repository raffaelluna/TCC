# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 11:39:19 2018

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
#    if not os.path.isdir(path_videos[i]+'of_832x480_mod'):
#        os.mkdir(path_videos[i]+'of_832x480_mod')
#
#for i in range(len(path_videos)):    
#    if not os.path.isdir(path_videos[i]+'of_640x360_mod'):
#        os.mkdir(path_videos[i]+'of_640x360_mod')
        
of_832x480_mod = []
of_640x360_mod = []  
of_output = []      
for i in xrange(len(videos)):
    of_832x480_mod.append(path_videos[i]+'of_832x480_mod/')
    of_640x360_mod.append(path_videos[i]+'of_640x360_mod/')
    of_output.append(path_videos[i]+'of_output/')
            
video_file = []
frame_folder = [] 
of_folder_640x360 = []
frames_640x360 = []       
for i in xrange(len(videos)):
    video_file.append(path_videos[i]+videos[i]+'.yuv')
    frame_folder.append(path_videos[i]+'frames')
    of_folder_640x360.append(path_videos[i]+'of_output_640x360')
    frames_640x360.append(path_videos[i]+'frames_640x360')


        
for i in xrange(len(of_folder_640x360)):
    of_comp = [y for y in os.listdir(of_folder_640x360[i])]

    for j in xrange(1,len(of_comp)/2 + 1):
        ofhorz = cv2.imread(of_folder_640x360[i]+'/dx_640x360_%d.png' %j,0)
        prvs = ofhorz.astype('uint16')

        ofvert = cv2.imread(of_folder_640x360[i]+'/dy_640x360_%d.png' %j,0)
        next = ofvert.astype('uint16')

        modulo = np.sqrt(next**2 + prvs**2)

        cv2.imwrite(of_640x360_mod[i]+'mod_640x360_'+str(j)+'.png', modulo)
    
#for i in xrange(len(of_output)):
#    of_comp = [y for y in os.listdir(of_output[i])]
#    
#    frame1 = cv2.imread(frames_640x360[i]+'/frame_640x360_1.png')
#    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
#    
#    count = 0
#    for j in xrange(1,len(videos_frame)):
#        frame2 = cv2.imread(frames_640x360[i]+'/frame_640x360_%d.png' %(j+1))
#        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
#    
#        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#    
#        horz = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)
#        vert = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)
#        horz = horz.astype('uint8')
#        vert = vert.astype('uint8')
#        
#        cv2.imwrite(of_folder_640x360[i]+'/dx_640x360_'+str(count+1)+'.png', horz)
#        cv2.imwrite(of_folder_640x360[i]+'/dy_640x360_'+str(count+1)+'.png', vert)
#        
#        prvs = next
#        count+=1