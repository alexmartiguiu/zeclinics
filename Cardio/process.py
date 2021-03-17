#
# VERSION 0.2
#

import sys
import os
import numpy as np
from os import listdir
from os.path import isfile, join

#FOR debugging
import time
import matplotlib.pyplot as plt

def read(path):
    start=time.time()

    path_files = [f for f in listdir(path)]

    #Bad execution handling
    if len(path_files)==0:
        print("No files found in the directory")
        return None

    #Check if we have one or many videos to treat
    # if there are subfolders --> many
    # if there are files inside --> one
    directory = os.path.isdir(path+'/'+path_files[0])
    if directory:
        num_files=len(path_files)
        img_arrays= [None] * num_files
        for i in range(num_files):
            files = [f for f in listdir(path+"/"+path_files[i]) if isfile(join(path+"/"+path_files[i], f))]
            num_frames=len(files)
            frames_arrays= [None] * num_frames
            for j in range(num_frames):
                frames_arrays[j]=np.load(path+"/"+path_files[i]+"/"+files[j])
            img_arrays[i]=frames_arrays
    else:
        num_files=len(path_files)
        img_arrays= [None] * num_files
        for i in range(num_files):
            img_arrays[i]=np.load(path+'/'+path_files[i])

    print("Elapsed time = ", time.time()-start)
    return img_arrays

def process_video(input_video, base_it=50, debug):
    if isinstance(input_video, str):
        video = read(input_video)
    else:
        video = input_video
    processed_frames, masks=[None]*len(video), [None]*len(video)]
    for i in range(len(video)):
        if (debug and (i+1)%100==0):
            print(" -> Processing frame ",i+1," of ",len(video))

        if i%2==0:
            if i ==0:
                iters=base_it
            else:
                iters=4

            masks[i] = None #ACTIVE CONTOURS

        else:
            masks[i] = masks[i-1]
            
        processed_frames[i] = video[i]*(1-masks[i])

    freq = None #get heartrate & things
    return [processed_frames, masks, freq]

def process_dir(input_video_arrays, debug=False):
    if isinstance(input_video_arrays, str):
        video_arrays = read(input_video_arrays)
    else:
        video_arrays = input_video_arrays
    if(debug):
        print("Number of read videos: "+str(len(video_arrays)))
    processed_videos = [None] * len(video_arrays)
    for i in range(len(video_arrays)):
        if (debug):
            print("Processing video ",i+1," of ",len(video_arrays))
        processed_videos[i]=process_video(video_arrays[i],debug)
    return processed_videos


#########
#EXECUTION:
# import process
# output = processdir('path' -> Directory with subfolders for each video, debug = T or F)
#########
#EXAMPLE:
# processdir("stored_imgs",True)
#########
