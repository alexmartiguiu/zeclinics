#
# VERSION 0.2
#

import sys
import numpy as np
from os import listdir
from os.path import isfile, join

#FOR debugging
import time
import matplotlib.pyplot as plt

def read(path):
    start=time.time()

    path_files = [f for f in listdir(path)]
    num_files=len(path_files)
    img_arrays= [None] * num_files
    for i in range(num_files):
        files = [f for f in listdir(path+"/"+path_files[i]) if isfile(join(path+"/"+path_files[i], f))]
        num_frames=len(files)
        frames_arrays= [None] * num_frames
        for j in range(num_frames):
            frames_arrays[j]=np.load(path+"/"+path_files[i]+"/"+files[j])
        img_arrays[i]=frames_arrays

    print("Elapsed time = ", time.time()-start)
    return img_arrays

def process_frame(frame,debug):
    #NOT IMPLEMENTED YET
    return frame

def process_video(video,debug):
    processed_frames=[None]*len(video)
    for i in range(len(video)):
        if (debug and (i+1)%100==0):
            print(" -> Processing frame ",i+1," of ",len(video))
        processed_frames[i]=process_frame(video[i],debug)
    return processed_frames

def processdir(path, debug=False):
    video_arrays = read(path)
    if(debug):
        print("Number of read videos: "+str(len(video_arrays)))
    processed_videos = [None] * len(video_arrays)
    for i in range(len(video_arrays)):
        if (debug):
            print("Processing video ",i+1," of ",len(video_arrays))
        processed_videos[i]=process_video(video_arrays[i],debug)
    print("DONE!")


#########
#EXECUTION:
# import process
# output = processdir('path' -> Directory with subfolders for each video, debug = T or F)
#########
#EXAMPLE:
# processdir("stored_imgs",True)
#########
