#
# VERSION 0.2
#

import sys
import os
import numpy as np
from os import listdir
from os.path import isfile, join

# For debugging
import time
import matplotlib.pyplot as plt

# For computing beats per minute
import heartpy as hp

def read(path):
    start=time.time()

    path_files = [f for f in listdir(path)]

    # Bad execution handling
    if len(path_files)==0:
        print("No files found in the directory")
        return None

    # Check if we have one or many videos to treat
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

def compute_area_size_heart(frames):
    # Initial level set
    area_per_frame = []
    masks = [None]*len(frames)
    times = [None]*len(frames)
    num_total_px   = 256*256
    for i in range(len(frames)):
        start=time.time()
        image = frames[i]
        if i==0:
            init_ls = checkerboard_level_set(image.shape, 6)
            iters = 50
        else:
            init_ls = inici
            iters = 5
        # List with intermediate results for plotting the evolution
        evolution = []
        callback = store_evolution_in(evolution)
        ls = morphological_chan_vese(image, iters, init_level_set=init_ls, smoothing=3,
                                    iter_callback=callback)
        
        if i==0:
            inici = evolution[49]
        else:
            inici = evolution[1]
        
        masks[i] = ls
        reduced = ls[20:235, 20:235]
        unique, counts = np.unique(reduced, return_counts=True)
        #dict(zip(unique, counts))
        area_per_frame.append(min(counts))
        times[i] = time.time()-start
    area_per_frame_percentage = [x / num_total_px for x in area_per_frame]
    return area_per_frame_percentage, min(area_per_frame), max(area_per_frame),masks,times

def compute_measures(frames):
    freq = compute_area_size(frames)
    

    array = freq[0]

    fs = 30
    wd, m = hp.process(array, fs, report_time=True)

    #set large figure
    plt.figure()

    #call plotter
    hp.plotter(wd, m)

    #display measures computed
    for measure in m.keys():
        print('%s: %f' %(measure, m[measure]))

#########
#EXECUTION:
# import process
# output = processdir('path' -> Directory with subfolders for each video, debug = T or F)
#########
#EXAMPLE:
# processdir("stored_imgs",True)
#########
