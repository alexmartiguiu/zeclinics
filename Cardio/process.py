#
# VERSION 0.2
#

import sys
sys.path.insert(1, './dependencies')
import os
import numpy as np
from os import listdir
from os.path import isfile, join
from preprocess import lifpreprocess

import skimage
from skimage.color import rgb2gray
from skimage import data, img_as_float
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.segmentation import (morphological_chan_vese,
                                  morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient,
                                  checkerboard_level_set)

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

    print("READ & LOAD: Elapsed time = ", time.time()-start)
    return img_arrays

def store_evolution_in(lst):
    #Returns a callback function to store the evolution of the level sets in
    # the given list.

    def _store(x):
        lst.append(np.copy(x))

    return _store

def process_video(input_video, base_it=50, update_it=4, skip=1, memory_it=1, debug=False, p_out_dir='output', p_index=2,p_out_shape=256,p_store=False):
    # FUNCTION TO PROCESS A LIST OF frames
    # This function applies morphological_chan_vese to each frame,
    #   implemented with memory for efficiency.
    #
    # --> input_video: EITHER A LifFile, DIR OR AN ARRAY
    # --> base_it: active contour iterations for step 0
    # --> update_it: iterations for steps > 0
    # --> skip: parameter to add skipping to the algorithm (skip n frames every n+1)
    # --> memory_it: iteration which is passed to the following frame as baseline

    if isinstance(input_video, str):
        if(".lif" in input_video):
            video = lifpreprocess(input_video,out_dir=p_out_dir,index_of_interest=p_index,out_shape=p_out_shape,store=p_store,debug=debug)
        else:
            video = read(input_video)
    else:
        video = input_video
    processed_frames, masks=[None]*len(video), [None]*len(video)

    if debug:
        start = time.time()

    inici = None
    for i in range(len(video)):
        evolution = []
        if (debug and (i+1)%100==0):
            print(" -> Processing frame ",i+1," of ",len(video))

        if not skip or i%(skip+1)==0:
            if i ==0:
                init_ls = checkerboard_level_set(video[i].shape, 6)
                iters=base_it
            else:
                init_ls = inici
                iters=update_it

            callback = store_evolution_in(evolution)
            masks[i] = morphological_chan_vese(video[i], iters, init_level_set=init_ls, smoothing=3, iter_callback=callback)

        else:
            masks[i] = masks[i-1]

        if i==0:
            inici = evolution[base_it-1]
        else:
            inici = evolution[memory_it]

        processed_frames[i] = video[i]*(1-masks[i])

    if debug:
        print("PROCESS: Elapsed time = ",time.time()-start)
    freq = None #get heartrate & things
    return processed_frames, masks, freq

def process_dir(input_video_arrays, raw=True, p_out_dir='output', p_index=2, p_out_shape=256, p_store=False, debug=False):
    if isinstance(input_video_arrays, str):
        if raw:
            video_arrays = lifpreprocess(input_video_arrays,out_dir=p_out_dir,index_of_interest=p_index,out_shape=p_out_shape,store=p_store,debug=debug)
        else:
            video_arrays = read(input_video_arrays)
    else:
        video_arrays = input_video_arrays
    if(debug):
        print("Number of read videos: "+str(len(video_arrays)))
    processed_videos,masks,freq = [None] * len(video_arrays),[None] * len(video_arrays),[None] * len(video_arrays)
    for i in range(len(video_arrays)):
        if (debug):
            print("Processing video ",i+1," of ",len(video_arrays))
        processed_videos[i],masks[i],freq[i]=process_video(video_arrays[i],debug=debug)
    return masks,processed_videos,freq


#########
#EXECUTION:
# import process
# output = processdir('path' -> Directory with subfolders for each video, debug = T or F)
#########
#EXAMPLE:
#p_vid,masks,freq=process_video("20170102_SME_085",debug=True,skip=0)
#plt.imshow(p_vid[2270], cmap='gray')
#plt.show()
#########
