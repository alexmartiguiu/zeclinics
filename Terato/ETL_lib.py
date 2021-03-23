import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import math

from read_roi import read_roi_file
import xml.etree.ElementTree as ET
from copy import copy, deepcopy

def ellipse_to_pol(roi):
    pts = []
    p1 = np.array([roi['ex1'],roi['ey1']])
    p2 = np.array([roi['ex2'],roi['ey2']])
    a = np.linalg.norm(p1-p2)/2
    c = (p1+p2)/2
    b = a*roi['aspect_ratio']
    vec = np.arange(roi['ex1'],roi['ex2'],1)
    vec2 = np.arange(roi['ex2'],roi['ex1'],-1)
    #print(vec2)
    for i in vec:
        pts.append([i,b*math.sqrt(1-((i-c[0])**2/a**2))+c[1]])
    for i in vec2:
        pts.append([i,-b*math.sqrt(1-((i-c[0])**2/a**2))+c[1]])
    return np.array(pts, 'int32')

#given the path of the image and its corresponding roi returns the image with a mask
#built from the roi
def obtain_mask(img,roi):
    if roi['type'] == 'polygon':
        pts = zip(roi['x'],roi['y'])
        pts2 = np.array(list(pts), 'int32')
        #Important to put the brackets []!!!!
        cv2.fillPoly( img , [pts2], (255))
    elif roi['type'] == 'freehand':
        #Important to put the brackets []!!!!
        cv2.fillPoly( img , [ellipse_to_pol(roi)], (255))
    return img

'''
input:
  roi_paths: list of absolute paths of the roi files
  mask_name: list of names of the masks to access them in the roi dictionaries
  root_data_folder: parent folder of all the output mask folders
  mask_folder: name of the output folder
  im_type: "dor" or "lat"
  image_name: name of the image to match with ("plate_name"_"well_name")
  shape: (width, height) of the output mask image
'''
def read_roi_and_get_mask(roi_paths,mask_names,root_data_folder,mask_folder,im_type,image_name,shape):
    #Create Black image to put the masks on:
    mask_img = np.zeros(shape, np.uint8)

    for i,(roi_path,mask_name) in enumerate(zip(roi_paths,mask_names)):
        #Get the roi
        roi = read_roi_file(roi_path)[mask_name]
        #Create the mask
        mask_img = obtain_mask(mask_img,roi)
        #Define the path to be written to
        mask_path = root_data_folder + "/" + mask_folder + "/" + image_name + "_" + im_type + ".jpg"
    return mask_path, mask_img

'''
Given a root path, a list of names for the masks folders and a name for
the input images folder, creates all the folders if they are not yet created
'''
def create_directories(output_root,mask_folders,images_folder):
    if not os.path.exists(output_root):
        os.system("mkdir " + output_root)
    if not os.path.exists(output_root + "/" + images_folder):
        os.system("mkdir " + output_root + "/" + images_folder)
    for folder in mask_folders:
        if not os.path.exists(output_root + "/" + folder):
            os.system("mkdir " + output_root + "/" + folder)
'''
given an image name in the form: plate_name + _ + well_name + _ + type + .jpg,
return plate_name, well_name

example:
    image_name = 20190902_1046_CS_R1_Well_left_A01_dor.jpg
    return: 20190902_1046_CS_R1, Well_left_A01
'''
def parse_image_name(image_name):
    s = image_name.split('Well')
    #get the first element and remove the "_" at the end
    plate_name = s[0][:-1]
    well_name = "Well" + s[1]
    return plate_name, well_name

'''
In drive:
    output_folder = "/content/drive/MyDrive/Zeclinics/TERATO/Data"
    raw_data_path = "/content/drive/MyDrive/UPC/BAT1"
'''
def data_generation_pipeline(raw_data_path,output_folder):
    #parameters to write the outputs
    mask_names = [['heart_lateral'],['yolk_lateral'],['fishoutline_lateral'],['ov_lateral'],['eye_up_dorsal','eye_down_dorsal'],['fishoutline_dorsal']]
    im_types = ['lat','lat','lat','lat','dor','dor']
    mask_folders = ['Heart_lateral','Yolk_lateral','Outline_lateral','Ov_lateral','Eyes_dorsal','Outline_dorsal']

    #Create the folders if they are not yet created
    create_directories(output_folder,mask_folders,"Image")

    #iterate over every plate
    for plate_name in os.listdir(raw_data_path):
        plate_path = raw_data_path + "/" + plate_name
        tree = ET.parse(plate_path + "/" + plate_name + ".xml")
        #The root is the plate
        plate = tree.getroot()
        print(plate.tag,plate.attrib)
        #Every child of the plate is a well
        for well in plate:
            #I guess that if show2user is 0 we can skip the well
            if int(well.attrib['show2user']):
                #If we iterate over the well we obtain the boolean features and other stuff
                well_name = well.attrib['well_folder']
                well_path = plate_path + "/" + well_name

                #paths images
                dorsal_img_path = well_path + "/" + well.attrib['dorsal_image']
                lateral_img_path = well_path + "/" + well.attrib['lateral_image']

                # roi path lateral
                heart_roi_path = well_path + "/" + "heart_lateral.roi"
                yolk_roi_path = well_path + "/" + "yolk_lateral.roi"
                outline_lat_roi_path = well_path + "/" + "fishoutline_lateral.roi"
                ov_roi_path = well_path + "/" + "ov_lateral.roi"

                # roi path dorsal
                eye_up_roi_path = well_path + "/" + "eye_up_dorsal.roi"
                eye_down_roi_path = well_path + "/" + "eye_down_dorsal.roi"
                outline_dor_roi_path = well_path + "/" + "fishoutline_dorsal.roi"

                #put the roi paths in a list
                roi_paths = [[heart_roi_path], [yolk_roi_path], [outline_lat_roi_path], [ov_roi_path], [eye_up_roi_path, eye_down_roi_path],[outline_dor_roi_path]]
                image_name = plate_name + "_" + well_name

                #We define a vector of pairs (image,path) in order to write them at the end
                outputs = []

                #lateral and dorsal image
                try:
                    im_lat = cv2.imread(lateral_img_path, 1)
                    im_dor = cv2.imread(dorsal_img_path, 1)
                    outputs.append((output_folder + "/Image/" + image_name + "_lat.jpg",im_lat))
                    outputs.append((output_folder + "/Image/" + image_name + "_dor.jpg",im_dor))
                except: continue

                height, width, channels = im_lat.shape
                mask_shape = (height,width,1)

                #Generate all the masks
                error = False
                for i in range(len(mask_folders)):
                    if error: break
                    try:
                        mask_path, mask = read_roi_and_get_mask(roi_paths[i],mask_names[i],output_folder,mask_folders[i],im_types[i],image_name,mask_shape)
                        outputs.append((mask_path,mask))
                    except:
                      error = True
                if error: continue

                #If we are here, everything went fine, so we can write the images
                for image_path, image in outputs:
                    cv2.imwrite(image_path, image)
                print('Complete Well')
            else:
                print("Show2User is False")
