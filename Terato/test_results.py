from pathlib import Path
import os
import click
import sklearn.metrics
from torch.utils import data
import torch

import datahandler
from evaluate import test_evaluation

'''
# WARNING:
Check https://pytorch.org/get-started/previous-versions/ and install the proper
pytorch and torchvision versions according to your cuda version.

You can figure out your cuda versions with:
/usr/local/cuda/bin/nvcc --version

NOTES:
-Batch size needs to be larger than one due to the batch normalization.

-The chosen loss funcion (nn.BCEWithLogitsLoss()) applies a sigmoid to the
output ans then applies the binary cross entropy loss function (a pixel belongs
to a class or doesn't)
'''


'''

This first part of the code is responsible to define where data is found and
where the model is going to be saved. In order to do that a main data directory
must be declared. Inside the this directory an image folder must exists with
all the raw images and the folders with the different masks containg the masks
with the exact same name of their corresponding original image:

                     ___________data_path _____________
                    /               |                  \
             Image_folder      Mask1_folder  ...  Maskn_folder
                  |                 |                  |
              img1.png          img1.png           img1.png
                  .                 .                  .
                  .                 .                  .
                  .                 .                  .
              imgk.png          imgk.png           imgk.png


'''

data_path = '../Data'

# Define images_path and masks paths from data_path
images_path = 'Image'
masks_paths = ['Eyes_dorsal', 'Outline_lateral', 'Yolk_lateral', 'Heart_lateral', 'Outline_dorsal', 'Ov_lateral']


################################### Model creation ###################################



model = torch.load('./Model_resnet50_BCELoss/weights.pt',map_location=torch.device('cpu'))
print(model)
criterion = torch.nn.BCEWithLogitsLoss(reduction='mean') # Specify the loss function

# Specify the evaluation metrics
metrics = {'f1_score': sklearn.metrics.f1_score,
           'precision': sklearn.metrics.precision_score,
           'recall': sklearn.metrics.recall_score}
           #'auroc': sklearn.metrics.roc_auc_score}
           #'accuracy_score': sklearn.metrics.accuracy_score}
# Ceation of the data loaders ['Train', 'Test']
dataloaders = datahandler.get_dataloader_single_folder(data_path,
                                                       images_path,
                                                       masks_paths,
                                                       batch_size=8,
                                                       num_workers = 8)

print(test_evaluation(model,
                criterion,
                dataloaders,
                bpath=exp_directory,
                masks_names = masks_paths,
                metrics=metrics))
