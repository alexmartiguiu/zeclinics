import os
import sys
import torch
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils import data
import model as m
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms

def get_dataloader_single_folder(data_dir: str,
                                 image_folder: str,
                                 mask_folders: list,
                                 fraction: float = 0.2,
                                 batch_size: int = 4):
    """Create train and test dataloader from a single directory containing
    the image and mask folders.
    Args:
        data_dir (str): Data directory path or root
        image_folder (str, optional): Image folder name. Defaults to 'Images'.
        mask_folder (str, optional): Mask folder name. Defaults to 'Masks'.
        fraction (float, optional): Fraction of Test set. Defaults to 0.2.
        batch_size (int, optional): Dataloader batch size. Defaults to 4.
    Returns:
        dataloaders: Returns dataloaders dictionary containing the
        Train and Test dataloaders.
    """
    data_transforms = transforms.Compose([transforms.ToTensor()])

    image_datasets = {
        x: m.SegmentationDataset(data_dir,
                               image_folder=image_folder,
                               mask_folders=mask_folders,
                               seed=100,
                               fraction=fraction,
                               subset=x,
                               transforms=data_transforms)
        for x in ['Train', 'Test']
    }
    dataloaders = {
        x: DataLoader(image_datasets[x],
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=8)
        for x in ['Train', 'Test']
    }
    return dataloaders

data_path = './Data'
masks = os.listdir(data_path)[2:]
dataloaders = get_dataloader_single_folder(data_path,'Image',masks)

exp_directory = './Model'

exp_directory = Path(exp_directory)
if not exp_directory.exists():
    exp_directory.mkdir()

model = m.createDeepLabv3()
# Specify the loss function
criterion = torch.nn.MSELoss(reduction='mean')
# Specify the optimizer with a lower learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Specify the evaluation metrics
metrics = {'f1_score': f1_score, 'auroc': roc_auc_score}

# Create the dataloader
dataloaders = get_dataloader_single_folder(
                  './Data', 'Image',['Eyes_dorsal', 'Outline_dorsal', 'Outline_lateral', 'Ov_lateral', 'Heart_lateral', 'Yolk_lateral'],batch_size=2)
_ = m.train_model(model,
                criterion,
                dataloaders,
                optimizer,
                bpath=exp_directory,
                metrics=metrics,
                num_epochs=1)
