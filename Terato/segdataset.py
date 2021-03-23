from pathlib import Path
from typing import Any, Callable, Optional
import torch
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
from torchvision.datasets.vision import VisionDataset
from ETL_lib import parse_image_name




class SegmentationDataset(VisionDataset):
    """A PyTorch dataset for image segmentation task.
    The dataset is compatible with torchvision transforms.
    The transforms passed would be applied to both the Images and Masks.
    """
    def __init__(self,
                 root: str,
                 image_folder: str,
                 mask_folders: list, #list of paths
                 transforms: Optional[Callable] = None,
                 seed: int = None,
                 fraction: float = None,
                 subset: str = None,
                 image_color_mode: str = "rgb",
                 mask_color_mode: str = "grayscale") -> None:
        """
        Args:
            root (str): Root directory path.
            image_folder (str): Name of the folder that contains the images in the root directory.
            mask_folder (str): Name of the folder that contains the masks in the root directory.
            transforms (Optional[Callable], optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.ToTensor`` for images. Defaults to None.
            seed (int, optional): Specify a seed for the train and test split for reproducible results. Defaults to None.
            fraction (float, optional): A float value from 0 to 1 which specifies the validation split fraction. Defaults to None.
            subset (str, optional): 'Train' or 'Test' to select the appropriate set. Defaults to None.
            image_color_mode (str, optional): 'rgb' or 'grayscale'. Defaults to 'rgb'.
            mask_color_mode (str, optional): 'rgb' or 'grayscale'. Defaults to 'grayscale'.
        Raises:
            OSError: If image folder doesn't exist in root.
            OSError: If mask folder doesn't exist in root.
            ValueError: If subset is not either 'Train' or 'Test'
            ValueError: If image_color_mode and mask_color_mode are either 'rgb' or 'grayscale'
        """
        super().__init__(root, transforms)

        self.mask_folders = mask_folders
        image_folder_path = Path(self.root) / image_folder #La barra es un operador de la clase Path
        self.mask_folder_paths = [Path(self.root) / p for p in mask_folders]

        if not image_folder_path.exists():
            raise OSError(f"{image_folder_path} does not exist.")
        for p in self.mask_folder_paths:
            if not p.exists():
                raise OSError(f"{p} does not exist.")

        if image_color_mode not in ["rgb", "grayscale"]:
            raise ValueError(
                f"{image_color_mode} is an invalid choice. Please enter from rgb grayscale."
            )
        if mask_color_mode not in ["rgb", "grayscale"]:
            raise ValueError(
                f"{mask_color_mode} is an invalid choice. Please enter from rgb grayscale."
            )

        self.image_color_mode = image_color_mode
        self.mask_color_mode = mask_color_mode

        if not fraction:
            self.image_names = sorted(image_folder_path.glob("*"))

        else:
            if subset not in ["Train", "Test"]:
                raise (ValueError(
                    f"{subset} is not a valid input. Acceptable values are Train and Test."
                ))
            self.fraction = fraction
            self.image_list = np.array(sorted(image_folder_path.glob("*")))
            '''
            Here we shuffle the data
            '''
            if seed:
                np.random.seed(seed)
                indices = np.arange(len(self.image_list))
                np.random.shuffle(indices)
                self.image_list = self.image_list[indices]
            '''
            Select the last fraction % ot the list as train
            '''
            if subset == "Train":
                self.image_names = self.image_list[:int(
                    np.ceil(len(self.image_list) * (1 - self.fraction)))]
            else:
                self.image_names = self.image_list[
                    int(np.ceil(len(self.image_list) * (1 - self.fraction))):]

    def __len__(self) -> int:
        return len(self.image_names)
    '''
    Given an index returns a dictionary in the form:
        {
            "image": original image tensor
            "masks": multidimensional tensor containing all the masks
        }
    '''
    def __getitem__(self, index: int) -> Any:
        image_path = self.image_names[index]

        # Let's read the image
        with open(image_path, "rb") as image_file:
            '''
            Read the image and save it in the dictionary
            '''
            image = Image.open(image_file)
            if self.image_color_mode == "rgb":
                image = image.convert("RGB")
            elif self.image_color_mode == "grayscale":
                image = image.convert("L")

        sample = {'image' : self.transforms(image)}
        sample['masks'] = []


        # Let's create a black image for the masks that don't correspond to the image
        THRESHOLD_VALUE = 255
        #Load image and convert to greyscale
        imgData = np.asarray(image.convert("L"))
        black_mask = (imgData > THRESHOLD_VALUE) * 1.0


        # Iterate over all the masks, read them and save them in the dictionary
        for mask_path in self.mask_folder_paths:
            try:
                with open(mask_path / image_path.parts[-1],"rb") as mask_file:
                    mask = Image.open(mask_file)
                    if self.mask_color_mode == "rgb":
                        mask = mask.convert("RGB")
                    elif self.mask_color_mode == "grayscale":
                        mask = mask.convert("L")
                    sample['masks'].append(self.transforms(mask))

            except:
                sample['masks'].append(self.transforms(black_mask))
        sample['masks'] = torch.cat(sample['masks'], 0).float() # Concatenate the masks in one unique tensor
        return sample

class ClassificationDataset(VisionDataset):
    """A PyTorch dataset for image classification task.
    The dataset is compatible with torchvision transforms.
    The transforms passed would be applied to both the Images and Masks.
    """
    def __init__(self,
                 root: str,
                 image_folder: str,
                 mask_folders: list, #list of paths
                 raw_data_path: str,
                 transforms: Optional[Callable] = None,
                 seed: int = None,
                 fraction: float = None,
                 subset: str = None,
                 image_color_mode: str = "rgb",
                 mask_color_mode: str = "grayscale") -> None:
        """
        Args:
            root (str): Root directory path.
            image_folder (str): Name of the folder that contains the images in the root directory.
            mask_folders (str): Name of the folders that contain the masks in the root directory.
                                It should be ['Outline_lateral','Outline_dorsal']
            raw_data_path (str): Path of the raw data. Here we look for the anotated phenotypes.
            transforms (Optional[Callable], optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.ToTensor`` for images. Defaults to None.
            seed (int, optional): Specify a seed for the train and test split for reproducible results. Defaults to None.
            fraction (float, optional): A float value from 0 to 1 which specifies the validation split fraction. Defaults to None.
            subset (str, optional): 'Train' or 'Test' to select the appropriate set. Defaults to None.
            image_color_mode (str, optional): 'rgb' or 'grayscale'. Defaults to 'rgb'.
            mask_color_mode (str, optional): 'rgb' or 'grayscale'. Defaults to 'grayscale'.
        Raises:
            OSError: If image folder doesn't exist in root.
            OSError: If mask folder doesn't exist in root.
            ValueError: If subset is not either 'Train' or 'Test'
            ValueError: If image_color_mode and mask_color_mode are either 'rgb' or 'grayscale'
        """
        super().__init__(root, transforms)

        self.raw_data_path = raw_data_path
        self.mask_folders = mask_folders
        self.image_folder_path = Path(self.root) / image_folder #La barra es un operador de la clase Path
        self.mask_folder_paths = [Path(self.root) / p for p in mask_folders]

        if not self.image_folder_path.exists():
            raise OSError(f"{self.image_folder_path} does not exist.")
        for p in self.mask_folder_paths:
            if not p.exists():
                raise OSError(f"{p} does not exist.")

        if image_color_mode not in ["rgb", "grayscale"]:
            raise ValueError(
                f"{image_color_mode} is an invalid choice. Please enter from rgb grayscale."
            )
        if mask_color_mode not in ["rgb", "grayscale"]:
            raise ValueError(
                f"{mask_color_mode} is an invalid choice. Please enter from rgb grayscale."
            )

        self.image_color_mode = image_color_mode
        self.mask_color_mode = mask_color_mode

        if not fraction:
            self.image_names = sorted(self.image_folder_path.glob("*"))

        else:
            if subset not in ["Train", "Test"]:
                raise (ValueError(
                    f"{subset} is not a valid input. Acceptable values are Train and Test."
                ))
            self.fraction = fraction
            self.image_list = [str(s)[:-8] for s in self.mask_folder_paths[0].glob("*")]
            self.image_list = np.array(sorted(self.image_list))
            '''
            Here we shuffle the data
            '''
            if seed:
                np.random.seed(seed)
                indices = np.arange(len(self.image_list))
                np.random.shuffle(indices)
                self.image_list = self.image_list[indices]
            '''
            Select the last fraction % ot the list as train
            '''
            if subset == "Train":
                self.image_names = self.image_list[:int(
                    np.ceil(len(self.image_list) * (1 - self.fraction)))]
            else:
                self.image_names = self.image_list[
                    int(np.ceil(len(self.image_list) * (1 - self.fraction))):]

    def __len__(self) -> int:
        return len(self.image_names)
    '''
    Given an index returns a dictionary in the form:
        {
            "image": concatenation of the multiplication of
                     the lateral and dorsal images tensors with their
                     respective outline masks
            "fenotypes": dictionary containing all the fenotypes values
        }
    '''
    def __getitem__(self, index: int) -> Any:
        image_name = self.image_names[index].split('/')[-1]
        # Let's read the images
        with open(str(self.image_folder_path / image_name) + '_lat.jpg', "rb") as image_lat_file:
            '''
            Read the image and save it in the dictionary
            '''
            image_lat = Image.open(image_lat_file)
            if self.image_color_mode == "rgb":
                image_lat = image_lat.convert("RGB")
            elif self.image_color_mode == "grayscale":
                image_lat = image_lat.convert("L")
            image_lat = self.transforms(image_lat)


        with open(str(self.image_folder_path / image_name) + '_dor.jpg', "rb") as image_dor_file:
            '''
            Read the image and save it in the dictionary
            '''
            image_dor = Image.open(image_dor_file)
            if self.image_color_mode == "rgb":
                image = image_dor.convert("RGB")
            elif self.image_color_mode == "grayscale":
                image = image_dor.convert("L")
            image_dor = self.transforms(image_dor)

        #Read the masks and multiply the images by them
        #Lateral:
        with open(str(self.mask_folder_paths[0] / image_name) + "_lat.jpg","rb") as mask_lat_file:
            mask_lat = Image.open(mask_lat_file)
            if self.mask_color_mode == "rgb":
                mask_lat = mask_lat.convert("RGB")
            elif self.mask_color_mode == "grayscale":
                mask_lat = mask_lat.convert("L")
            mask_lat = self.transforms(mask_lat)
        #Dorsal:
        with open(str(self.mask_folder_paths[1] / image_name) + "_dor.jpg","rb") as mask_dor_file:
            mask_dor = Image.open(mask_dor_file)
            if self.mask_color_mode == "rgb":
                mask_dor = mask_dor.convert("RGB")
            elif self.mask_color_mode == "grayscale":
                mask_dor = mask_dor.convert("L")
            mask_dor = self.transforms(mask_dor)

        #Multiply by the masks
        image_lat = image_lat.mul(mask_lat)
        image_dor = image_dor.mul(mask_dor)

        #Concatenate the obtained masks:
        sample = {'image' : torch.cat((image_dor,image_lat),dim = 1)}

        #get the boolean fenotypes from the xml:
        translate_feno_nomenclature = {
        'False' : 0,
        'True' : 1,
        'ND' : 2
        }
        plate_name, well_name = parse_image_name(image_name)
        tree = ET.parse(self.raw_data_path + '/' + plate_name + '/' + plate_name + '.xml')
        #The root is the plate
        plate = tree.getroot()
        fenotypes = {}
        for well in plate:
            print(well.attrib['well_folder'], well_name)
            if well.attrib['well_folder'] == well_name:
                for feno in well:
                    fenotypes[feno.tag] = translate_feno_nomenclature[feno.attrib['value']]
        sample['fenotypes'] = fenotypes
        return sample
