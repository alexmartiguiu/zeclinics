from pathlib import Path
from typing import Any, Callable, Optional
import torch
import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset




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

        sample['masks'] = torch.cat(sample['masks'], 0).double() # Concatenate the masks in one unique tensor
        return sample
