#from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
import torch

def binary_fenotypes(outputchannels=10):
    """DeepLabv3 class with custom head
    Args:
        outputchannels (int, optional): The number of output channels
        in your dataset masks. Defaults to 1.
    Returns:
        model: Returns the DeepLabv3 model with the ResNet101 backbone.
    """
    model = models.mobilenet_v3_small(pretrained=True,
                                     progress=True)
    '''
    Original header
      (classifier): Sequential(
    (0): Linear(in_features=576, out_features=1024, bias=True)
    (1): Hardswish()
    (2): Dropout(p=0.2, inplace=True)
    (3): Linear(in_features=1024, out_features=1000, bias=True)
  )
    '''
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(in_features=576, out_features=1024, bias=True),
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=1024, out_features=outputchannels, bias=True)
    )

    return model
