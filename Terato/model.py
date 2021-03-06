from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models

def createDeepLabv3_resnet_50(outputchannels=6):
    """DeepLabv3 class with custom head
    Args:
        outputchannels (int, optional): The number of output channels
        in your dataset masks. Defaults to 1.
    Returns:
        model: Returns the DeepLabv3 model with the ResNet101 backbone.
    """
    model = models.segmentation.deeplabv3_resnet50(pretrained=True,
                                                                 progress=True)
    model.classifier = DeepLabHead(2048, outputchannels)
    return model

def createDeepLabv3_mobilenet(outputchannels=6):
    """DeepLabv3 class with custom head
    Args:
        outputchannels (int, optional): The number of output channels
        in your dataset masks. Defaults to 1.
    Returns:
        model: Returns the DeepLabv3 model with the ResNet101 backbone.
    """
    model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True,
                                     progress=True)
    model.classifier = DeepLabHead(960, outputchannels)
    return model
