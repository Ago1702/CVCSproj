import torch
import torch.nn as nn
from signaling.wavelets import WaveletTransform
from utils.modules import CBAM, SpatialMultiCBAM, ChannelMultiCBAM
from utils.transform import RandomTransform
from data.datasets import DirectoryRandomDataset, DirectorySequentialDataset
from data.dataloader import TransformDataLoader

class SignalNet:
    def __init__(self, in_channels):
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32)
        )
        pass

if __name__ == "__main__":
    ds = DirectoryRandomDataset("//work//cvcs2024//VisionWise//test")
    dataloader = TransformDataLoader(RandomTransform.GLOBAL_CROP, ds, batch_size=8, 
                                     dataset_mode=DirectoryRandomDataset.BASE, num_workers=2,
                                     pacman=False, transform=WaveletTransform(), num_channels=39)
    for i, img in enumerate(dataloader):
        print(img[0].shape)
        if(i == 0):
            break
    pass
