import torch
import torch.nn as nn
from signaling.wavelets import WaveletTransform
from utils.modules import CBAM, ChannelMultiCBAM
from utils.transform import RandomTransform
from data.datasets import DirectoryRandomDataset, DirectorySequentialDataset
from data.dataloader import TransformDataLoader
from torchvision import models
from utils.modules import initialize_weights

class SignalNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.resnet = models.resnet50(models.ResNet50_Weights.IMAGENET1K_V1)
        self.input = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=3,
                               stride=1, padding=1))
        self.output = nn.Sequential(
            nn.Linear(in_features=self.resnet.fc.out_features, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=1)
        )
        pass

    def forward(self, x):
        x = self.input(x)
        x = self.resnet(x)
        x = self.output(x)
        return x
    
    def freeze(self):
        self.input.apply(initialize_weights)
        self.output.apply(initialize_weights)
        for param in self.resnet.parameters():
            param.requires_grad_(False)

    def unfreeze(self):
        for param in self.resnet.parameters():
            param.requires_grad_(True)


if __name__ == "__main__":
    ds = DirectoryRandomDataset("//work//cvcs2024//VisionWise//test")
    dataloader = TransformDataLoader(RandomTransform.GLOBAL_CROP, ds, batch_size=8, 
                                     dataset_mode=DirectoryRandomDataset.BASE, num_workers=2,
                                     pacman=False, transform=WaveletTransform(), num_channels=39)
    