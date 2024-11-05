import torch
import torch.nn as nn
from torchvision import models
from utils.modules import SpatialMultiCBAM,ChannelMultiCBAM

class v1(nn.Module):
    def __init__(self):
        super(v1,self).__init__()

        #declaration of the layers
        resnet =  models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        '''for param in resnet.layer1.parameters():
            param.requires_grad = False

        for param in resnet.layer2.parameters():
            param.requires_grad = False'''

        self.resnet = nn.Sequential(*(list(resnet.children())[:-2]))
        self.cbam = nn.Sequential(
            SpatialMultiCBAM(2048, 8),
            SpatialMultiCBAM(2048, 16),
            SpatialMultiCBAM(2048, 32)
            )
        self.flatten = nn.Flatten()

        

    def forward(self,x:torch.Tensor):
        x = self.resnet(x)
        #x = self.cbam(x)
        #x = self.flatten(x)
        return x

class v2(nn.Module):
    def __init__(self):
        super(v2, self).__init__()
        self.spatial_cbam = nn.Sequential(
            SpatialMultiCBAM(3, 8),
            SpatialMultiCBAM(3, 16),
            SpatialMultiCBAM(3, 32)
            )
        
        resnet =  models.resnet50()
        #resnet =  models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.resnet = nn.Sequential(*(list(resnet.children())[:-1]))

        self.channel_cbam = nn.Sequential(
            ChannelMultiCBAM(2048, 8),
            ChannelMultiCBAM(2048, 16),
            ChannelMultiCBAM(2048, 32)
            )
        
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=2048,out_features=512)
        )
    def forward(self,x:torch.Tensor):
        x = self.spatial_cbam(x)
        x = self.resnet(x)
        x = self.channel_cbam(x)
        x = self.dense(x)
        return x
    
if __name__ == '__main__':
    sandwich_net = v2()
    print(sandwich_net(torch.zeros(10, 3, 200, 200)).shape)