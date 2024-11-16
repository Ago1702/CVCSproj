import torch
import torch.nn as nn
from torchvision import models

class v1(nn.Module):
    def __init__(self):
        super(v1,self).__init__()
        resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        resnet_children = list(resnet.children())[:-1]
        resnet_children.append(nn.Flatten())
        resnet_children.append(nn.Linear(in_features=2048,out_features=1))
        self.resnet = nn.DataParallel(nn.Sequential(*resnet_children)).cuda()
        
    def forward(self,x:torch.Tensor):
        return self.resnet(x)
    
class v2(nn.Module):
    def __init__(self):
        super(v2,self).__init__()
        resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        resnet_children = list(resnet.children())[:-1]
        resnet_children.append(nn.Flatten())
        resnet_children.append(nn.Linear(in_features=2048,out_features=1))
        self.resnet = nn.DataParallel(nn.Sequential(*resnet_children)).cuda()
        
    def forward(self,x:torch.Tensor):
        return self.resnet(x)
if __name__ == '__main__':
    model = v1()
    print(model(torch.Tensor(10,3,200,200).cuda()).shape)