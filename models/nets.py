import torch
import torch.nn as nn
from torchvision import models
from utils.modules import MultiCBAM
from utils.modules import initialize_weights
class Complete_Module(nn.Module):
    def __init__(self,name:str = 'Unimplemented Complete Module'):
        super(Complete_Module,self).__init__()
        self.name = name
        
    def forward(self,x:torch):
        return x
    
    def test_net(self):
        x = torch.Tensor(size=(10,3,224,224))
        print(self.name + '\'s output shape --> '+str(self.forward(x).shape))
        
class vanilla_resnet_classifier(Complete_Module):
    def __init__(self,name:str = 'Vanilla Resnet Classifier'):
        super(vanilla_resnet_classifier ,self).__init__(name)
        resnet = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
        resnet_children = list(resnet.children())[:-1]

        self.resnet=nn.Sequential(*resnet_children)
        self.flatten = nn.Flatten()
        self.embedder = nn.Linear(in_features=2048,out_features=512)
        self.classifier = nn.Linear(in_features=512,out_features=1)
        
        self.embedder.apply(initialize_weights)
        self.classifier.apply(initialize_weights)
        
    def forward(self,x:torch.Tensor):
        x = self.resnet(x)
        x = self.flatten(x)
        x = self.embedder(x)
        x = self.classifier(x)
        return x
    
class cbam_classifier(Complete_Module):
    def __init__(self,name = 'Resnet-CBAM Classifier'):
        super(cbam_classifier,self).__init__(name)
        resnet = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
        resnet_children = list(resnet.children())[:-1]

        self.resnet=nn.Sequential(*resnet_children)
        self.cbam = nn.Sequential(
            MultiCBAM(2048,8),
            MultiCBAM(2048,16),
            MultiCBAM(2048,32)
        )
        self.flatten = nn.Flatten()
        self.embedder = nn.Linear(in_features=2048,out_features=512)
        self.classifier = nn.Linear(in_features=512,out_features=1)
        
        self.cbam.apply(initialize_weights)
        self.embedder.apply(initialize_weights)
        self.classifier.apply(initialize_weights)
        
    def forward(self,x:torch.Tensor):
        x = self.resnet(x)
        x = self.cbam(x)
        x = self.flatten(x)
        x = self.embedder(x)
        x = self.classifier(x)
        return x
if __name__ == '__main__':
    test = cbam_classifier()
    test.test_net()