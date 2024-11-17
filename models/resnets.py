import torch
import torch.nn as nn
from torchvision import models
from utils.modules import MultiCBAM
from utils.helpers import state_dict_adapter
import utils.modules as modules
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
    def __init__(self,freeze_resnet = True,weight_file='/work/cvcs2024/VisionWise/weights/checkpoint_res_class_r4_10000.pth'):
        super(v2,self).__init__()
        resnet = v1().cpu()
        cbam = nn.Sequential(
            MultiCBAM(2048, 8),
            MultiCBAM(2048, 16),
            MultiCBAM(2048, 32)
            )
        cbam.apply(modules.initialize_weights)
        resnet.load_state_dict(torch.load(weight_file,weights_only=False)['model'])
        
        resnet_list = list(list(list(resnet.children())[0].children())[0].children())[:-3]
        self.resnet = nn.Sequential(*resnet_list)
        
        if freeze_resnet:
            for param in self.resnet.parameters():
                param.requires_grad = False
                
        self.cbam = cbam
        
        final = nn.Sequential(nn.AvgPool2d(kernel_size=7),nn.Flatten(),nn.Linear(in_features=2048,out_features=1))
        final.apply(modules.initialize_weights)
        self.final = final
        
    def forward(self,x:torch.Tensor):
        x = self.resnet(x)
        x = self.cbam(x)
        x = self.final(x)
        return x
    
class v3(nn.Module):
    def __init__(self):
        super(v3,self).__init__()
        resnet_cbam = v2()
        
        self.resnet_cbam = nn.Sequential(*list(resnet_cbam.children())[:-2])
        final = nn.Sequential(nn.AvgPool2d(kernel_size=7),nn.Flatten(),nn.Linear(in_features=2048,out_features=512))
        final.apply(modules.initialize_weights)
        self.final = final
    def forward(self,x:torch.Tensor):
        x = self.resnet_cbam(x)
        x = self.final(x)
        return x
    
class v4(nn.Module):
    def __init__(self,weight_file='/work/cvcs2024/VisionWise/weights/checkpoint_rescbam_contr_r1_1100.pth'):
        super(v4,self).__init__()

        resnet_cbam_embedder = v3()
        resnet_cbam_embedder.load_state_dict(state_dict_adapter(torch.load(weight_file,weights_only=False)['model'],string_to_remove='module.'))
        for param in resnet_cbam_embedder.parameters():
                param.requires_grad = False
        self.resnet_cbam_embedder = resnet_cbam_embedder

        final = nn.Linear(in_features=512,out_features=1)
        final.apply(modules.initialize_weights)

        self.final = final
    def forward(self,x:torch.Tensor):
        x = self.resnet_cbam_embedder(x)
        x = self.final(x)
        return x
if __name__ == '__main__':
    torch.backends.cudnn.enabled = False
    model = v4().cuda()
    print(model(torch.Tensor(10,3,200,200).cuda()).shape)