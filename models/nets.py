import torch
import torch.nn as nn
from torchvision import models
from utils.modules import MultiCBAM
from utils.modules import initialize_weights
from utils.helpers import load_checkpoint

import transformers
from huggingface_hub import hf_hub_download
from PIL import Image
import torch
import torch.nn as nn
import joblib
from torchvision import transforms
import signaling.signalnet

class Complete_Module(nn.Module):
    def __init__(self,name:str = 'Unimplemented Complete Module'):
        super(Complete_Module,self).__init__()
        self.name = name
        
    def forward(self,x:torch):
        return x
    
    def test_net(self,size=(10,3,224,224)):
        x = torch.Tensor(size=size)
        print(self.name + '\'s output shape --> '+str(self.forward(x).shape))
        
class vanilla_resnet_classifier_152(Complete_Module):
    def __init__(self,name:str = 'Vanilla Resnet152 Classifier'):
        super(vanilla_resnet_classifier_152 ,self).__init__(name)
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
    
class vanilla_resnet_classifier_50(Complete_Module):
    def __init__(self,name:str = 'Vanilla Resnet50 Classifier'):
        super(vanilla_resnet_classifier_50 ,self).__init__(name)
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
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
    
class cbam_classifier_50(Complete_Module):
    def __init__(self,name = 'Resnet50-CBAM Classifier',freeze_mode = 'resnet',drop_classifier = False):
        super(cbam_classifier_50,self).__init__(name)
        self.drop_classifier = drop_classifier
        
        vanilla = vanilla_resnet_classifier_50()
        load_checkpoint('ch_vanilla_resnet50_classifier',model=vanilla,un_parallelize=True)
        children = list(vanilla.children())
        self.resnet = nn.Sequential(*list(children[0].children())[:-1])
        self.cbam = nn.Sequential(
            MultiCBAM(2048,8),
            MultiCBAM(2048,16),
            MultiCBAM(2048,32)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flatten = nn.Flatten()
        self.embedder = nn.Linear(in_features=2048,out_features=512)
        self.classifier = nn.Linear(in_features=512,out_features=1)
        
        if freeze_mode == 'none':
            pass
        elif freeze_mode == 'resnet':
            for param in self.resnet.parameters():
                param.requires_grad = False
        elif freeze_mode == 'embedder':
            for param in self.resnet.parameters():
                param.requires_grad = False
            for param in self.cbam.parameters():
                param.requires_grad = False
            for param in self.embedder.parameters():
                param.requires_grad = False
        else:
            raise RuntimeError('Invalid freeze_mode parameter: ' + freeze_mode)
        
        self.cbam.apply(initialize_weights)
        self.embedder.apply(initialize_weights)
        self.classifier.apply(initialize_weights)

        
    def forward(self,x:torch.Tensor):
        x = self.resnet(x)
        x = self.cbam(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.embedder(x)
        if not self.drop_classifier:
            x = self.classifier(x)
        return x
    
class cbam_classifier_152(Complete_Module):
    def __init__(self,name = 'Resnet152-CBAM Classifier',freeze_mode = 'resnet',drop_classifier = False,load_checkpoints = True):
        super(cbam_classifier_152,self).__init__(name)
        
        self.drop_classifier = drop_classifier
        vanilla = vanilla_resnet_classifier_152()
        if load_checkpoints:
            load_checkpoint('ch_vanilla_resnet152_classifier',model=vanilla,un_parallelize=True)
        children = list(vanilla.children())
        self.resnet = nn.Sequential(*list(children[0].children())[:-1])
        self.cbam = nn.Sequential(
            MultiCBAM(2048,8),
            MultiCBAM(2048,16),
            MultiCBAM(2048,32)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flatten = nn.Flatten()
        self.embedder = nn.Linear(in_features=2048,out_features=512)
        self.classifier = nn.Linear(in_features=512,out_features=1)
        
        if freeze_mode == 'none':
            pass
        elif freeze_mode == 'resnet':
            for param in self.resnet.parameters():
                param.requires_grad = False
        elif freeze_mode == 'embedder':
            for param in self.resnet.parameters():
                param.requires_grad = False
            for param in self.cbam.parameters():
                param.requires_grad = False
            for param in self.embedder.parameters():
                param.requires_grad = False
            for param in self.classifier.parameters():
                param.requires_grad = True
        else:
            raise RuntimeError('Invalid freeze_mode parameter: ' + freeze_mode)
                
        self.cbam.apply(initialize_weights)
        self.embedder.apply(initialize_weights)
        self.classifier.apply(initialize_weights)

        
    def forward(self,x:torch.Tensor):
        x = self.resnet(x)
        x = self.cbam(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.embedder(x)
        if not self.drop_classifier:
            x = self.classifier(x)
        return x
    
class ViT_n(Complete_Module):
    def __init__(self, num_channels=39, num_classes=1,name='Visual Transformer'):
        super(ViT_n, self).__init__(name=name)
        
        # Load pre-trained ViT model
        self.vit = models.vision_transformer.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

        # Modify the first layer to accept 39-channel input instead of 3 (RGB)
        self.vit.conv_proj = nn.Conv2d(num_channels, 768, kernel_size=(16, 16), stride=(16, 16))

        # Modify the classifier head for binary classification (2 classes)
        self.vit.heads = nn.Linear(768, num_classes)

    def forward(self, x):
        return self.vit(x)
    
class ViT_3(Complete_Module):
    def __init__(self,num_classes=1,name='Visual Transformer'):
        super(ViT_3, self).__init__(name=name)
        
        self.vit = models.vision_transformer.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

        self.vit.heads = nn.Linear(768, num_classes)

    def forward(self, x):
        return self.vit(x)
    
    

class VITContrastiveHF(nn.Module):
    """
    This class is a wrapper for the CoDE model. It is used to load the model and the classifier
    """

    def __init__(self, classificator_type):
        """
        Constructor of the class
        :param repo_name: the name of the repository
        :param classificator_type: the type of the classifier
        """
        super(VITContrastiveHF, self).__init__()
        self.model = transformers.AutoModel.from_pretrained("aimagelab/CoDE")
        self.model.pooler = nn.Identity()
        self.classificator_type = classificator_type
        self.processor = transformers.AutoProcessor.from_pretrained("aimagelab/CoDE")
        self.processor.do_resize = False
        # define the correct classifier /// consider to use the `cache_dir`` parameter
        if classificator_type == "svm":
            file_path = hf_hub_download(
                repo_id="aimagelab/CoDE",
                filename="sklearn/ocsvm_kernel_poly_gamma_auto_nu_0_1_crop.joblib",
            )
            self.classifier = joblib.load(file_path)

        elif classificator_type == "linear":
            file_path = hf_hub_download(
                repo_id="aimagelab/CoDE",
                filename="sklearn/linear_tot_classifier_epoch-32.sav",
            )
            self.classifier = joblib.load(file_path)

        elif classificator_type == "knn":
            file_path = hf_hub_download(
                repo_id="aimagelab/CoDE",
                filename="sklearn/knn_tot_classifier_epoch-32.sav",
            )
            self.classifier = joblib.load(file_path)

        else:
            raise ValueError("Selected an invalid classifier")

    def forward(self, x, return_feature=False):
        features = self.model(x)
        if return_feature:
            return features
        features = features.last_hidden_state[:, 0, :].cpu().detach().numpy()
        predictions = self.classifier.predict(features)
        return torch.from_numpy(predictions)

class SuperEnsemble(Complete_Module):
    def __init__(self,name='SuperEnsemble',load_checkpoints = False,freeze_models = True):
        super(SuperEnsemble,self).__init__(name=name)
        self.transformer = ViT_3()
        self.cbam = cbam_classifier_152(load_checkpoints=False)
        self.signal_net = signaling.signalnet.SignalNet(do_wavelet_transform=True)
        
        if load_checkpoints:
            ch_1 = load_checkpoint('vit_classifier',model=self.transformer,iteration_index = 10000,un_parallelize=True)
            ch_2 = load_checkpoint('ch_cbam152_contrastive',model=self.cbam,iteration_index=4000,un_parallelize=True)
            ch_3 = load_checkpoint('wave50trans_classifier',model=self.signal_net,un_parallelize=True)
            print(f'loader default checkpoints: {ch_1}, {ch_2}, {ch_3}')
            
        if freeze_models:
            for param in self.transformer.parameters():
                param.requires_grad = False
            for param in self.cbam.parameters():
                param.requires_grad = False
            for param in self.signal_net.parameters():
                param.requires_grad = False
        self.final = nn.Sequential(nn.Linear(in_features=3,out_features=3),nn.ReLU(),nn.Linear(in_features=3,out_features=1))
        self.final.apply(initialize_weights)
    def forward(self,x:torch.Tensor):
        x1 = self.transformer(x)
        x2 = self.cbam(x)
        x3 = self.signal_net(x)
        out = torch.cat([x1,x2,x3],dim=1)
        return self.final(out)

    
if __name__ == '__main__':
    test = SuperEnsemble(load_checkpoints=True)
    test.test_net(size=(10,3,224,224))