import torch
import math
import torch.nn as nn
import torch.nn.functional as func

INIT = torch.nn.init.xavier_normal_

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups)
        INIT(self.conv.weight)
        self.bn = nn.BatchNorm2d(out_planes, 1e-5, momentum=0.1, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None
    
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelAttention(nn.Module):
    def __init__(self, channels:int, r=16, pool_types=["avg", "max"]):
        super(ChannelAttention, self).__init__()
        self.gate_channels = channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(channels, max(channels // r,1)),
            nn.ReLU(),
            nn.Linear(max(channels // r,1), channels)
        )
        self.pool_types = pool_types
        for module in self.mlp.children():
            if isinstance(module, nn.Linear):
                INIT(module.weight)

    def forward(self, x:torch.Tensor):
        spacial_descr = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = func.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = func.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = func.lp_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type =='lse':
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if spacial_descr is None:
                spacial_descr = channel_att_raw
            else:
                spacial_descr += channel_att_raw
        
        scale = func.sigmoid(spacial_descr).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale
                
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1)//2, relu=False)
    
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = func.sigmoid(x_out)
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, kernel_size = 7, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(gate_channels, reduction_ratio, pool_types)
        self.spatial_att = SpatialAttention(kernel_size) if not no_spatial else None
    
    def forward(self, x):
        x_out = self.channel_att(x)
        if self.spatial_att is not None:
            x_out = self.spatial_att(x_out)
        return x_out

# 
# Idea di base
# Usare 3 moduli di attenzione differenti
# Questi hanno, per l'elemento, spaziale una differente dimensionalità
# In questo modo si prova a concentrarsi su caratteristiche di dimensione differente
# Questo perchè, a differenza di un vero blocco di attenzione,
# CBAM usa delle CNN per capire dove concentrarsi.
# Usando 3 dimensioni di kernel diverse si punta a dare una visione:
# - Specifica   dim = (3, 3)
# - Generale    dim = (15, 15)
# - Intermedia  dim = (7, 7)
# del fenomeno
# 
# !!Questo elemento deve ancora essere valutato!!
# Il risultato finale viene unito concatenando ed eseguendo una 1*1 conv.
# 
# L'idea è, di base, quella della GoogleNet 2014
#  

class MultiCBAM(nn.Module):
    def __init__(self, channel:int, r:float | int | list[int|float] = 16):
        super(MultiCBAM, self).__init__()
        if isinstance(r, list):
            self.bam_small = CBAM(channel, kernel_size=3, reduction_ratio=r[0])
            self.bam_medium = CBAM(channel, kernel_size=7, reduction_ratio=r[1])
            #self.bam_large = CBAM(channel, kernel_size=15, reduction_ratio=r[2])
        else:
            self.bam_small = CBAM(channel, kernel_size=3, reduction_ratio=r)
            self.bam_medium = CBAM(channel, kernel_size=7, reduction_ratio=r)
            #self.bam_large = CBAM(channel, kernel_size=15, reduction_ratio=r)
        self.conv1 = nn.Conv2d(in_channels=channel * 2, out_channels=channel, kernel_size=1)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x_small = self.bam_small(x)
        x_medium = self.bam_medium(x)
        #x_large = self.bam_large(x)
        x_final = torch.concat([x_small, x_medium], 1)
        x_final = self.conv1(x_final)
        return x + x_final
    
class ChannelMultiCBAM(nn.Module):
    def __init__(self, channel:int, r:float | int | list[int|float] = 16):
        super(ChannelMultiCBAM, self).__init__()
        if isinstance(r, list):
            self.bam_small = CBAM(channel, kernel_size=3, reduction_ratio=r[0],no_spatial=True)
            self.bam_medium = CBAM(channel, kernel_size=7, reduction_ratio=r[1],no_spatial=True)
            self.bam_large = CBAM(channel, kernel_size=15, reduction_ratio=r[2],no_spatial=True)
        else:
            self.bam_small = CBAM(channel, kernel_size=3, reduction_ratio=r,no_spatial=True)
            self.bam_medium = CBAM(channel, kernel_size=7, reduction_ratio=r,no_spatial=True)
            self.bam_large = CBAM(channel, kernel_size=15, reduction_ratio=r,no_spatial=True)
        self.conv1 = nn.Conv2d(in_channels=channel * 3, out_channels=channel, kernel_size=1)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x_small = self.bam_small(x)
        x_medium = self.bam_medium(x)
        x_large = self.bam_large(x)
        x_final = torch.concat([x_small, x_medium, x_large], 1)
        x_final = self.conv1(x_final)
        return x + x_final
    

if __name__ == '__main__':
    test = ChannelAttention(3)
    tensor = torch.zeros(1,3,200,200)
    print(test(tensor).shape)
        