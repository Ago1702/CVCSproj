import torch
import math
import torch.nn as nn
import torch.nn.functional as func

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
            nn.Linear(channels, channels // r),
            nn.ReLU(),
            nn.Linear(channels // r, channels)
        )
        self.pool_types = pool_types

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
        x_out = self.spatial(x)
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
