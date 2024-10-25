import torch
import math
import torch.nn as nn
from utils.modules import MultiCBAM, CBAM

class DummyCBAM(nn.Module):
    def __init__(self):
        super(DummyCBAM,self).__init__()
        self.net = nn.Sequential(
        nn.Conv2d(3, 9, 3, padding=2),
        nn.Conv2d(9, 27, 4, padding=3),
        nn.Conv2d(27, 81, 5, padding=4),
        MultiCBAM(81, 8),
        MultiCBAM(81, 16),
        MultiCBAM(81, 32),
        nn.Conv2d(81, 27, 5),
        nn.Conv2d(27, 9, 4),
        nn.Conv2d(9, 3, 3),
        nn.Conv2d(3, 1, 3)
    )
        
    def forward(self, image):
        return self.net.forward(image)
    
#test code
'''
if __name__ == "__main__":
    net = nn.Sequential(
        nn.Conv2d(3, 9, 3, padding=2),
        nn.Conv2d(9, 27, 4, padding=3),
        nn.Conv2d(27, 81, 5, padding=4),
        MultiCBAM(81, 8),
        MultiCBAM(81, 16),
        MultiCBAM(81, 32),
        nn.Conv2d(81, 27, 5),
        nn.Conv2d(27, 9, 4),
        nn.Conv2d(9, 3, 3),
        nn.Conv2d(3, 1, 3)
    )
    x = torch.rand((2, 3, 120, 120))
    x = net.forward(x)
    print(x.shape)
'''
