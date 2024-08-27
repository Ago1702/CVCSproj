import torch
import math
import torch.nn as nn
from moduli.cbam import CBAM

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
# Il risultato finale viene unito eseguendo una media
# 
# L'idea è, di base, quella della GoogleNet 2014
#  

class MultiCBAM(nn.Module):
    def __init__(self, channel:int, r:float | int | list[int|float] = 16):
        super(MultiCBAM, self).__init__()
        if isinstance(r, list):
            self.bam_small = CBAM(channel, 3, r[0])
            self.bam_medium = CBAM(channel, 7, r[1])
            self.bam_large = CBAM(channel, 15, r[2])
        else:
            self.bam_small = CBAM(channel, 3, r)
            self.bam_medium = CBAM(channel, 7, r)
            self.bam_large = CBAM(channel, 15, r)
        self.conv1 = nn.Conv2d(9, 3, 1)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x_small = self.bam_small(x)
        x_medium = self.bam_medium(x)
        x_large = self.bam_large(x)
        x_final = torch.concat([x_small, x_medium, x_large], 1)
        x_final = self.conv1(x_final)
        return x + x_final
        
if __name__ == "__main__":
    net = nn.Sequential(MultiCBAM(3))
    x = torch.rand([1, 3, 4, 4])
    print(x)
    print(net.forward(x))