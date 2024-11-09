import pywt
import torch
import torchvision
import numpy as np
from torchvision.transforms import v2
from data.iter_dataset import DirectoryRandomDataset

class WaveletTransform(v2.Transform):
    def __init__(self, wavelet:str = 'haar'):
        """
        A simple class torchvision transform that encapsulate a wavelet transform

        Args:
            - wavelet(str) : wavelet to use. Default 'haar'
        """
        super().__init__()
        self.w = wavelet
    
    def forward(self, image:torch.Tensor) -> torch.Tensor:
        """
        Appliy the 2 dimensionale wavelet transform to all channel separately

        Args:
            - image (torch.Tensor): a torch tensor of shape (B, C, W, H) or (C, W, H)
        
        Returns:
            - a tensor with shape (B, C * 4, ~W/2, ~H/2). Every channel il transformed in 4 tensor wich are:
                - cA = Approx Coefficient
                - ch = Horizontal detail
                - cV = Vertical detail
                - cD = Diagonal detail
        """
        x = image.numpy()
        x_trans = pywt.dwtn(x, self.w, axes=[-2, -1])
        res = []
        if len(x.shape) == 3:
            for c in range(x.shape[0]):
                res.extend((x_trans['aa'][c], x_trans['ad'][c], x_trans['da'][c], x_trans['dd'][c]))
            return torch.Tensor(np.array(res))
        for b in range(x.shape[0]):
            el = []
            for c in range(x.shape[1]):
                el.extend((x_trans['aa'][b, c], x_trans['ad'][b, c], x_trans['da'][b, c], x_trans['dd'][b, c]))
            res.append(el)
        return torch.Tensor(np.array(res))

"""
if __name__ == "__main__":
    ds = DirectoryRandomDataset("//work//cvcs2024//VisionWise//test")
    print("ds aperto")
    img = []
    for i in range(2):
        dim1 = 0
        dim2 = 0
        while dim1 < 200 or dim2 < 200:
            img_t, _ = next(ds.__iter__())
            dim1 = img_t.numpy().shape[-1]
            dim2 = img_t.numpy().shape[-2]
        img.append(img_t[:, :, 0:200, 0:200])
    img = torch.cat(img)
    print(img.shape)
    t = WaveletTransform()
    res = t.forward(img)
    a, (b, c, d) = pywt.dwt2(img[0, 0].numpy(), 'haar')
    print(a)
    print(res[0, 0])
    print(b)
    print(res[0, 1])
    
    #print(res)
"""