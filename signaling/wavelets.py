import pywt
import torch
import torchvision
import numpy as np
from torchvision.transforms import v2
from data.datasets import DirectoryRandomDataset

class WaveletTransform(v2.Transform):
    def __init__(self, wavelet:str = 'haar', lv:int = 4):
        """
        A simple class torchvision transform that encapsulate a wavelet transform

        Args:
            - wavelet(str) : wavelet to use. Default 'haar'
            - lv(int) : level of multimodal detail
        """
        super().__init__()
        self.w = wavelet
        self.lv = lv
    
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
        if len(x.shape) < 2:
            raise ValueError("Incorrect shape, the tensor must be, at least, bi-dimensional")
        while len(x.shape) < 4:
            x = np.expand_dims(x, 0)

        B, C, W, H = x.shape
        x_trans = pywt.mra2(x, 'haar', level=self.lv, transform='dwt2', mode="symmetric")
        res = [x_trans[0]]
        for tensor in x_trans[1:]:
            res.extend(tensor)
        res = np.concatenate(res, -3)
        return torch.tensor(res)


if __name__ == "__main__":
    ds = DirectoryRandomDataset("//work//cvcs2024//VisionWise//test")
    print("ds aperto")
    img = []
    for i in range(50):
        dim1 = 0
        dim2 = 0
        while dim1 < 200 or dim2 < 200:
            img_t, _ = next(ds.__iter__())
            dim1 = img_t.numpy().shape[-1]
            dim2 = img_t.numpy().shape[-2]
        img.append(img_t[:, :3, :200, :200])
    img = torch.cat(img)
    print(img.shape)
    t = WaveletTransform("db2")
    res = t.forward(img)
    #a, (b, c, d) = pywt.dwt2(img[0, 0].numpy(), 'haar')
    #print(a)
    #print(res[0, 0])
    #print(b)
    #print(res[0, 1])
    
    print(res.shape)