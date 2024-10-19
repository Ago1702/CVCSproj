import torch
import io
import glob
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.v2 as transforms
#import matplotlib.pyplot as plt

class FileDataset(Dataset):

    FAKE = 1
    REAL = 0

    def __init__(self, dir:str | Path, ext:str = "png") -> None:
        """
        Dataset class. It retrives images from a particular directory

        Args:
            dir (str | Path): Path to the directory containing all images
        """
        super().__init__()
        if not isinstance(dir, Path):
            dir = Path(dir)
        if not dir.is_dir():
            raise ValueError("The path have to be a directory")
        self.dir = dir
        self.paths = [f for f in dir.iterdir() if f.is_file() and (f.match(f"real-*.{ext}") or f.match(f"fake-*.{ext}"))]
        self.label = [self.REAL if "real" in p.name else self.FAKE for p in self.paths]
        self.tensorizzatore = transforms.Compose([transforms.PILToTensor()])

    def __len__(self):
        """
        Returns Dataset size
        """
        return len(self.paths)
    
    def __getitem__(self, index:int) -> tuple[torch.Tensor, int]:
        """
        Return an image

        Args:
            index (int): Image index

        Returns:
            Tensor: the image
            int: the Label, 1 (self.FAKE) and 0 (self.REAL)
        """
        p = self.paths[index]
        image = Image.open(p).convert("RGB")
        
        return self.tensorizzatore(image), torch.tensor(self.label[index], dtype=torch.long)
