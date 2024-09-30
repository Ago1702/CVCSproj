import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, IterableDataset
import requests
import torchvision.transforms.v2 as transforms
from datasets import load_dataset
from torchvision.io import read_image
from PIL import Image

class BufferedIterable(IterableDataset):
    """
    A wrapper for Iterable Dataset, implements a buffer to simulate a local shuffling

    suggested by:
        sharvil Sharvil Nanavati: https://discuss.pytorch.org/t/how-to-shuffle-an-iterable-dataset/64130/4

    Args:
            buffer_size (int)
    """    
    def __init__(self, buffer_size:int):
        super(BufferedIterable).__init__()
        self.transform = transforms.Compose(transforms.PILToTensor())
        self.dataset = load_dataset("elsaEU/ELSA1M_track1", split="train", streaming=True)
        self.buffer_size = buffer_size
    
    def __iter__(self) -> np.Iterator:
        buff = []
        try:
            dataset_iter = iter(self.dataset)
            for i in range(self.buffer_size):
                buff.append(next(dataset_iter)["image"])
        except:
            self.buffer_size = len(buff)
        
        try:
            while True:
                try:
                    item = next(dataset_iter)["image"]
                    rem_idx = random.randint(0, self.buffer_size - 1)
                    yield buff[rem_idx]
                    buff[rem_idx] = item
                except StopIteration:
                    break
            while len(buff) > 0:
                yield buff.pop()
        except GeneratorExit:
            pass