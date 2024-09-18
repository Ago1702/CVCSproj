import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()