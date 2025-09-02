import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

# ---------- 2) Dataset ----------
class RatingsDS(Dataset):
    def __init__(self, u, i, r):
        self.u = torch.from_numpy(u)
        self.i = torch.from_numpy(i)
        self.r = torch.from_numpy(r)

    def __len__(self):
        return len(self.r)

    def __getitem__(self, idx):
        return self.u[idx], self.i[idx], self.r[idx]
    
    
