import numpy as np
import torch
from torch.utils.data import Dataset
import os

class GWDataset(Dataset):
    def __init__(self, data_dir):
        # Expected shape: (N, 2, 4096) or pairs of (noisy, clean)
        pairs_path = os.path.join(data_dir, 'pairs.npy')
        self.pairs = np.load(pairs_path, allow_pickle=True)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        noisy, clean = self.pairs[idx]
        # Shape: [1, 4096]
        noisy_tensor = torch.FloatTensor(noisy).unsqueeze(0)
        clean_tensor = torch.FloatTensor(clean).unsqueeze(0)
        return noisy_tensor, clean_tensor
