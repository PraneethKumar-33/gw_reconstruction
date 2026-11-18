import numpy as np
import torch
from torch.utils.data import Dataset
import os

class GWDataset(Dataset):
    def __init__(self, data_dir):
        # Resolve data_dir relative to cwd or project root
        if not os.path.isabs(data_dir):
            if not os.path.exists(data_dir):
                proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
                candidate = os.path.join(proj_root, data_dir)
                if os.path.exists(candidate):
                    data_dir = candidate

        pairs_path = os.path.join(data_dir, 'pairs.npy')
        if not os.path.exists(pairs_path):
            raise FileNotFoundError(f"Could not locate pairs.npy at '{pairs_path}'")
            
        self.pairs = np.load(pairs_path, allow_pickle=True)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        noisy, clean = self.pairs[idx]
        noisy_tensor = torch.FloatTensor(noisy).unsqueeze(0)
        clean_tensor = torch.FloatTensor(clean).unsqueeze(0)
        return noisy_tensor, clean_tensor
