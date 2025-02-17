# dataset_16class.py

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path

class EyeDataset16Class(Dataset):
    def __init__(self, npy_dir, label_dict):
        """
        npy_dir:  "dataset/output_16"
        label_dict: { "xxx.npy": [16-dim], ... }
        """
        self.npy_dir = Path(npy_dir)
        self.label_dict = label_dict
        self.image_files = []

        # 只保留在 label_dict 中出现的文件
        for f in self.npy_dir.glob("*.npy"):
            if f.name in label_dict:
                self.image_files.append(f)

        print(f"[EyeDataset16Class] 文件数: {len(self.image_files)}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fpath = self.image_files[idx]
        arr = np.load(str(fpath))  # shape(6,H,W)
        image_tensor = torch.tensor(arr, dtype=torch.float32)

        label_16 = self.label_dict[fpath.name]  # e.g. [0,1,0,0,0,1,0,0, 1,0,0,0,0,0,1,0]
        label_tensor = torch.tensor(label_16, dtype=torch.float32)

        return image_tensor, label_tensor, fpath.name
