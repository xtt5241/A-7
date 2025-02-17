# dataset_8class.py

import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np

class EyeDataset8Class(Dataset):
    def __init__(self, npy_dir, label_dict):
        """
        npy_dir: 存放 .npy 文件的目录
        label_dict: { filename.npy: [8dim_label], ... }
        """
        self.npy_dir = Path(npy_dir)
        self.label_dict = label_dict  # filename->list of 8(0/1)

        # 收集所有npy文件, 并过滤只保留出现在 label_dict 的
        self.image_files = []
        for f in self.npy_dir.glob("*.npy"):
            if f.name in self.label_dict:
                self.image_files.append(f)

        print(f"[EyeDataset8Class] 找到 .npy 文件数: {len(self.image_files)}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fpath = self.image_files[idx]
        # 读取 .npy -> (6,H,W)
        arr = np.load(str(fpath))  # shape(6,H,W)
        # 转成 torch
        image_tensor = torch.tensor(arr, dtype=torch.float32)

        # 对应的 8维标签
        label_8 = self.label_dict[fpath.name]
        label_8_tensor = torch.tensor(label_8, dtype=torch.float32)

        return image_tensor, label_8_tensor, fpath.name
