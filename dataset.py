import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# 1️⃣ 读取 Excel 获取标签
def load_labels(excel_path):
    df = pd.read_excel(excel_path)
    labels = {}
    for _, row in df.iterrows():
        left_img = row["Left-Fundus"]
        right_img = row["Right-Fundus"]
        left_label = row["Left-Diagnostic Keywords"]
        right_label = row["Right-Diagnostic Keywords"]

        labels[left_img.replace("_left.jpg", ".npy")] = left_label
        labels[right_img.replace("_right.jpg", ".npy")] = right_label
    return labels

# 2️⃣ 自定义 Dataset
class EyeDataset(Dataset):
    def __init__(self, data_dir, excel_path):
        self.data_dir = Path(data_dir)
        self.labels = load_labels(excel_path)
        self.image_files = list(self.data_dir.glob("*.npy"))

        self.classes = ["normal fundus", "diabetic retinopathy", "glaucoma", "cataract",
                        "age-related macular degeneration", "hypertensive retinopathy", "myopia", "other"]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = np.load(image_path)
        image = torch.tensor(image, dtype=torch.float32)

        label_name = self.labels[image_path.name].lower().strip().split(",")[0]
        label = self.classes.index(label_name) if label_name in self.classes else self.classes.index("other")

        return image, label

# 3️⃣ 训练集 / 验证集划分
excel_path = "dataset\Traning_Dataset.xlsx"
data_dir = "dataset\output"
dataset = EyeDataset(data_dir, excel_path)

train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
train_loader = DataLoader(Subset(dataset, train_idx), batch_size=8, shuffle=True)
val_loader = DataLoader(Subset(dataset, val_idx), batch_size=8, shuffle=False)
