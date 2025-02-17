import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# 1️⃣ 读取 98 个关键词到 8 类的映射
def load_keyword_mapping(mapping_path):
    mapping_df = pd.read_csv(mapping_path, encoding='gbk')

    # 🚀 **确保跳过第一行标题**
    if "English Keyword" in mapping_df.iloc[0].values:
        mapping_df = mapping_df.iloc[1:].reset_index(drop=True)

    keyword_to_category = {
        row["English Keyword"].lower().strip(): row["对应的疾病类型"].strip()
        for _, row in mapping_df.iterrows()
    }
    
    keywords = list(keyword_to_category.keys())  # 98 个关键词
    return keyword_to_category, keywords


# 2️⃣ 读取 Excel 获取图像标签，并转换为 98 关键词格式
def load_labels(excel_path, keyword_to_category):
    df = pd.read_excel(excel_path)
    labels = {}

    for _, row in df.iterrows():
        for eye in ['Left', 'Right']:
            img_name = row[f"{eye}-Fundus"].replace(f"_{eye.lower()}.jpg", ".npy")
            diagnostic_keywords = row[f"{eye}-Diagnostic Keywords"].lower().strip().split(',')

            # 过滤出在 mapping 里的关键词
            image_keywords = [kw.strip() for kw in diagnostic_keywords if kw.strip() in keyword_to_category]

            # 🚀 **重要：如果没有匹配的关键词，不要加 `other`**
            if not image_keywords:
                image_keywords = []

            labels[img_name] = image_keywords

    return labels

# 3️⃣ 自定义数据集
class EyeDatasetMultiLabel(Dataset):
    def __init__(self, data_dir, excel_path, mapping_path):
        self.data_dir = Path(data_dir)
        self.keyword_to_category, self.keywords_list = load_keyword_mapping(mapping_path)
        self.labels_dict = load_labels(excel_path, self.keyword_to_category)
        self.image_files = list(self.data_dir.glob("*.npy"))

        # 🚀 关键修改：`MultiLabelBinarizer` 只使用 98 个类别
        self.mlb = MultiLabelBinarizer(classes=self.keywords_list)  
        self.mlb.fit([self.keywords_list])  # 让 `mlb.classes_` 自动获取 98 类

        print(f"✅ `data_mul.py` 关键词数: {len(self.mlb.classes_)}")  # 🚀 打印类别数，确保和 `model.py` 一致

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = np.load(image_path)
        image = torch.tensor(image, dtype=torch.float32)

        # 获取 98 关键词标签
        image_keywords = self.labels_dict.get(image_path.name, [])
        label = torch.tensor(self.mlb.transform([image_keywords])[0], dtype=torch.float32)

        return image, label

# 4️⃣ 训练集 / 验证集划分
excel_path = "dataset\Traning_Dataset.xlsx"
mapping_path = "dataset\English-Chinese_Disease_Mapping.csv"
data_dir = "dataset\output"

dataset = EyeDatasetMultiLabel(data_dir, excel_path, mapping_path)

train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
train_loader = DataLoader(
    Subset(dataset, train_idx), batch_size=8, shuffle=True, num_workers=0, pin_memory=True
)
val_loader = DataLoader(
    Subset(dataset, val_idx), batch_size=8, shuffle=False, num_workers=0, pin_memory=True
)

