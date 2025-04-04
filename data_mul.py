# data_mul.py

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer

def load_mapping(mapping_path):
    """
    从 CSV 读取关键词->疾病类型映射，并收集所有关键词和所有可能的疾病类型。
    假设总共有98个关键字，8~10个疾病类型（具体数量看文件）。
    """
    df = pd.read_csv(mapping_path, encoding='gbk')

    # 若首行是标题，则跳过
    if "English Keyword" in df.iloc[0].values:
        df = df.iloc[1:].reset_index(drop=True)

    keyword2cat = {}
    categories_set = set()

    for _, row in df.iterrows():
        kw = str(row["English Keyword"]).lower().strip()
        cat = str(row["对应的疾病类型"]).strip()
        keyword2cat[kw] = cat
        categories_set.add(cat)

    keywords_list = list(keyword2cat.keys())       # 98 个关键词
    categories_list = sorted(list(categories_set)) # 8~10 个大类

    return keyword2cat, keywords_list, categories_list

def load_labels(excel_path, keyword2cat):
    """
    从 Excel 中读取 (Left-Fundus, Right-Fundus) -> (诊断关键词) -> (对应的疾病类型)
    并返回字典: { img_name.npy : [关键词列表], ... }, { img_name.npy : [类别列表], ... }
    """
    df = pd.read_excel(excel_path)
    img2kws = {}
    img2cats = {}

    for _, row in df.iterrows():
        for eye in ["Left", "Right"]:
            fundus_name = str(row[f"{eye}-Fundus"])  # 如 "0_left.jpg"
            npy_name = fundus_name.replace(f"_{eye.lower()}.jpg", ".npy")  # 如 "0_left.jpg" -> "0.npy"

            diag_str = row[f"{eye}-Diagnostic Keywords"]
            if isinstance(diag_str, str):
                diag_list = [x.strip().lower() for x in diag_str.split(',')]
            else:
                diag_list = []

            valid_kws = []
            valid_cats = set()
            for kw in diag_list:
                if kw in keyword2cat:
                    valid_kws.append(kw)
                    valid_cats.add(keyword2cat[kw])

            img2kws[npy_name] = valid_kws
            img2cats[npy_name] = list(valid_cats)

    return img2kws, img2cats

class EyeDatasetMultiTask(Dataset):
    """
    多标签 + 多类别:
    返回 (图像[6通道], 关键词多标签[98维], 疾病类型多标签[8~10维], 文件名)
    """
    def __init__(self, data_dir, excel_path, mapping_path):
        self.data_dir = Path(data_dir)

        # 1) 加载映射
        self.keyword2cat, self.keywords_list, self.categories_list = load_mapping(mapping_path)

        # 2) 从 Excel 中读出每张图对应的关键词/类别
        self.img2kws, self.img2cats = load_labels(excel_path, self.keyword2cat)

        # 3) 收集所有 .npy 文件
        self.image_files = list(self.data_dir.glob("*.npy"))

        # 4) 初始化多标签 binarizer
        self.mlb_kws = MultiLabelBinarizer(classes=self.keywords_list)
        self.mlb_kws.fit([self.keywords_list])

        self.mlb_cats = MultiLabelBinarizer(classes=self.categories_list)
        self.mlb_cats.fit([self.categories_list])

        print(f"[EyeDatasetMultiTask] 图像数: {len(self.image_files)}")
        print(f"[EyeDatasetMultiTask] 关键词数: {len(self.keywords_list)}, 疾病类别数: {len(self.categories_list)}")
        # 在这里插入一条打印，查看具体有哪些类别
        print(">>> All categories list:", self.categories_list)

        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = np.load(str(image_path))  # shape: (6, H, W)
        image_tensor = torch.tensor(image, dtype=torch.float32)

        fname = image_path.name
        kw_list = self.img2kws.get(fname, [])
        cat_list = self.img2cats.get(fname, [])

        # 转多标签 0/1
        label_kws = torch.tensor(self.mlb_kws.transform([kw_list])[0], dtype=torch.float32)
        label_cats = torch.tensor(self.mlb_cats.transform([cat_list])[0], dtype=torch.float32)

        return image_tensor, label_kws, label_cats, fname
