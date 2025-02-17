# data_pre_16class.py

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

############################################
# 1) 配置增广 Pipeline (可按需修改)
############################################
transform = A.Compose([
    A.Resize(456, 456),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

def preprocess_one_image(image_path):
    """读图 -> albumentations -> 返回 (C,H,W) tensor. 若读失败返回 None."""
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[警告] 无法读取图像: {image_path}")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transformed = transform(image=image)
    return transformed["image"]  # (C,H,W)

def preprocess_stereo(left_path, right_path):
    """拼接左右眼 -> (6,H,W). 若任意一眼读失败返回 None."""
    left_img  = preprocess_one_image(left_path)
    right_img = preprocess_one_image(right_path)
    if left_img is None or right_img is None:
        return None
    merged = np.concatenate((left_img, right_img), axis=0)  # shape: (6,H,W)
    return merged

############################################
# 2) 处理 CSV 中每行, 生成 {filename: [16维标签]}
############################################
def process_dataset_16class(input_dir, output_dir, csv_path):
    """
    假设 csv 有这些列:
      ID, Age, Sex, 
      Left-Fundus, Right-Fundus,
      Left-N, Left-D, Left-G, Left-C, Left-A, Left-H, Left-M, Left-O,
      Right-N, Right-D, Right-G, Right-C, Right-A, Right-H, Right-M, Right-O

    输入:
      input_dir:  存放图像 (xxx_left.jpg, xxx_right.jpg)
      output_dir: 存放生成的 .npy
      csv_path:   Traning_Dataset_2.csv (可能gbk编码)
    返回:
      label_dict: { "xxx.npy": [16维], ... }
    """
    # 如果出现编码错误, 尝试 encoding='gbk'
    df = pd.read_csv(csv_path, encoding='gbk')

    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_dict = {}

    # 遍历每行
    for _, row in df.iterrows():
        left_img  = str(row["Left-Fundus"])
        right_img = str(row["Right-Fundus"])

        # 左眼8维: [Left-N, Left-D, ... Left-O]
        left_8 = [
            row["Left-N"], row["Left-D"], row["Left-G"], row["Left-C"],
            row["Left-A"], row["Left-H"], row["Left-M"], row["Left-O"]
        ]
        # 右眼8维: [Right-N, Right-D, ... Right-O]
        right_8 = [
            row["Right-N"], row["Right-D"], row["Right-G"], row["Right-C"],
            row["Right-A"], row["Right-H"], row["Right-M"], row["Right-O"]
        ]
        label_16 = list(left_8) + list(right_8)

        # 生成 .npy 文件名
        # 假设 left_img 形如 "0_left.jpg", 则 .npy -> "0.npy"
        if "_left.jpg" in left_img:
            npy_name = left_img.replace("_left.jpg", ".npy")
        else:
            # 若命名不符合, 需自定义
            npy_name = left_img + ".npy"

        left_path  = input_dir / left_img
        right_path = input_dir / right_img

        merged = preprocess_stereo(left_path, right_path)
        if merged is None:
            continue  # 跳过无法处理的

        # 保存 .npy
        np.save(output_dir / npy_name, merged)
        # 记录标签
        label_dict[npy_name] = label_16

    return label_dict

if __name__ == "__main__":
    # 示例用法
    csv_path   = "dataset/Traning_Dataset_2.csv"
    input_dir  = "dataset/Training_Dataset"
    output_dir = "dataset/output_16"

    label_dict = process_dataset_16class(input_dir, output_dir, csv_path)
    print(f"生成 .npy 数量: {len(label_dict)}")

    # 保存 label_dict 以备训练
    np.save(output_dir + "/label_dict_16.npy", label_dict, allow_pickle=True)
    print("已保存 label_dict_16.npy")
