# data_pre_8class.py

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2

######################################
# 1) 配置增广 Pipeline (可按需修改)
######################################
transform = A.Compose([
    A.Resize(456, 456),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

######################################
# 2) 从 Excel 读取 (Left-Fundus, Right-Fundus) + 8列标签
######################################
def load_8class_labels(excel_path):
    """
    假设 Excel 中的列名如下:
    ID, Patient Age, Patient Sex, Left-Fundus, Right-Fundus, N, D, G, C, A, H, M, O
    (中间可能还有 Diagnostic Keywords, 但我们不管)
    我们只读取 Left-Fundus, Right-Fundus, [N, D, G, C, A, H, M, O].
    """
    df = pd.read_excel(excel_path)
    # 假设列名和顺序固定, 如果不一样, 需要自己改
    # 例如:
    # Left-Fundus -> df["Left-Fundus"]
    # Right-Fundus -> df["Right-Fundus"]
    # N -> df["N"], D->df["D"], ...
    # 这里简化处理, 直接返回一个 list of rows
    data_list = []
    for _, row in df.iterrows():
        left_img = str(row["Left-Fundus"])
        right_img = str(row["Right-Fundus"])
        label_8 = [
            row["N"], row["D"], row["G"], row["C"],
            row["A"], row["H"], row["M"], row["O"]
        ]
        # label_8 形如 [0, 0, 0, 1, 0, 0, 0, 0]
        data_list.append((left_img, right_img, label_8))
    return data_list

######################################
# 3) 预处理单张图像
######################################
def preprocess_one_image(image_path):
    """
    读图 -> albumentations处理 -> 返回 (C,H,W) tensor
    如果图像读失败, 返回None
    """
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[警告] 无法读取图像: {image_path}")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transformed = transform(image=image)
    return transformed["image"]  # (C,H,W) tensor

######################################
# 4) 拼接左右眼图像
######################################
def preprocess_stereo(left_path, right_path):
    left_img = preprocess_one_image(left_path)
    right_img = preprocess_one_image(right_path)
    if left_img is None or right_img is None:
        return None
    # (3,H,W) + (3,H,W) -> (6,H,W)
    merged = np.concatenate((left_img, right_img), axis=0)  # numpy array
    return merged

######################################
# 5) 处理整个数据集, 保存 .npy 并记录 8维标签
######################################
def process_dataset_8class(input_dir, output_dir, excel_path):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读出 (left_img, right_img, label_8)
    data_list = load_8class_labels(excel_path)

    # 准备一个 dict: { npy_filename: [8dim_label] }
    img2label8 = {}

    for left_img, right_img, label_8 in data_list:
        # example: left_img="0_left.jpg", right_img="0_right.jpg"
        # npy_filename="0.npy"
        if "_left.jpg" in left_img:
            npy_name = left_img.replace("_left.jpg", ".npy")
        else:
            # 如果命名不一致, 需自行处理
            npy_name = left_img + ".npy"

        left_path = input_dir / left_img
        right_path = input_dir / right_img

        # 处理左右眼, 拼为6通道
        merged = preprocess_stereo(left_path, right_path)
        if merged is None:
            continue  # 跳过读不到的图像

        # 保存
        np.save(output_dir / npy_name, merged)

        # 记录标签
        # label_8 可能是[0,1,0,0,1,0,0,0]之类
        # 转成int或float都可, 训练时注意 dtype
        img2label8[npy_name] = label_8

    return img2label8

if __name__ == "__main__":
    # 测试示例
    excel_path = "dataset/Traning_Dataset.xlsx"
    input_dir = "dataset/Training_Dataset"
    output_dir = "dataset/output/_8class"

    # 执行预处理
    label_dict = process_dataset_8class(input_dir, output_dir, excel_path)
    print(f"已完成图像预处理, 生成 .npy 文件. 共处理 {len(label_dict)} 条.")
    # 如有需要, 可以把 label_dict 再保存成 .json / .pkl 方便后续使用.
