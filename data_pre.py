import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 1️⃣ 读取 Excel 数据
def load_labels(excel_path):
    df = pd.read_excel(excel_path)
    labels = {}
    for _, row in df.iterrows():
        left_img = row["Left-Fundus"]
        right_img = row["Right-Fundus"]
        left_label = row["Left-Diagnostic Keywords"]
        right_label = row["Right-Diagnostic Keywords"]

        labels[left_img] = left_label
        labels[right_img] = right_label
    return labels

# 2️⃣ 预处理 Pipeline
transform = A.Compose([
    A.Resize(456, 456),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# 3️⃣ 预处理单张图像
def preprocess_image(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"警告：无法读取 {image_path}")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image=image)["image"]
    return image

# 4️⃣ 预处理左右眼数据
def preprocess_stereo_images(left_image_path, right_image_path):
    left_image = preprocess_image(left_image_path)
    right_image = preprocess_image(right_image_path)

    if left_image is None or right_image is None:
        return None

    merged_image = np.concatenate((left_image, right_image), axis=0)  # 6 通道
    return merged_image

# 5️⃣ 处理整个数据集
def process_dataset(input_dir, output_dir, excel_path):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(excel_path)

    for left_image_name in labels.keys():
        if "left" in left_image_name:
            left_image_path = input_dir / left_image_name
            right_image_path = input_dir / left_image_name.replace("_left.jpg", "_right.jpg")

            if left_image_path.exists() and right_image_path.exists():
                processed_image = preprocess_stereo_images(left_image_path, right_image_path)
                if processed_image is not None:
                    np.save(output_dir / left_image_name.replace("_left.jpg", ".npy"), processed_image)

# 运行预处理
excel_path = "XTT\dataset\Traning_Dataset.xlsx"
input_dir = "XTT\dataset/Training_Dataset"
output_dir = "XTT\dataset\output"
process_dataset(input_dir, output_dir, excel_path)
