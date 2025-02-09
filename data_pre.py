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
def preprocess_image(image_path, save_path=None):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"警告：无法读取 {image_path}")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transformed = transform(image=image)
    image = transformed["image"]

    # 如果提供了保存路径，保存增强后的图像
    if save_path:
        image_np = image.permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

    return image

# 4️⃣ 预处理左右眼数据
def preprocess_stereo_images(left_image_path, right_image_path, save_enhanced=False, output_dir=None):
    left_image_name = Path(left_image_path).stem
    right_image_name = Path(right_image_path).stem

    left_image = preprocess_image(left_image_path, save_path=(output_dir / f"{left_image_name}_enhanced.jpg") if save_enhanced else None)
    right_image = preprocess_image(right_image_path, save_path=(output_dir / f"{right_image_name}_enhanced.jpg") if save_enhanced else None)

    if left_image is None or right_image is None:
        return None

    merged_image = np.concatenate((left_image, right_image), axis=0)  # 6 通道
    return merged_image

# 5️⃣ 处理整个数据集
def process_dataset(input_dir, output_dir, excel_path, save_enhanced=False):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    enhanced_dir = output_dir / "enhanced_images" if save_enhanced else None
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(excel_path)

    for left_image_name in labels.keys():
        if "left" in left_image_name:
            left_image_path = input_dir / left_image_name
            right_image_path = input_dir / left_image_name.replace("_left.jpg", "_right.jpg")

            if left_image_path.exists() and right_image_path.exists():
                processed_image = preprocess_stereo_images(left_image_path, right_image_path, save_enhanced, enhanced_dir)
                if processed_image is not None:
                    np.save(output_dir / left_image_name.replace("_left.jpg", ".npy"), processed_image)

# 运行预处理
excel_path = "dataset/Traning_Dataset.xlsx"
input_dir = "dataset/Training_Dataset"
output_dir = "dataset/output"

# 如果需要保存增强后的图像，设置 save_enhanced=True
process_dataset(input_dir, output_dir, excel_path, save_enhanced=True)
