# data_pre.py

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

######################################
# 1) 加载 Excel，获得左右眼图像文件名
######################################
def load_labels(excel_path):
    """
    读取 Excel, 返回 { left_img_name: keywords, right_img_name: keywords }
    用于确定要处理哪些图像, 本处仅用于遍历文件名.
    """
    df = pd.read_excel(excel_path)
    labels = {}
    for _, row in df.iterrows():
        left_img = str(row["Left-Fundus"])
        right_img = str(row["Right-Fundus"])
        # 这里只是存储下，为后续判断文件名是否存在
        labels[left_img] = row["Left-Diagnostic Keywords"]
        labels[right_img] = row["Right-Diagnostic Keywords"]
    return labels

######################################
# 2) 预处理 Pipeline (可优化)
######################################
# 这里采用了更丰富的增广操作, 您可以根据效果进行微调或删减:
transform = A.Compose([
    A.RandomResizedCrop(height=456, width=456, scale=(0.8, 1.0), ratio=(1.0, 1.0), p=1.0),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

def preprocess_image(image_path, save_path=None):
    """
    读取一张图像, 做增广, 返回tensor(C,H,W).
    若 save_path 不为空, 也可将增广后的图像以jpg形式保存(仅调试或可视化).
    """
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[警告] 无法读取图像: {image_path}")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transformed = transform(image=image)  # albumentations pipeline
    image_aug = transformed["image"]      # tensor shape: (C,H,W)

    # 如果要另存可视化
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # 反归一化后再保存
        vis_img = image_aug.permute(1, 2, 0).cpu().numpy()
        vis_img = (vis_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
        vis_img = np.clip(vis_img, 0, 255).astype(np.uint8)
        cv2.imwrite(str(save_path), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))

    return image_aug

def preprocess_stereo_images(left_path, right_path, save_enhanced=False, output_dir=None):
    """
    同时处理左右眼, 最终拼接成 6 通道并返回 numpy array (6,H,W).
    """
    left_stem = Path(left_path).stem
    right_stem = Path(right_path).stem

    left_img = preprocess_image(left_path,
                                save_path=(output_dir / f"{left_stem}_enhanced.jpg") if save_enhanced else None)
    right_img = preprocess_image(right_path,
                                 save_path=(output_dir / f"{right_stem}_enhanced.jpg") if save_enhanced else None)

    if left_img is None or right_img is None:
        return None

    # 拼接通道: (3 + 3) = 6
    merged_image = np.concatenate((left_img, right_img), axis=0)  # shape: (6,H,W)
    return merged_image

######################################
# 3) 处理整个数据集
######################################
def process_dataset(input_dir, output_dir, excel_path, save_enhanced=False):
    """
    读取excel, 找到 left.jpg, right.jpg, 做增广并拼接, 存成 .npy
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    enhanced_dir = output_dir / "enhanced_images" if save_enhanced else None
    output_dir.mkdir(parents=True, exist_ok=True)

    label_dict = load_labels(excel_path)

    # 遍历Excel中的左眼文件名, 对应右眼
    for left_image_name in label_dict.keys():
        if "_left.jpg" in left_image_name:
            left_image_path = input_dir / left_image_name
            # right图像命名假设一致:  "xxx_left.jpg" -> "xxx_right.jpg"
            right_image_path = input_dir / left_image_name.replace("_left.jpg", "_right.jpg")

            if left_image_path.exists() and right_image_path.exists():
                merged = preprocess_stereo_images(left_image_path, right_image_path,
                                                  save_enhanced=save_enhanced,
                                                  output_dir=enhanced_dir)
                if merged is not None:
                    # 保存为 "xxx.npy"
                    out_name = left_image_name.replace("_left.jpg", ".npy")
                    np.save(output_dir / out_name, merged)

# 如果只想在脚本被直接运行时执行
if __name__ == "__main__":
    excel_path = "dataset/Traning_Dataset.xlsx"
    input_dir = "dataset/Training_Dataset"
    output_dir = "dataset/output"

    # 设置 save_enhanced=True 时，会将增广后的图像以jpg形式另存以便可视化调试
    process_dataset(input_dir, output_dir, excel_path, save_enhanced=True)
