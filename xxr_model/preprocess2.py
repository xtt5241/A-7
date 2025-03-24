# 预处理
import cv2
import os
import numpy as np

def preprocess_image(image_path, save_dir):
    """
    预处理图像并保存结果
    :param image_path: 输入图像路径
    :param save_dir: 保存图像的目录
    :return: None
    """
    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    # 获取原始文件名
    file_name = os.path.basename(image_path)

    # 圆形裁剪
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 6, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # 找到最大轮廓
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)

        # 检查裁剪区域是否在图像范围内
        if x >= 0 and y >= 0 and (x + w) <= image.shape[1] and (y + h) <= image.shape[0]:
            cropped_image = image[y:y+h, x:x+w]
        else:
            print(f"Crop region out of bounds for image: {image_path}")
            cropped_image = image
    else:
        print(f"No contours found for image: {image_path}")
        cropped_image = image

    # 调整大小到 512x512
    resized_image = cv2.resize(cropped_image, (512, 512))

    # CLAHE对比度增强
    lab = cv2.cvtColor(resized_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    clahe_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # 中值滤波去噪
    final_image = cv2.medianBlur(clahe_image, 3)

    # 保存处理后的图像
    save_path = os.path.join(save_dir, file_name)
    cv2.imwrite(save_path, final_image)
    print(f"Processed and saved image to {save_path}")

def preprocess_images(input_dir, output_dir):
    """
    批量处理图像
    :param input_dir: 输入图像目录
    :param output_dir: 输出图像目录
    :return: None
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历输入目录中的所有图像
    for file_name in os.listdir(input_dir):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, file_name)
            preprocess_image(image_path, output_dir)

if __name__ == "__main__":
    # 输入图像目录
    input_dir = 'dataset\Training_Dataset'

    # 输出图像目录
    output_dir = 'dataset\preprocess_images'

    # 批量处理图像
    preprocess_images(input_dir, output_dir)
    print(f"All images processed and saved to {output_dir}")