import os

def add_preprocess_suffix_to_images(folder_path):
    """
    将指定文件夹下所有图片文件重命名，文件名结尾加上 "_preprocess"。
    例如 "1_left.jpg" 变为 "1_left_preprocess.jpg"。
    
    :param folder_path: 文件夹的路径（字符串）
    """
    
    # 列出文件夹中的所有文件
    file_list = os.listdir(folder_path)

    for old_name in file_list:
        # 获取文件的绝对路径
        old_path = os.path.join(folder_path, old_name)

        # 如果是一个子文件夹，则跳过（只处理文件）
        if os.path.isdir(old_path):
            continue

        # 拆分文件名和扩展名
        filename, extension = os.path.splitext(old_name)
        
        # 仅当扩展名属于常见图片类型时（可自行按需调整）
        # 你可以加更多，比如 ".jpeg", ".png", ".bmp" 等等
        valid_exts = [".jpg", ".jpeg", ".png", ".bmp"]
        if extension.lower() not in valid_exts:
            continue
        
        # 如果已经包含 "_preprocess"，为了避免重复或冲突，可以选择跳过或按需处理
        if "_preprocess" in filename:
            # 如果你希望跳过已经带后缀的文件，就执行 continue
            # 如果你愿意再次加后缀也可以，只是不太常见
            continue
        
        # 构造新的文件名
        new_name = filename + "_preprocess" + extension
        new_path = os.path.join(folder_path, new_name)

        # 重命名
        os.rename(old_path, new_path)
        print(f"已将 {old_name} 重命名为 {new_name}")

# ========== 使用示例 ==========
if __name__ == "__main__":
    # 请把此处改成你想要处理的文件夹路径
    folder = r"dataset\xxr\preprocess_images"
    add_preprocess_suffix_to_images(folder)
