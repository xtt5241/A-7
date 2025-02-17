import torch
import torch.nn as nn
import torchvision.models as models
import pandas as pd

# 1️⃣ 读取 99 关键词
def load_keywords(mapping_path):
    mapping_df = pd.read_csv(mapping_path, encoding='gbk')

    # 🚀 **确保跳过第一行标题**
    if "English Keyword" in mapping_df.iloc[0].values:
        mapping_df = mapping_df.iloc[1:].reset_index(drop=True)  # **跳过标题行**

    keywords = mapping_df["English Keyword"].str.lower().str.strip().tolist()
    print(f"✅ `model.py` 关键词数: {len(keywords)}")  # 🚀 打印类别数，确保和 `data_mul.py` 一致
    return keywords

def load_model(mapping_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    keywords = load_keywords(mapping_path)
    num_keywords = len(keywords)  # 🚀 **98 个类别**

    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    model.features[0][0] = nn.Conv2d(6, 40, kernel_size=3, stride=2, padding=1, bias=False)

    # 🚀 确保模型输出和 `data_mul.py` 一致
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[1].in_features, num_keywords),
        nn.Sigmoid()
    )

    print(f"✅ `model.py` 输出关键词类别数: {num_keywords}")  # 🚀 打印类别数，确保和 `data_mul.py` 一致
    return model, keywords

