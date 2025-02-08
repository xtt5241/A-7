import torch
import torch.nn as nn
import torchvision.models as models

CLASSES = ["normal fundus", "diabetic retinopathy", "glaucoma", "cataract",
           "age-related macular degeneration", "hypertensive retinopathy", "myopia", "other"]

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.efficientnet_b3(weights=None)  # 不加载预训练权重
    new_conv = nn.Conv2d(6, 40, kernel_size=3, stride=2, padding=1, bias=False)
    model.features[0][0] = new_conv  # 替换输入层
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))  # 8 个类别

    # 🚀 修改输出激活函数，让输出变为 0-1 概率
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[1].in_features, len(CLASSES)),
        nn.Sigmoid()  # 🚀 让每个类别独立预测
    )

    try:
        model.load_state_dict(torch.load("G:\Python\eye_disease\XTT\efficientnet_6ch.pth", map_location=device), strict=False)
        model = model.to(device)
        model.eval()  # 进入推理模式
        print("✅ 模型加载成功！")
    except FileNotFoundError:
        print("❌ 错误：未找到 `efficientnet_6ch.pth`，请先训练模型！")
        exit(1)

    return model
