# model_8class.py

import torch
import torch.nn as nn
import torchvision.models as models

def build_8class_model():
    """
    EfficientNet-B3, 输入6通道, 输出8维. Sigmoid for multi-label.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)

    # 修改第一层输入
    old_conv = model.features[0][0]
    in_channels = 6
    out_channels = old_conv.out_channels
    kernel_size = old_conv.kernel_size
    stride = old_conv.stride
    padding = old_conv.padding
    bias = (old_conv.bias is not None)

    new_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
    with torch.no_grad():
        # 拷贝前3通道
        new_conv.weight[:, :3, :, :] = old_conv.weight
        # 后3通道随机初始化
        nn.init.xavier_uniform_(new_conv.weight[:, 3:, :, :])
        if bias:
            new_conv.bias.copy_(old_conv.bias)

    model.features[0][0] = new_conv

    # 去掉最后分类层
    in_feats = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_feats, 8),  # 8 classes
        nn.Sigmoid()
    )

    return model.to(device)
