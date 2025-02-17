import torch
import torch.nn as nn
import torchvision.models as models

def build_16class_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 以 EfficientNet-B3 为例
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)

    # 修改第一层: 3->6通道
    old_conv = model.features[0][0]
    in_channels = 6
    out_channels = old_conv.out_channels

    new_conv = nn.Conv2d(in_channels, out_channels,
                         kernel_size=old_conv.kernel_size,
                         stride=old_conv.stride,
                         padding=old_conv.padding,
                         bias=(old_conv.bias is not None))
    with torch.no_grad():
        # 拷贝前3通道
        new_conv.weight[:, :3, :, :] = old_conv.weight
        # 后3通道随机初始化
        nn.init.xavier_uniform_(new_conv.weight[:, 3:, :, :])
        if new_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    model.features[0][0] = new_conv

    # 去掉原有分类层, 自己加一个Linear输出16维, 无Sigmoid
    in_feats = model.classifier[1].in_features
    model.classifier = nn.Linear(in_feats, 16)

    return model.to(device)
