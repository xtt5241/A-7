# model.py
import torch
import torch.nn as nn
import torchvision.models as models

def load_model(num_keywords=98, num_categories=8):
    """
    构建双输出头的EfficientNet-B3:
      - head_keywords: 输出 num_keywords 维 (Sigmoid)
      - head_categories: 输出 num_categories 维 (Sigmoid)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 加载预训练的 EfficientNet-B3
    backbone = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)

    # 2) 修改输入通道 3->6
    old_conv = backbone.features[0][0]
    in_channels = 6
    out_channels = old_conv.out_channels
    kernel_size = old_conv.kernel_size
    stride = old_conv.stride
    padding = old_conv.padding
    bias = (old_conv.bias is not None)

    new_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
    with torch.no_grad():
        # 复制前3通道权重
        new_conv.weight[:, :3, :, :] = old_conv.weight
        # 后3通道随机初始化
        nn.init.xavier_uniform_(new_conv.weight[:, 3:, :, :])
        if bias:
            new_conv.bias.copy_(old_conv.bias)

    backbone.features[0][0] = new_conv

    # 3) 去掉最后分类层
    in_feats = backbone.classifier[1].in_features
    backbone.classifier = nn.Identity()

    # 4) 定义双头
    head_keywords = nn.Sequential(
        nn.Linear(in_feats, num_keywords),
        nn.Sigmoid()
    )
    head_categories = nn.Sequential(
        nn.Linear(in_feats, num_categories),
        nn.Sigmoid()
    )

    # 封装成一个自定义模块
    model = MultiTaskModel(backbone, head_keywords, head_categories)
    return model.to(device)

class MultiTaskModel(nn.Module):
    def __init__(self, backbone, head_keywords, head_categories):
        super().__init__()
        self.backbone = backbone
        self.head_keywords = head_keywords
        self.head_categories = head_categories

    def forward(self, x):
        feat = self.backbone(x)          # (batch, in_feats)
        pred_kws = self.head_keywords(feat)   # (batch, num_keywords)
        pred_cats = self.head_categories(feat) # (batch, num_categories)
        return pred_kws, pred_cats
