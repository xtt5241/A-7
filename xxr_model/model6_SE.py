import torch
import torchvision
from torch import nn

# 注意力模块
class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.dconv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pconv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.bn_d = nn.BatchNorm2d(in_channels)
        self.bn_p = nn.BatchNorm2d(in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_d = self.bn_d(self.dconv(x))
        x_p = self.bn_p(self.pconv(x))
        weight = self.sigmoid(torch.mean(x_d, dim=(2,3)) + torch.mean(x_p, dim=(2,3)))
        return x * weight.unsqueeze(-1).unsqueeze(-1)

# SENet模块（按照论文实现）
class SENetBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SENetBlock, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),  # 降维
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),   # 升维
            nn.Sigmoid()
        )

    def forward(self, x):
        # Squeeze: 全局平均池化
        y = self.gap(x).view(x.size(0), -1)  # [batch_size, in_channels]
        # Excitation: 全连接层生成通道权重
        y = self.fc(y).view(x.size(0), -1, 1, 1)  # [batch_size, in_channels, 1, 1]
        # Scale: 对输入特征进行加权
        return x * y.expand_as(x)

class NCNV0(nn.Module):
    def __init__(self, backbone, classifier_out_shape, attention_channels):
        super(NCNV0, self).__init__()
        self.backbone = backbone
        self.attention_block = AttentionBlock(attention_channels)
        self.senet_block = SENetBlock(attention_channels)  # 添加SENetBlock
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化
        self.classifier = nn.Linear(attention_channels * 2, classifier_out_shape)  # 输入维度应为 2048 * 2=4096

    def forward(self, x_left, x_right):
        # 提取特征图 (形状: [batch_size, 2048, 7, 7])
        x_left = self.backbone(x_left)
        x_right = self.backbone(x_right)

        # 应用注意力模块
        x_left = self.attention_block(x_left)
        x_right = self.attention_block(x_right)

        # 应用SENet模块（分别对左右特征图进行通道注意力加权）
        x_left = self.senet_block(x_left)
        x_right = self.senet_block(x_right)

        # 全局平均池化 (形状: [batch_size, 2048, 1, 1])
        x_left = self.global_avg_pool(x_left)
        x_right = self.global_avg_pool(x_right)

        # 展平为 1D 向量 (形状: [batch_size, 2048])
        x_left = x_left.view(x_left.size(0), -1)
        x_right = x_right.view(x_right.size(0), -1)

        # 拼接并分类 (拼接后维度: [batch_size, 2048 * 2=4096])
        x_cat = torch.cat((x_left, x_right), dim=1)
        x = self.classifier(x_cat)
        return x

def create_resnet50_dual(version=0, fine_tune=False):
    # 导入 ResNet50 并移除最后两层（全局池化和全连接）
    weights = torchvision.models.ResNet50_Weights.DEFAULT
    backbone = torchvision.models.resnet50(weights=weights)
    backbone = torch.nn.Sequential(*list(backbone.children())[:-2])

    # 冻结参数
    if not fine_tune:
        for param in backbone.parameters():
            param.requires_grad = False

    # 创建模型
    if version == 0:
        model = NCNV0(
            backbone=backbone,
            classifier_out_shape=8,
            attention_channels=2048  # ResNet50 最后一层卷积的输出通道数
        )
        model.name = 'resnet50_dual'

    print(f"[INFO] Created new {model.name} model.")
    return model

# 测试代码
if __name__ == "__main__":
    # 创建模型
    model = create_resnet50_dual(version=0, fine_tune=False)

    # 生成随机输入数据（模拟 4 张 224x224 的 RGB 图像）
    batch_size = 4
    x_left = torch.randn(batch_size, 3, 512, 512)  # 左图
    x_right = torch.randn(batch_size, 3, 512, 512)  # 右图

    # 前向传播
    output = model(x_left, x_right)

    # 打印输出形状
    print(f"Output shape: {output.shape}")  # 预期输出: [4, 8]