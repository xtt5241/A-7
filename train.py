import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from dataset import train_loader, val_loader

# 🚀 1. 设备选择（确保 GPU 运行）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 使用设备: {device}")

# 🚀 2. 禁用 cuDNN 以避免 `CUDNN_STATUS_BAD_PARAM_STREAM_MISMATCH`
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# 🚀 3. 加载 EfficientNet-B3（减少显存）
model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)

# 🚀 4. 修改输入层（支持 6 通道）
new_conv = nn.Conv2d(6, 40, kernel_size=3, stride=2, padding=1, bias=False)
new_conv.weight.data[:, :3, :, :] = model.features[0][0].weight.data  # 复制原始权重
model.features[0][0] = new_conv  # 替换第一层卷积

# 🚀 5. 修改分类层（8 类）
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 8)
model = model.to(device)

# 🚀 6. 定义损失函数 & 优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 🚀 7. 启用混合精度（减少显存占用）
scaler = torch.amp.GradScaler()

def train_model(model, train_loader, val_loader, epochs=50):
    print("🚀 开始训练模型...")
    model.train()

    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # 🚀 使用新的 `autocast("cuda")`（修复 FutureWarning）
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            # 🚀 混合精度优化
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        train_acc = 100 * correct / total
        print(f"✅ Epoch {epoch+1} | Loss: {total_loss:.4f} | Accuracy: {train_acc:.2f}%")

        torch.cuda.empty_cache()  # 🚀 清理 CUDA 缓存
        evaluate_model(model, val_loader)

def evaluate_model(model, val_loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"🟠 验证集准确率: {accuracy:.2f}%")
    model.train()

# 🚀 8. 训练
train_model(model, train_loader, val_loader, epochs=50)

# 🚀 9. 保存模型
torch.save(model.state_dict(), "efficientnet_6ch_50epoch.pth")
print("✅ 训练完成，模型已保存！")
