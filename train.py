import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from data_mul import train_loader, val_loader
from model import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# 🚀 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 设备: {device}")

# 🚀 计算 `pos_weight`
def compute_pos_weight(train_loader):
    total_samples = 0
    positive_counts = torch.zeros(98).to(device)

    for _, labels in train_loader:
        labels = labels.to(device).sum(dim=0)  # 统计每个类别的正样本数
        positive_counts += labels
        total_samples += labels.shape[0]

    # 防止 `pos_weight` 过大
    pos_weight = (total_samples - positive_counts) / (positive_counts + 1e-5)
    pos_weight = torch.clamp(pos_weight, max=10.0)  # 🚀 限制最大值
    return pos_weight.to(device)

# 计算 `pos_weight`
pos_weight = compute_pos_weight(train_loader)

# 🚀 `Xavier` 初始化，稳定训练
def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

# 🚀 训练模型
def train_model(model, train_loader, val_loader, epochs=50):
    optimizer = optim.Adam(model.parameters(), lr=1e-5)  # 🚀 降低 lr
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # 🚀 使用 `pos_weight`

    train_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        print(f"🚀 开始 Epoch {epoch+1}")

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device).float()  # 🚀 确保 `labels` 是 float 类型
            optimizer.zero_grad()
            outputs = model(images)

            # 🚨 **检查 `labels` 是否异常**
            if labels.min() < 0 or labels.max() > 1:
                print(f"❌ 发现异常 `labels` 值: min={labels.min()}, max={labels.max()}")
                exit()

            loss = criterion(outputs, labels)

            # 🚨 **检查 `loss` 是否 NaN**
            if torch.isnan(loss):
                print("❌ 发现 NaN 损失！")
                print("outputs:", outputs)
                print("labels:", labels)
                exit()

            loss.backward()

            # 🚀 **梯度裁剪**
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"✅ Batch {batch_idx+1} Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"✅ Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

        scheduler.step(avg_loss)  # 🚀 `ReduceLROnPlateau`
        evaluate_model(model, val_loader)

    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.show()

# 🚀 评估模型
def evaluate_model(model, val_loader):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = torch.sigmoid(outputs).cpu().numpy()

            all_labels.append(labels.cpu().numpy())
            all_predictions.append(probabilities)

    all_labels = np.vstack(all_labels)
    all_predictions = np.vstack(all_predictions)

    # 🚀 **防止 `F1-score` 计算错误**
    if all_predictions.shape != all_labels.shape:
        print("❌ `evaluate_model()` 维度错误！")
        print(f"all_labels.shape={all_labels.shape}, all_predictions.shape={all_predictions.shape}")
        return

    accuracy = accuracy_score(all_labels, all_predictions > 0.5)
    precision = precision_score(all_labels, all_predictions > 0.5, average='samples', zero_division=0)
    recall = recall_score(all_labels, all_predictions > 0.5, average='samples', zero_division=0)
    f1 = f1_score(all_labels, all_predictions > 0.5, average='samples', zero_division=0)

    print(f"✅ 验证集结果: 准确率: {accuracy:.4f}, 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1分数: {f1:.4f}")



if __name__ == '__main__':
    model, _ = load_model("dataset/English-Chinese_Disease_Mapping.csv")
    model = model.to(device)
    model.apply(init_weights)  # 🚀 使用 `Xavier` 初始化
    train_model(model, train_loader, val_loader)
