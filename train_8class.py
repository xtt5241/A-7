# train_8class.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 根据您的项目实际路径导入
from data_pre_8class import process_dataset_8class
from dataset_8class import EyeDataset8Class
from model_8class import build_8class_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_8class_main():
    ######################################
    # 1) 数据预处理 (若之前已做,可跳过)
    ######################################
    excel_path = "dataset/Traning_Dataset.xlsx"
    input_dir  = "dataset/Training_Dataset"
    output_dir = "dataset/output/_8class"

    # 处理Excel,合并左右眼,得到 .npy + 8维标签
    print(f"数据准备中")
    label_dict = process_dataset_8class(input_dir, output_dir, excel_path)
    print(f"数据准备完成, 共处理 {len(label_dict)} 个样本.")

    ######################################
    # 2) 构建Dataset, DataLoader
    ######################################
    dataset = EyeDataset8Class(output_dir, label_dict)
    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=8, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(Subset(dataset, val_idx), batch_size=8, shuffle=False, num_workers=0)

    ######################################
    # 3) 构建模型
    ######################################
    model = build_8class_model()  # EfficientNet-B3, 6通道输入, 输出8维
    model.to(device)

    ######################################
    # 4) 定义损失 & 优化器
    ######################################
    criterion = nn.BCEWithLogitsLoss()  # 多标签
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    ######################################
    # 5) 训练循环
    ######################################
    epochs = 10
    train_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        print(f"\nEpoch {epoch+1}/{epochs}")

        for batch_idx, (images, labels8, fnames) in enumerate(train_loader):
            images = images.to(device)         # (B,6,H,W)
            labels8 = labels8.to(device)       # (B,8)

            optimizer.zero_grad()
            outputs = model(images)            # (B,8)
            loss = criterion(outputs, labels8)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx} | Loss={loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

        # 调整学习率
        scheduler.step(avg_loss)

        # 验证 + 保存预测结果
        evaluate_and_save_8class(model, val_loader, epoch+1)

    # Plot 训练Loss
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve (8 classes)")
    plt.legend()
    plt.show()

########################################
# 验证并将结果保存到CSV
########################################
def evaluate_and_save_8class(model, val_loader, epoch_id):
    model.eval()

    all_labels = []
    all_preds  = []
    sample_results = []  # 用于保存到CSV

    with torch.no_grad():
        for images, labels8, fnames in val_loader:
            images = images.to(device)
            outputs = model(images)             # (B,8)
            probs = torch.sigmoid(outputs).cpu().numpy()  # (B,8) 概率

            all_labels.append(labels8.numpy())  # (B,8)
            all_preds.append(probs)             # (B,8)

            # 逐样本处理
            for i in range(len(fnames)):
                filename = fnames[i]
                true_vec = labels8[i].numpy()
                pred_vec = probs[i]

                # 转int 0/1
                bin_pred_vec = (pred_vec > 0.5).astype(int)

                sample_results.append((
                    filename,
                    true_vec,      # e.g. [0,1,0,0,0,1,0,0]
                    pred_vec,      # e.g. [0.02, 0.93, 0.1, ...]
                    bin_pred_vec   # e.g. [0,1,0,0,0,1,0,0]
                ))

    all_labels = np.vstack(all_labels)  # shape(N,8)
    all_preds  = np.vstack(all_preds)   # shape(N,8)

    bin_preds = (all_preds > 0.5).astype(int)

    acc  = accuracy_score(all_labels, bin_preds)
    prec = precision_score(all_labels, bin_preds, average='samples', zero_division=0)
    rec  = recall_score(all_labels, bin_preds, average='samples', zero_division=0)
    f1   = f1_score(all_labels, bin_preds, average='samples', zero_division=0)

    print(f"[Epoch {epoch_id} - Val] Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")

    # ========== 保存到 CSV ==========
    csv_filename = f"val_predictions_epoch{epoch_id}.csv"
    import csv
    with open(csv_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # 表头
        writer.writerow(["filename", "true_label(8dim)", "pred_prob(8dim)", "pred_bin(8dim)"])

        # 每一行
        for (fname, tvec, pvec, binpvec) in sample_results:
            # 为了美观, 可将数组转成字符串
            tvec_str    = ",".join(str(x) for x in tvec)
            pvec_str    = ",".join(f"{x:.3f}" for x in pvec)
            binpvec_str = ",".join(str(x) for x in binpvec)

            writer.writerow([fname, tvec_str, pvec_str, binpvec_str])

    print(f"✅ 已将验证集预测结果保存到: {csv_filename}")


if __name__ == "__main__":
    train_8class_main()
