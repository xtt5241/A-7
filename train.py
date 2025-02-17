# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt

from data_mul import EyeDatasetMultiTask
from model import load_model
from focal_loss import FocalLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################
# 1. 构建数据集 & train/val split
################################
excel_path = "dataset/Traning_Dataset.xlsx"
mapping_path = "dataset/English-Chinese_Disease_Mapping.csv"
data_dir = "dataset/output"

dataset = EyeDatasetMultiTask(data_dir, excel_path, mapping_path)
train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)

train_loader = DataLoader(Subset(dataset, train_idx), batch_size=8, shuffle=True, num_workers=0)
val_loader   = DataLoader(Subset(dataset, val_idx), batch_size=8, shuffle=False, num_workers=0)

num_keywords   = len(dataset.keywords_list)     # 98
num_categories = len(dataset.categories_list)   # 8~10

################################
# 2. 加载模型
################################
model = load_model(num_keywords, num_categories)
model = model.to(device)

################################
# 3. 定义 FocalLoss & 优化器
################################
criterion_kws  = FocalLoss(alpha=0.75, gamma=1.5, reduction='mean')
criterion_cats = FocalLoss(alpha=0.75, gamma=1.5, reduction='mean')

optimizer = optim.Adam(model.parameters(), lr=3e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

################################
# 4. 训练流程
################################
def train_model(model, train_loader, val_loader, epochs=10):
    train_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        print(f"\nEpoch {epoch+1}/{epochs}")
        for batch_idx, (images, label_kws, label_cats, _) in enumerate(train_loader):
            images = images.to(device)
            label_kws = label_kws.to(device)
            label_cats = label_cats.to(device)

            optimizer.zero_grad()

            pred_kws, pred_cats = model(images)
            loss_k = criterion_kws(pred_kws, label_kws)
            loss_c = criterion_cats(pred_cats, label_cats)
            loss   = loss_k + loss_c

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx} | Loss={loss.item():.4f} (k={loss_k.item():.4f}, c={loss_c.item():.4f})")

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

        # 调整学习率
        scheduler.step(avg_loss)

        # 在验证集上做评估 + 打印并保存预测
        evaluate_and_save(model, val_loader, epoch+1)

    # 画出训练损失曲线
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.show()

################################
# 5. 验证集评估 + 输出每张图的预测 + 保存到CSV
################################
def evaluate_and_save(model, val_loader, epoch_id):
    model.eval()

    all_kws_labels = []
    all_kws_preds  = []
    all_cat_labels = []
    all_cat_preds  = []

    kws_list = val_loader.dataset.dataset.keywords_list
    cat_list = val_loader.dataset.dataset.categories_list

    # 存储逐图像的预测信息
    sample_results = []

    with torch.no_grad():
        for images, label_kws, label_cats, fname in val_loader:
            images = images.to(device)
            pred_kws, pred_cats = model(images)

            kws_prob = torch.sigmoid(pred_kws).cpu().numpy()  # (batch, 98)
            cats_prob = torch.sigmoid(pred_cats).cpu().numpy()# (batch, 8/10)

            true_kws_np = label_kws.numpy()
            true_cats_np = label_cats.numpy()

            # 汇总到计算整体指标
            all_kws_labels.append(true_kws_np)
            all_kws_preds.append(kws_prob)
            all_cat_labels.append(true_cats_np)
            all_cat_preds.append(cats_prob)

            # 转换为可读格式
            for i in range(len(fname)):
                f = fname[i]

                # 真值
                true_kws_idx = np.where(true_kws_np[i] > 0.5)[0]
                true_cats_idx = np.where(true_cats_np[i] > 0.5)[0]

                true_kws_str  = [kws_list[idx] for idx in true_kws_idx]
                true_cats_str = [cat_list[idx] for idx in true_cats_idx]

                # 预测标签(阈值=0.5)
                pred_kws_idx = np.where(kws_prob[i] > 0.5)[0]
                pred_cats_idx = np.where(cats_prob[i] > 0.5)[0]

                pred_kws_str  = [kws_list[idx] for idx in pred_kws_idx]
                pred_cats_str = [cat_list[idx] for idx in pred_cats_idx]

                # 拼到结果list
                sample_results.append((
                    f,
                    ";".join(true_kws_str),
                    ";".join(pred_kws_str),
                    ";".join(true_cats_str),
                    ";".join(pred_cats_str)
                ))

    # 计算整体多标签指标
    all_kws_labels = np.vstack(all_kws_labels)
    all_kws_preds  = np.vstack(all_kws_preds)
    all_cat_labels = np.vstack(all_cat_labels)
    all_cat_preds  = np.vstack(all_cat_preds)

    bin_kws  = (all_kws_preds > 0.5).astype(int)
    bin_cats = (all_cat_preds > 0.5).astype(int)

    kws_acc  = accuracy_score(all_kws_labels, bin_kws)
    kws_prec = precision_score(all_kws_labels, bin_kws, average='samples', zero_division=0)
    kws_rec  = recall_score(all_kws_labels, bin_kws, average='samples', zero_division=0)
    kws_f1   = f1_score(all_kws_labels, bin_kws, average='samples', zero_division=0)

    cats_acc  = accuracy_score(all_cat_labels, bin_cats)
    cats_prec = precision_score(all_cat_labels, bin_cats, average='samples', zero_division=0)
    cats_rec  = recall_score(all_cat_labels, bin_cats, average='samples', zero_division=0)
    cats_f1   = f1_score(all_cat_labels, bin_cats, average='samples', zero_division=0)

    print(f"[Epoch {epoch_id} - Val] Keywords -> Acc={kws_acc:.4f}, Prec={kws_prec:.4f}, Rec={kws_rec:.4f}, F1={kws_f1:.4f}")
    print(f"[Epoch {epoch_id} - Val] Category -> Acc={cats_acc:.4f}, Prec={cats_prec:.4f}, Rec={cats_rec:.4f}, F1={cats_f1:.4f}")

    print("========== 验证集逐样本预测结果 (关键词 + 疾病类别) ==========")
    for (f, t_k, p_k, t_c, p_c) in sample_results:
        print(f"文件: {f}")
        print(f"  True Keywords: {t_k.split(';') if t_k else []}")
        print(f"  Pred Keywords: {p_k.split(';') if p_k else []}")
        print(f"  True Categories: {t_c.split(';') if t_c else []}")
        print(f"  Pred Categories: {p_c.split(';') if p_c else []}")
        print("------------------------------------------------")

    # ★★★ 重点：把结果保存到CSV文件 ★★★
    csv_filename = f"val_predictions_epoch{epoch_id}.csv"
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # 写表头
        writer.writerow(["file_name", "true_keywords", "pred_keywords", "true_categories", "pred_categories"])

        # 写每行
        for (f_name, true_kws_str, pred_kws_str, true_cats_str, pred_cats_str) in sample_results:
            writer.writerow([f_name, true_kws_str, pred_kws_str, true_cats_str, pred_cats_str])

    print(f"✅ 已将验证集预测结果保存至 {csv_filename}")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

###############################
# 6. 训练主入口
###############################
if __name__ == "__main__":
    train_model(model, train_loader, val_loader, epochs=10)
