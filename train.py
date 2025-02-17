# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

from data_mul import EyeDatasetMultiTask
from model import load_model
from focal_loss import FocalLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##########################
# 1. 构建数据集
##########################
excel_path = "dataset/Traning_Dataset.xlsx"
mapping_path = "dataset/English-Chinese_Disease_Mapping.csv"
data_dir = "dataset/output"

dataset = EyeDatasetMultiTask(data_dir, excel_path, mapping_path)
train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)

train_loader = DataLoader(Subset(dataset, train_idx), batch_size=8, shuffle=True, num_workers=0)
val_loader = DataLoader(Subset(dataset, val_idx), batch_size=8, shuffle=False, num_workers=0)

num_keywords = len(dataset.keywords_list)      # 98
num_categories = len(dataset.categories_list)  # 例如 8

##########################
# 2. 加载模型
##########################
model = load_model(num_keywords, num_categories)
model = model.to(device)

##########################
# 3. 定义FocalLoss
##########################
# 这里简单用 alpha=1, gamma=2
criterion_kws = FocalLoss(alpha=1.0, gamma=2.0, reduction='mean')
criterion_cats = FocalLoss(alpha=1.0, gamma=2.0, reduction='mean')

##########################
# 4. 优化器 & 调度
##########################
optimizer = optim.Adam(model.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

##########################
# 5. 训练函数
##########################
def train_model(model, train_loader, val_loader, epochs=10):
    train_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        print(f"\nEpoch {epoch+1}/{epochs}")
        for batch_idx, (images, label_kws, label_cats) in enumerate(train_loader):
            images = images.to(device)
            label_kws = label_kws.to(device)
            label_cats = label_cats.to(device)

            optimizer.zero_grad()
            pred_kws, pred_cats = model(images)

            # 分别计算关键词、类别的FocalLoss
            loss_k = criterion_kws(pred_kws, label_kws)
            loss_c = criterion_cats(pred_cats, label_cats)
            loss = loss_k + loss_c

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f" Batch {batch_idx}: loss={loss.item():.4f} (kws={loss_k.item():.4f}, cats={loss_c.item():.4f})")

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

        scheduler.step(avg_loss)
        evaluate(model, val_loader)

    # 训练曲线
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.show()

##########################
# 6. 验证集评估
##########################
def evaluate(model, val_loader):
    model.eval()
    all_kws_labels = []
    all_kws_preds = []

    all_cat_labels = []
    all_cat_preds = []

    with torch.no_grad():
        for images, label_kws, label_cats in val_loader:
            images = images.to(device)
            pred_kws, pred_cats = model(images)

            kws_prob = torch.sigmoid(pred_kws).cpu().numpy()
            cats_prob = torch.sigmoid(pred_cats).cpu().numpy()

            all_kws_labels.append(label_kws.numpy())
            all_kws_preds.append(kws_prob)

            all_cat_labels.append(label_cats.numpy())
            all_cat_preds.append(cats_prob)

    # 拼接
    all_kws_labels = np.vstack(all_kws_labels)
    all_kws_preds = np.vstack(all_kws_preds)
    all_cat_labels = np.vstack(all_cat_labels)
    all_cat_preds = np.vstack(all_cat_preds)

    # 阈值0.5
    bin_kws = (all_kws_preds > 0.5).astype(int)
    bin_cats = (all_cat_preds > 0.5).astype(int)

    # 多标签评估(sklearn的'accuracy_score'会要求完全匹配才算1)
    kws_acc = accuracy_score(all_kws_labels, bin_kws)
    kws_prec = precision_score(all_kws_labels, bin_kws, average='samples', zero_division=0)
    kws_rec = recall_score(all_kws_labels, bin_kws, average='samples', zero_division=0)
    kws_f1 = f1_score(all_kws_labels, bin_kws, average='samples', zero_division=0)

    cats_acc = accuracy_score(all_cat_labels, bin_cats)
    cats_prec = precision_score(all_cat_labels, bin_cats, average='samples', zero_division=0)
    cats_rec = recall_score(all_cat_labels, bin_cats, average='samples', zero_division=0)
    cats_f1 = f1_score(all_cat_labels, bin_cats, average='samples', zero_division=0)

    print(f"[Keywords] Acc={kws_acc:.4f}, Prec={kws_prec:.4f}, Rec={kws_rec:.4f}, F1={kws_f1:.4f}")
    print(f"[Category] Acc={cats_acc:.4f}, Prec={cats_prec:.4f}, Rec={cats_rec:.4f}, F1={cats_f1:.4f}")

##########################
# 7. 主入口
##########################
if __name__ == "__main__":
    train_model(model, train_loader, val_loader, epochs=10)
