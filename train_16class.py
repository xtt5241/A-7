import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from dataset_16class import EyeDataset16Class
from model_16class import build_16class_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_16class_main():
    ######################################
    # 1) 读 label_dict_16 + 构建Dataset
    ######################################
    label_dict_path = "dataset/output_16/label_dict_16.npy"
    npy_dir         = "dataset/output_16"

    label_dict = np.load(label_dict_path, allow_pickle=True).item()
    dataset = EyeDataset16Class(npy_dir, label_dict)

    # 划分 train / val
    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=8, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(Subset(dataset, val_idx), batch_size=8, shuffle=False, num_workers=0)

    ######################################
    # 2) 构建模型(16维输出, 无Sigmoid)
    ######################################
    model = build_16class_model()
    model = model.to(device)

    ######################################
    # 3) 定义损失 & 优化器
    ######################################
    # 这里用 BCEWithLogitsLoss, 要注意模型输出是 raw logits
    criterion = nn.BCEWithLogitsLoss()  # 16维多标签
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    ######################################
    # 4) 训练
    ######################################
    epochs = 10
    train_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        print(f"\nEpoch {epoch+1}/{epochs}")

        for batch_idx, (images, labels16, fnames) in enumerate(train_loader):
            images = images.to(device)
            labels16 = labels16.to(device)

            optimizer.zero_grad()
            outputs = model(images)  # raw logits, shape(B,16)
            loss = criterion(outputs, labels16)
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

        # 验证 + 保存
        evaluate_and_save_16class(model, val_loader, epoch+1)

    # 训练曲线
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("16-class (Left+Right 8) multi-label training")
    plt.legend()
    plt.show()

def evaluate_and_save_16class(model, val_loader, epoch_id):
    model.eval()
    all_labels = []
    all_preds  = []
    sample_results = []

    with torch.no_grad():
        for images, labels16, fnames in val_loader:
            images = images.to(device)
            # outputs是logits => 计算prob需要显式sigmoid
            outputs = model(images)  # (B,16)
            probs = torch.sigmoid(outputs).cpu().numpy()  # (B,16)

            all_labels.append(labels16.numpy())
            all_preds.append(probs)

            for i in range(len(fnames)):
                fname = fnames[i]
                tvec = labels16[i].numpy()
                pvec = probs[i]
                # 阈值0.5, 也可改成0.51或其它
                binpvec = (pvec > 0.6).astype(int)

                sample_results.append((fname, tvec, pvec, binpvec))

    all_labels = np.vstack(all_labels)  # shape(N,16)
    all_preds  = np.vstack(all_preds)   # shape(N,16)
    bin_preds  = (all_preds > 0.6).astype(int)

    acc  = accuracy_score(all_labels, bin_preds)
    prec = precision_score(all_labels, bin_preds, average='samples', zero_division=0)
    rec  = recall_score(all_labels, bin_preds, average='samples', zero_division=0)
    f1   = f1_score(all_labels, bin_preds, average='samples', zero_division=0)

    print(f"[Epoch {epoch_id}] Val => Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")

    # 保存到 CSV (可选)
    csv_filename = f"val_predictions_epoch{epoch_id}.csv"
    with open(csv_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "true_label(16dim)", "pred_prob(16dim)", "pred_bin(16dim)"])
        for (fname, tvec, pvec, bvec) in sample_results:
            tvec_str = ",".join(str(x) for x in tvec)
            pvec_str = ",".join(f"{x:.3f}" for x in pvec)
            bvec_str = ",".join(str(x) for x in bvec)
            writer.writerow([fname, tvec_str, pvec_str, bvec_str])

    print(f"✅ 验证集预测已保存到: {csv_filename}")

if __name__=="__main__":
    train_16class_main()
