# metrics.py
"""
示例 metrics.py 文件，定义 compute_challenge_metrics 函数，实现对模型输出的多标签预测结果
进行常见指标计算（kappa、f1、auc 等），以及一个自定义综合分数 final_score。
用法：
    from metrics import compute_challenge_metrics
    kappa, f1, auc, final_score = compute_challenge_metrics(all_y, all_probs)
"""

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score

def compute_challenge_metrics(all_y, all_probs, threshold=0.5):
    """
    计算挑战赛常见指标 Kappa、F1、AUC，以及一个自定义 final_score。
    
    参数:
    -------
    all_y : numpy.ndarray
        真实标签, 形状 (N, C)，其中 N 为样本数, C 为标签数(例如 8).
        通常在 engine.py 里是通过 `all_y = all_y.detach().cpu().numpy()` 得到的.
    all_probs : numpy.ndarray
        模型输出的预测概率, 形状 (N, C).
    threshold : float, 默认 0.5
        将概率转换为二分类标签的阈值.
    
    返回值:
    -------
    kappa : float
    f1 : float
    auc : float
    final_score : float
        自定义的综合分数 (此处仅作演示，将上述三个指标平均).
    """
    # 将预测概率转为 0/1 的预测结果
    preds = (all_probs >= threshold).astype(int)

    # 如果是多标签场景，需要分别对每个标签进行计算，然后再取平均
    # 注意：对于某些标签，正负样本可能全部为同一类，导致 roc_auc_score 无法计算
    # 这里简单处理: 如果该标签只有单一类，就记该标签的 auc=0.0 或其它默认值
    kappa_list = []
    f1_list = []
    auc_list = []

    num_labels = all_y.shape[1]
    for i in range(num_labels):
        # 计算 Cohen's Kappa
        kappa_i = cohen_kappa_score(all_y[:, i], preds[:, i])
        kappa_list.append(kappa_i)

        # 计算 F1
        f1_i = f1_score(all_y[:, i], preds[:, i])
        f1_list.append(f1_i)

        # 计算 AUC (需保证该标签正负样本都存在)
        if len(np.unique(all_y[:, i])) == 2:
            auc_i = roc_auc_score(all_y[:, i], all_probs[:, i])
        else:
            # 若此标签只有 0 或只有 1，就无法计算真正的 auc，这里简单返回 0
            # 也可选择跳过不纳入平均
            auc_i = 0.0
        auc_list.append(auc_i)

    # 取平均，得到整体指标
    kappa = np.mean(kappa_list)
    f1 = np.mean(f1_list)
    auc = np.mean(auc_list)

    # 这里给出一个简单示例，把 kappa、f1、auc 三者的平均当作最终分数
    # 你可以根据需要做更复杂的加权
    final_score = (kappa + f1 + auc) / 3.0

    return kappa, f1, auc, final_score
