#!/usr/bin/env python3
"""训练与评估工具：单 epoch 训练、模型评估、方向感知损失。"""

from __future__ import annotations

import torch
import torch.nn as nn


def direction_loss_fn(y_hat: torch.Tensor, y: torch.Tensor,
                      threshold: float = 0.0) -> torch.Tensor:
    """方向感知损失：当预测值与真实值在决策边界两侧时施加惩罚。

    使用 ReLU 确保正确方向的样本零惩罚（softplus 有 ln2 基线开销）。
    在 log 空间下，边界为 0；在原始空间下，边界为 1.0。
    """
    # 当 y_hat 和 y 符号相同 → 乘积为正 → margin 为负 → ReLU=0
    # 当符号不同 → 乘积为负 → margin 为正 → 产生惩罚
    margin = -(y_hat - threshold) * (y - threshold)
    return torch.relu(margin).mean()


def train_one_epoch(model, loader, criterion, optimizer, device,
                    max_grad_norm: float = 1.0,
                    noise_std: float = 0.0,
                    direction_lambda: float = 0.0,
                    log_target: bool = False):
    model.train()
    total_loss = 0.0
    n = 0
    for x1, x2, y, lv1, lv2 in loader:
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        lv1, lv2 = lv1.to(device), lv2.to(device)

        # 数据增强：对有效区域添加高斯噪声
        if noise_std > 0:
            x1 = x1 + torch.randn_like(x1) * noise_std
            x2 = x2 + torch.randn_like(x2) * noise_std

        optimizer.zero_grad()
        y_hat = model(x1, x2, lv1, lv2)
        loss = criterion(y_hat, y)

        # 方向感知辅助损失
        if direction_lambda > 0:
            threshold = 0.0 if log_target else 1.0
            dir_loss = direction_loss_fn(y_hat, y, threshold=threshold)
            loss = loss + direction_lambda * dir_loss

        loss.backward()

        # 梯度裁剪：防止梯度爆炸
        if max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        total_loss += loss.item() * y.size(0)
        n += y.size(0)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_pred, all_true = [], []
    n = 0
    for x1, x2, y, lv1, lv2 in loader:
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        lv1, lv2 = lv1.to(device), lv2.to(device)
        y_hat = model(x1, x2, lv1, lv2)
        loss = criterion(y_hat, y)
        total_loss += loss.item() * y.size(0)
        n += y.size(0)
        all_pred.append(y_hat.cpu())
        all_true.append(y.cpu())

    avg_loss = total_loss / n
    pred = torch.cat(all_pred)
    true = torch.cat(all_true)
    mae = (pred - true).abs().mean().item()
    return avg_loss, mae, pred, true


def binary_clf_metrics(pred: torch.Tensor, true: torch.Tensor,
                       threshold: float = 0.0) -> dict:
    """将回归输出二值化后计算二分类指标：accuracy / precision / recall / F1。

    正类（label=1）：true > threshold（即 v1 比 v2 快）。
    负类（label=0）：true <= threshold（即 v2 不慢于 v1）。

    threshold 取值：
        - log 空间（log_target=True）：0.0（对应原始加速比 = 1.0）
        - 原始空间（log_target=False）：1.0（加速比 > 1 表示 v1 更快）
    """
    pred_bin = (pred > threshold).long()
    true_bin = (true > threshold).long()

    tp = int(((pred_bin == 1) & (true_bin == 1)).sum())
    fp = int(((pred_bin == 1) & (true_bin == 0)).sum())
    tn = int(((pred_bin == 0) & (true_bin == 0)).sum())
    fn = int(((pred_bin == 0) & (true_bin == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


@torch.no_grad()
def evaluate_binary(model, loader, criterion, device):
    """二分类评估：返回 (loss, f1, logits, true_labels, metrics_dict)。

    模型输出被视为 logit（未经 sigmoid）；正类判定条件为 logit > 0。
    """
    model.eval()
    total_loss = 0.0
    all_logits, all_true = [], []
    n = 0
    for x1, x2, y, lv1, lv2 in loader:
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        lv1, lv2 = lv1.to(device), lv2.to(device)
        logits = model(x1, x2, lv1, lv2)
        loss = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        n += y.size(0)
        all_logits.append(logits.cpu())
        all_true.append(y.cpu())

    avg_loss = total_loss / n
    logits = torch.cat(all_logits)
    true = torch.cat(all_true)
    # logit > 0 等价于概率 > 0.5；true 为 0/1 浮点标签，阈值 0.0 时 true>0 即正类
    metrics = binary_clf_metrics(logits, true, threshold=0.0)
    return avg_loss, metrics["f1"], logits, true, metrics
