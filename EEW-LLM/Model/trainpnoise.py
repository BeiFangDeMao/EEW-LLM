#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project : LLM
# @File : trainp.py
# @IDE : PyCharm
# @Author : 张嘉南
# @Date : 2026/3/11 上午11:34
"""
P 波震相拾取训练脚本（普通 BCE 损失）
支持噪声数据（标签-1）处理
输入：波形 (3, T)
输出：每个时间点的 P 波概率（经 sigmoid，范围0~1）
标签：有事件样本为高斯概率序列，无事件样本为全零序列
"""

from transformers import get_cosine_schedule_with_warmup
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import logging
import h5py
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import csv
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 调整字体设置
plt.rcParams["font.family"] = ["STIXGeneral", "DejaVu Sans", "SimHei", "Microsoft YaHei"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams['axes.unicode_minus'] = False

# 导入模型
from inputhead import EEW_LLMBART_PP

MODEL_REGISTRY = {
    "EEW_LLMBART_PP": EEW_LLMBART_PP
}

# -------------------------- 1. 基础配置 --------------------------
TRAIN_START_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_DIR = os.path.abspath("")
CHECKPOINT_DIR = os.path.join(BASE_DIR, f"checkpoint_pphase_{TRAIN_START_TIME}")
LOG_DIR = os.path.join(BASE_DIR, f"log_pphase_{TRAIN_START_TIME}")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 数据与训练参数
H5_FILE_PATH = r"G:\迁移学习\LLM\20251209NewTry\2636dataset\JKnet_300_Pphase_FS.h5"
TARGET_LENGTH = 300
TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT = 0.80, 0.15, 0.05
BATCH_SIZE = 96
EPOCHS = 200
WEIGHT_DECAY = 1e-5
EARLY_STOP_PATIENCE = 20
EARLY_STOP_EPS = 1e-6
TOLERANCE = 10
GAUSSIAN_SIGMA = 5
EVENT_THRESHOLD = 0.5  # 事件检测阈值
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------- 2. 日志配置 --------------------------
log_format = logging.Formatter(
    "%(asctime)s - %(name)s - %(module)s - %(levelname)s - %(message)s")
file_handler = logging.FileHandler(
    os.path.join(LOG_DIR, "train_detail.log"), encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(log_format)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(log_format)
logging.basicConfig(
    level=logging.DEBUG, handlers=[file_handler, console_handler])

logging.info("=" * 50)
logging.info(f"训练启动时间：{TRAIN_START_TIME}")
logging.info(f"使用设备：{device}")
logging.info(f"训练参数：")
logging.info(f" H5数据路径：{H5_FILE_PATH}")
logging.info(f" 波形长度：{TARGET_LENGTH}")
logging.info(f" 数据拆分：训练{TRAIN_SPLIT * 100}% | 验证{VAL_SPLIT * 100}% | 测试{TEST_SPLIT * 100}%")
logging.info(f" 批次大小：{BATCH_SIZE} | 总轮次：{EPOCHS}")
logging.info(f" 权重衰减：{WEIGHT_DECAY}")
logging.info(f" 早停阈值：{EARLY_STOP_PATIENCE}轮 | 最小改进：{EARLY_STOP_EPS}")
logging.info(f" 宽容准确率容忍点数：{TOLERANCE}")
logging.info(f" 高斯标签标准差 σ：{GAUSSIAN_SIGMA}")
logging.info(f" 事件检测阈值：{EVENT_THRESHOLD}")
logging.info("=" * 50)


# -------------------------- 3. 数据集类（支持噪声数据）--------------------------
class SeismicDataset(Dataset):
    def __init__(self, h5_file_path, target_length=300, gaussian_sigma=5, max_samples=None):
        self.data_list = []
        self.target_length = target_length
        self.gaussian_sigma = gaussian_sigma
        self.normal_count = 0
        self.noise_count = 0

        with h5py.File(h5_file_path, "r") as f:
            waveform_keys = list(f.keys())
            if 'fft_spectrum' in waveform_keys:
                waveform_keys.remove('fft_spectrum')
            if max_samples:
                waveform_keys = waveform_keys[:max_samples]
            waveform_keys.sort()

            for key in tqdm(waveform_keys, desc="加载H5数据"):
                try:
                    wave_group = f[key]
                    # 加载波形
                    wave = wave_group[:].astype(np.float32).T
                    if wave.shape[0] != 3:
                        continue
                    if wave.shape[1] < self.target_length:
                        pad = self.target_length - wave.shape[1]
                        wave = np.pad(wave, ((0, 0), (0, pad)), mode='constant')
                    else:
                        wave = wave[:, :self.target_length]

                    # 加载 P 波到达索引
                    p_idx_attr = wave_group.attrs.get("new_pat_sample", np.nan)

                    # 处理噪声数据（标签为-1）
                    if p_idx_attr == -1 or (isinstance(p_idx_attr, float) and p_idx_attr == -1.0):
                        # 噪声数据：生成全零标签
                        label_seq = np.zeros(self.target_length, dtype=np.float32)
                        self.noise_count += 1
                    else:
                        # 正常数据：生成高斯标签
                        if np.isnan(p_idx_attr):
                            continue
                        p_idx = int(float(p_idx_attr))
                        if p_idx < 0 or p_idx >= self.target_length:
                            continue

                        x = np.arange(self.target_length)
                        label_seq = np.exp(-(x - p_idx) ** 2 / (2 * self.gaussian_sigma ** 2))
                        label_seq[label_seq < 1e-3] = 0
                        self.normal_count += 1

                    # 波形标准化（每个分量独立）
                    mean = wave.mean(axis=1, keepdims=True)
                    std = wave.std(axis=1, keepdims=True)
                    std[std == 0] = 1.0
                    wave = (wave - mean) / std

                    wave_tensor = torch.from_numpy(wave).to(torch.float32)
                    label_tensor = torch.from_numpy(label_seq).unsqueeze(0).to(torch.float32)  # [1, T]

                    self.data_list.append((wave_tensor, label_tensor, key))

                except Exception as e:
                    logging.warning(f"跳过样本 {key}: {e}")

        logging.info(f"H5数据集加载完成，正常样本数：{self.normal_count}，噪声样本数：{self.noise_count}")
        logging.info(f"总样本数：{len(self.data_list)}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


# -------------------------- 4. 辅助函数 --------------------------
def compute_picking_error(pred_probs, true_labels, tolerance=10):
    """
    计算P波拾取误差（仅对有事件的样本）
    pred_probs: [B,1,T] 或 [B,T] 概率值
    true_labels: [B,1,T] 或 [B,T] 高斯标签
    """
    if pred_probs.dim() == 3:
        pred_probs = pred_probs.squeeze(1)
    if true_labels.dim() == 3:
        true_labels = true_labels.squeeze(1)

    pred_probs = pred_probs.detach().cpu().numpy()
    true_labels = true_labels.detach().cpu().numpy()

    # 判断哪些样本有事件
    has_event = true_labels.max(axis=1) > 0.1

    errors = []
    pred_positions = []
    true_positions = []
    within_tolerance = []

    for i in range(pred_probs.shape[0]):
        if has_event[i]:
            true_pos = np.argmax(true_labels[i])
            pred_pos = np.argmax(pred_probs[i])
            error = abs(pred_pos - true_pos)
            errors.append(error)
            pred_positions.append(pred_pos)
            true_positions.append(true_pos)
            within_tolerance.append(error <= tolerance)

    return errors, pred_positions, true_positions, within_tolerance, has_event


def compute_event_detection_metrics(pred_probs, true_labels, threshold=0.5):
    """
    计算事件检测指标（二分类）
    """
    if pred_probs.dim() == 3:
        pred_probs = pred_probs.squeeze(1)
    if true_labels.dim() == 3:
        true_labels = true_labels.squeeze(1)

    pred_probs = pred_probs.detach().cpu().numpy()
    true_labels = true_labels.detach().cpu().numpy()

    # 每个样本的最大值
    pred_max = pred_probs.max(axis=1)
    true_max = true_labels.max(axis=1)

    # 二分类：有事件（1） vs 无事件（0）
    pred_event = (pred_max > threshold).astype(int)
    true_event = (true_max > 0.1).astype(int)

    # 计算指标
    accuracy = accuracy_score(true_event, pred_event)
    precision = precision_score(true_event, pred_event, zero_division=0)
    recall = recall_score(true_event, pred_event, zero_division=0)
    f1 = f1_score(true_event, pred_event, zero_division=0)
    cm = confusion_matrix(true_event, pred_event)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'true_event': true_event,
        'pred_event': pred_event
    }


def compute_comprehensive_metrics(pred_probs, true_labels, tolerance=10, event_threshold=0.5):
    """
    计算所有评估指标
    """
    # 事件检测指标
    event_metrics = compute_event_detection_metrics(pred_probs, true_labels, threshold=event_threshold)

    # P波拾取指标
    errors, pred_pos, true_pos, within_tol, has_event = compute_picking_error(
        pred_probs, true_labels, tolerance=tolerance
    )

    # 统计漏检和虚警
    missed_detections = np.sum((event_metrics['true_event'] == 1) & (event_metrics['pred_event'] == 0))
    false_alarms = np.sum((event_metrics['true_event'] == 0) & (event_metrics['pred_event'] == 1))

    # P波拾取统计
    if len(errors) > 0:
        picking_metrics = {
            'mae': np.mean(errors),
            'mse': np.mean(np.square(errors)),
            'rmse': np.sqrt(np.mean(np.square(errors))),
            'median': np.median(errors),
            'std': np.std(errors),
            'tolerance_accuracy': np.mean(within_tol),
            'total_samples': len(errors)
        }
    else:
        picking_metrics = {
            'mae': 0.0, 'mse': 0.0, 'rmse': 0.0,
            'median': 0.0, 'std': 0.0, 'tolerance_accuracy': 0.0, 'total_samples': 0
        }

    return {
        'event_detection': event_metrics,
        'picking': picking_metrics,
        'missed_detections': missed_detections,
        'false_alarms': false_alarms,
        'errors': errors,
        'pred_positions': pred_pos,
        'true_positions': true_pos,
        'within_tolerance': within_tol,
        'has_event': has_event
    }


# -------------------------- 5. 训练/验证/测试函数 --------------------------
def train_one_epoch(model, loader, criterion, optimizer, scheduler, backbone_params, head_params, epoch):
    model.train()
    total_loss = 0.0
    all_errors = []
    total_samples = 0

    for i, (wave, y, _) in enumerate(tqdm(loader, desc=f"训练 Epoch {epoch}")):
        wave, y = wave.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(wave)

        loss = criterion(out, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(backbone_params, max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(head_params, max_norm=5.0)
        optimizer.step()
        scheduler.step()

        batch_size = len(wave)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # 计算拾取误差（仅对有事件的样本）
        with torch.no_grad():
            errors, _, _, _, _ = compute_picking_error(out, y, tolerance=TOLERANCE)
            all_errors.extend(errors)

        if (i + 1) % 100 == 0:
            avg_err = np.mean(all_errors[-len(errors):]) if all_errors else 0
            logging.debug(
                f"Epoch {epoch} 批次 {i + 1}/{len(loader)} - 损失: {loss.item():.6f} | 批次拾取误差: {avg_err:.2f}")

    epoch_loss = total_loss / total_samples
    epoch_pick_err = np.mean(all_errors) if all_errors else 0.0
    return epoch_loss, epoch_pick_err


def evaluate(model, loader, criterion, tolerance=10, event_threshold=0.5, mode="验证"):
    """验证函数，输出详细评估指标"""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    all_pred_probs = []
    all_true_labels = []

    with torch.no_grad():
        for wave, y, _ in tqdm(loader, desc=mode):
            wave, y = wave.to(device), y.to(device)
            out = model(wave)

            loss = criterion(out, y)
            batch_size = len(wave)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_pred_probs.append(out.cpu())
            all_true_labels.append(y.cpu())

    # 合并所有结果
    pred_probs = torch.cat(all_pred_probs, dim=0)
    true_labels = torch.cat(all_true_labels, dim=0)
    avg_loss = total_loss / total_samples

    # 计算综合指标
    metrics = compute_comprehensive_metrics(pred_probs, true_labels, tolerance, event_threshold)

    # 打印结果
    logging.info(f"\n{'=' * 50}")
    logging.info(f"{mode}结果:")
    logging.info(f"  损失: {avg_loss:.6f}")
    logging.info(
        f"  样本统计 - 总样本: {total_samples}, 事件样本: {sum(metrics['has_event'])}, 噪声样本: {total_samples - sum(metrics['has_event'])}")

    logging.info(f"\n  事件检测指标:")
    logging.info(f"    准确率: {metrics['event_detection']['accuracy']:.4f}")
    logging.info(f"    精确率: {metrics['event_detection']['precision']:.4f}")
    logging.info(f"    召回率: {metrics['event_detection']['recall']:.4f}")
    logging.info(f"    F1分数: {metrics['event_detection']['f1_score']:.4f}")
    logging.info(f"    漏检数: {metrics['missed_detections']} (真实有事件但预测为无)")
    logging.info(f"    虚警数: {metrics['false_alarms']} (真实无事件但预测为有)")

    if metrics['picking']['total_samples'] > 0:
        logging.info(f"\n  P波拾取指标 (仅对真实事件样本，共{metrics['picking']['total_samples']}个):")
        logging.info(f"    MAE: {metrics['picking']['mae']:.2f} 点")
        logging.info(f"    MSE: {metrics['picking']['mse']:.2f}")
        logging.info(f"    RMSE: {metrics['picking']['rmse']:.2f}")
        logging.info(f"    中位数误差: {metrics['picking']['median']:.2f} 点")
        logging.info(f"    标准差: {metrics['picking']['std']:.2f}")
        logging.info(f"    宽容准确率(±{tolerance}): {metrics['picking']['tolerance_accuracy']:.4f}")
    logging.info(f"{'=' * 50}\n")

    return avg_loss, metrics['picking']['mae'], metrics['event_detection']['accuracy'], metrics


def test_model(model, loader, criterion, tolerance=10, event_threshold=0.5):
    """测试函数，返回详细结果"""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    all_pred_probs = []
    all_true_labels = []
    all_keys = []

    with torch.no_grad():
        for wave, y, key in tqdm(loader, desc="测试中"):
            wave, y = wave.to(device), y.to(device)
            out = model(wave)

            loss = criterion(out, y)
            batch_size = len(wave)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_pred_probs.append(out.cpu())
            all_true_labels.append(y.cpu())
            all_keys.extend(key)

    # 合并所有结果
    pred_probs = torch.cat(all_pred_probs, dim=0)
    true_labels = torch.cat(all_true_labels, dim=0)
    avg_loss = total_loss / total_samples

    # 计算综合指标
    metrics = compute_comprehensive_metrics(pred_probs, true_labels, tolerance, event_threshold)

    # 打印结果
    logging.info(f"\n{'=' * 50}")
    logging.info(f"测试结果汇总:")
    logging.info(f"  总样本数: {total_samples}")
    logging.info(f"  事件样本数: {sum(metrics['has_event'])}")
    logging.info(f"  噪声样本数: {total_samples - sum(metrics['has_event'])}")
    logging.info(f"  损失: {avg_loss:.6f}")

    logging.info(f"\n  事件检测指标:")
    logging.info(f"    准确率: {metrics['event_detection']['accuracy']:.4f}")
    logging.info(f"    精确率: {metrics['event_detection']['precision']:.4f}")
    logging.info(f"    召回率: {metrics['event_detection']['recall']:.4f}")
    logging.info(f"    F1分数: {metrics['event_detection']['f1_score']:.4f}")
    logging.info(f"    漏检数: {metrics['missed_detections']}")
    logging.info(f"    虚警数: {metrics['false_alarms']}")

    if metrics['picking']['total_samples'] > 0:
        logging.info(f"\n  P波拾取指标 (仅对真实事件样本):")
        logging.info(f"    MAE: {metrics['picking']['mae']:.2f} 点")
        logging.info(f"    MSE: {metrics['picking']['mse']:.2f}")
        logging.info(f"    RMSE: {metrics['picking']['rmse']:.2f}")
        logging.info(f"    中位数误差: {metrics['picking']['median']:.2f} 点")
        logging.info(f"    标准差: {metrics['picking']['std']:.2f}")
        logging.info(f"    宽容准确率(±{tolerance}): {metrics['picking']['tolerance_accuracy']:.4f}")
    logging.info(f"{'=' * 50}\n")

    return metrics, avg_loss, all_keys


def save_test_table(metrics, keys, save_dir, timestamp, tolerance):
    """保存测试表格"""
    csv_path = os.path.join(save_dir, f"test_table_{timestamp}.csv")

    true_event = metrics['event_detection']['true_event']
    pred_event = metrics['event_detection']['pred_event']

    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            "事件ID", "真实类型", "预测类型",
            "真实P波位置", "预测P波位置", "拾取误差",
            f"是否在容忍±{tolerance}内", "事件检测是否正确"
        ])

        # 创建映射：每个有事件样本的拾取信息
        pick_info = {}
        event_idx = 0
        for i, has_event in enumerate(metrics['has_event']):
            if has_event:
                pick_info[i] = {
                    'true_pos': metrics['true_positions'][event_idx],
                    'pred_pos': metrics['pred_positions'][event_idx],
                    'error': metrics['errors'][event_idx],
                    'within': metrics['within_tolerance'][event_idx]
                }
                event_idx += 1

        for i, key in enumerate(keys):
            true_type = "有事件" if true_event[i] == 1 else "无事件"
            pred_type = "有事件" if pred_event[i] == 1 else "无事件"
            event_correct = (true_event[i] == pred_event[i])

            if true_event[i] == 1 and i in pick_info:
                true_pos = pick_info[i]['true_pos']
                pred_pos = pick_info[i]['pred_pos']
                error = pick_info[i]['error']
                within = pick_info[i]['within']
            else:
                true_pos = "-"
                pred_pos = "-"
                error = "-"
                within = "-"

            writer.writerow([
                key, true_type, pred_type,
                true_pos, pred_pos, error,
                within, "是" if event_correct else "否"
            ])

    logging.info(f"测试表格已保存至：{csv_path}")


def plot_results(train_losses, val_losses, train_pick_errs, val_pick_errs,
                 val_event_acc, val_f1_scores, test_metrics, save_dir, timestamp, tolerance):
    """绘制训练和测试结果图"""
    plt.figure(figsize=(20, 12))

    # 1. 损失曲线
    plt.subplot(2, 3, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='训练损失')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='验证损失')
    plt.title('训练与验证损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. 拾取误差曲线
    plt.subplot(2, 3, 2)
    plt.plot(range(1, len(train_pick_errs) + 1), train_pick_errs, label='训练拾取误差')
    plt.plot(range(1, len(val_pick_errs) + 1), val_pick_errs, label='验证拾取误差')
    plt.title('训练与验证拾取误差曲线')
    plt.xlabel('Epoch')
    plt.ylabel('平均绝对误差（采样点）')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. 事件检测指标曲线
    plt.subplot(2, 3, 3)
    plt.plot(range(1, len(val_event_acc) + 1), val_event_acc, label='事件检测准确率', color='green')
    plt.plot(range(1, len(val_f1_scores) + 1), val_f1_scores, label='F1分数', color='orange')
    plt.title('事件检测指标曲线')
    plt.xlabel('Epoch')
    plt.ylabel('分数')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4. 事件检测混淆矩阵
    plt.subplot(2, 3, 4)
    cm = test_metrics['event_detection']['confusion_matrix']
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('测试集事件检测混淆矩阵')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['无事件', '有事件'])
    plt.yticks(tick_marks, ['无事件', '有事件'])
    plt.xlabel('预测')
    plt.ylabel('真实')

    # 添加数值标签
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # 5. 真实vs预测位置散点图
    plt.subplot(2, 3, 5)
    if len(test_metrics['true_positions']) > 0:
        plt.scatter(test_metrics['true_positions'], test_metrics['pred_positions'],
                    s=10, alpha=0.6, c='blue', label='预测点')
        min_val = min(min(test_metrics['true_positions']), min(test_metrics['pred_positions']))
        max_val = max(max(test_metrics['true_positions']), max(test_metrics['pred_positions']))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='理想线')
        plt.fill_between([min_val, max_val], min_val - tolerance, max_val + tolerance,
                         color='green', alpha=0.2, label=f'±{tolerance}容忍带')
        plt.xlabel('真实位置')
        plt.ylabel('预测位置')
        plt.title(
            f'真实 vs 预测 P 波位置\n(MAE={test_metrics["picking"]["mae"]:.2f}点, 准确率={test_metrics["picking"]["tolerance_accuracy"]:.3f})')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # 6. 拾取误差分布
    plt.subplot(2, 3, 6)
    if len(test_metrics['errors']) > 0:
        plt.hist(test_metrics['errors'], bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(x=tolerance, color='r', linestyle='--', label=f'容忍阈值 (±{tolerance})')
        plt.xlabel('拾取误差（采样点）')
        plt.ylabel('频数')
        plt.title(f'P波拾取误差分布 (MAE={test_metrics["picking"]["mae"]:.2f})')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"training_test_results_{timestamp}.png")
    plt.savefig(save_path, dpi=300)
    logging.info(f"训练与测试结果图已保存：{save_path}")


def set_deterministic(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# -------------------------- 6. 主函数 --------------------------
def main(model_name="EEW_LLMBART_PP", h5_path=None):
    set_deterministic(42)

    if h5_path is None:
        raise ValueError("必须提供 H5 数据文件路径！")

    # 加载数据集
    logging.info("开始加载波形数据集...")
    dataset = SeismicDataset(
        h5_file_path=h5_path,
        target_length=TARGET_LENGTH,
        gaussian_sigma=GAUSSIAN_SIGMA
    )
    total = len(dataset)
    if total == 0:
        logging.error("数据集无有效样本，终止训练！")
        return

    # 拆分数据集
    train_size, val_size = int(total * TRAIN_SPLIT), int(total * VAL_SPLIT)
    test_size = total - train_size - val_size
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )
    logging.info(f"数据集拆分完成: 训练集{len(train_ds)} | 验证集{len(val_ds)} | 测试集{len(test_ds)}")

    # 数据加载器
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False)

    # 初始化模型
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"未知模型名称: {model_name}。可选: {list(MODEL_REGISTRY.keys())}")
    model_class = MODEL_REGISTRY[model_name]
    model = model_class().to(device)
    logging.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 损失函数
    criterion = nn.BCELoss()

    # 参数分组
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    logging.info(f"参数分组完成：Backbone (LoRA) {len(backbone_params)} 个 | Head {len(head_params)} 个")

    # 优化器
    BACKBONE_LR = 2e-4
    HEAD_LR = 1e-3
    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": BACKBONE_LR},
            {"params": head_params, "lr": HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # 学习率调度器
    total_steps = len(train_loader) * EPOCHS
    num_warmup_steps = int(0.05 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps,
        num_cycles=0.5,
    )

    # 最佳模型跟踪
    best_metrics = {
        "val_pick_err": float("inf"),
        "val_loss": float("inf"),
        "val_event_acc": 0.0,
        "epoch": 0,
        "model_state": None
    }
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"best_model_{TRAIN_START_TIME}.pth")
    early_stop_counter = 0

    # 记录训练过程指标
    train_losses = []
    val_losses = []
    train_pick_errs = []
    val_pick_errs = []
    val_event_accs = []
    val_f1_scores = []

    # 训练循环
    logging.info("开始训练...")
    for epoch in range(1, EPOCHS + 1):
        logging.info(f"\n{'=' * 20} Epoch {epoch}/{EPOCHS} {'=' * 20}")
        logging.info(
            f"当前学习率: Backbone={optimizer.param_groups[0]['lr']:.6e}, Head={optimizer.param_groups[1]['lr']:.6e}")

        train_loss, train_pick = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            backbone_params, head_params, epoch
        )
        val_loss, val_pick, val_event_acc, val_metrics = evaluate(
            model, val_loader, criterion, tolerance=TOLERANCE,
            event_threshold=EVENT_THRESHOLD, mode="验证"
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_pick_errs.append(train_pick)
        val_pick_errs.append(val_pick)
        val_event_accs.append(val_event_acc)
        val_f1_scores.append(val_metrics['event_detection']['f1_score'])

        logging.info(f"训练集 - 损失: {train_loss:.6f} | 拾取误差: {train_pick:.2f}")
        logging.info(f"验证集 - 损失: {val_loss:.6f} | 拾取误差: {val_pick:.2f} | 事件检测准确率: {val_event_acc:.4f}")

        # 根据验证拾取误差保存最佳模型
        if val_pick < best_metrics["val_pick_err"] - EARLY_STOP_EPS:
            best_metrics.update({
                "val_pick_err": val_pick,
                "val_loss": val_loss,
                "val_event_acc": val_event_acc,
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            })
            torch.save(best_metrics, ckpt_path)
            logging.info(
                f"更新最佳模型 (Epoch {epoch}) - 拾取误差: {val_pick:.2f} | 事件检测准确率: {val_event_acc:.4f}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            logging.info(f"未更新最佳模型 ({early_stop_counter}/{EARLY_STOP_PATIENCE})")

        if early_stop_counter >= EARLY_STOP_PATIENCE:
            logging.info(f"早停触发！最佳模型出现在 Epoch {best_metrics['epoch']}")
            break

    # 测试最佳模型
    logging.info("\n" + "=" * 50)
    logging.info("开始测试最佳模型...")
    best_checkpoint = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state"])
    logging.info(
        f"加载最佳模型 (Epoch {best_checkpoint['epoch']}): 验证拾取误差={best_checkpoint['val_pick_err']:.2f}, 验证事件检测准确率={best_checkpoint['val_event_acc']:.4f}")

    test_metrics, test_loss, test_keys = test_model(
        model, test_loader, criterion, tolerance=TOLERANCE, event_threshold=EVENT_THRESHOLD
    )

    # 保存测试表格
    save_test_table(test_metrics, test_keys, CHECKPOINT_DIR, TRAIN_START_TIME, TOLERANCE)

    # 绘制结果图
    plot_results(
        train_losses, val_losses, train_pick_errs, val_pick_errs,
        val_event_accs, val_f1_scores, test_metrics,
        CHECKPOINT_DIR, TRAIN_START_TIME, TOLERANCE
    )

    logging.info("所有训练流程完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train P-phase picking model with noise data support")
    parser.add_argument("--model_name", type=str, default="EEW_LLMBART_PP",
                        choices=list(MODEL_REGISTRY.keys()),
                        help=f"Model to train. Options: {list(MODEL_REGISTRY.keys())}")
    parser.add_argument("--h5_path", type=str,
                        default=r"G:\迁移学习\LLM\20251209NewTry\2636dataset\JKnet_3001_Pphase_noise.h5",
                        help="Path to the H5 dataset file")
    args = parser.parse_args()
    main(model_name=args.model_name, h5_path=args.h5_path)