#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project : LM
# @File : train_multitask.py
# @IDE : PyCharm
# @Author : 张嘉南
# @Date : 2025/12/28 下午6:37
"""
多任务地震参数估计模型训练脚本 (方位角+震中距+震级)
适配三输入模型（wave, spec, feat），支持同时预测多个目标
"""
from transformers import get_cosine_schedule_with_warmup
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
import h5py
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import csv
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup

# 调整字体设置
plt.rcParams["font.family"] = ["STIXGeneral", "DejaVu Sans", "SimHei", "Microsoft YaHei"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams['axes.unicode_minus'] = False

# 导入模型
from EEWLMBART_duo import EEW_LMBART

MODEL_REGISTRY = {
    "EEW_LMBART": EEW_LMBART,
}

# 定义15个特征名称列表（与数据集保持一致）
FEATURE_NAMES = [
    'Pa', 'Pv', 'Pd', 'Pav', 'Pad', 'Pvd', 'IAA', 'IAV', 'IAD', 'IV2', 'Ia', 'Tc', 'TP', 'DI', 'Tva'
]


def save_test_table(all_keys, all_trues_azi, all_preds_azi, all_trues_dis, all_preds_dis, all_trues_mag, all_preds_mag,
                    save_dir, timestamp):
    """生成测试结果表格 (事件ID, 真实值, 预测值, 残差)"""
    csv_path = os.path.join(save_dir, f"multitask_test_table_{timestamp}.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            "事件ID",
            "真实方位角", "预测方位角", "方位角残差",
            "真实震中距", "预测震中距", "震中距残差",
            "真实震级", "预测震级", "震级残差"
        ])
        for k, t_azi, p_azi, t_dis, p_dis, t_mag, p_mag in zip(
                all_keys, all_trues_azi, all_preds_azi,
                all_trues_dis, all_preds_dis,
                all_trues_mag, all_preds_mag
        ):
            writer.writerow([
                k,
                round(float(t_azi), 4), round(float(p_azi), 4), round(float(t_azi - p_azi), 4),
                round(float(t_dis), 4), round(float(p_dis), 4), round(float(t_dis - p_dis), 4),
                round(float(t_mag), 4), round(float(p_mag), 4), round(float(t_mag - p_mag), 4)
            ])
    logging.info(f"多任务测试表格已保存至：{csv_path}")


def baz_loss(pred, target):
    """ pred, target: [B, 2] -> [sinθ, cosθ] """
    # 确保pred是正确的形状 [B, 2]
    if isinstance(pred, tuple):
        # 如果pred是元组 (sin, cos)，将其合并为 [B, 2]
        pred = torch.cat([pred[0], pred[1]], dim=1)
    elif pred.shape[1] == 1:
        # 如果只有1维，需要检查是否是sin和cos分别堆叠的情况
        # 根据实际模型输出调整
        pass

    loss_sin = F.mse_loss(pred[:, 0], target[:, 0])
    loss_cos = F.mse_loss(pred[:, 1], target[:, 1])
    return loss_sin + loss_cos


def sincos_to_deg(sin_v, cos_v):
    return (np.degrees(np.arctan2(sin_v, cos_v)) + 360) % 360


def baz_angle_error(pred, target):
    """ pred, target: [N, 2] return: mean absolute angular error (degree) """
    # 处理pred的格式
    if isinstance(pred, tuple):
        pred_sin = pred[0].cpu().numpy().squeeze()
        pred_cos = pred[1].cpu().numpy().squeeze()
        pred_deg = sincos_to_deg(pred_sin, pred_cos)
    else:
        if pred.shape[1] == 2:
            pred_deg = sincos_to_deg(pred[:, 0], pred[:, 1])
        else:
            pred_deg = sincos_to_deg(pred[:, 0], pred[:, 1])

    # 处理target的格式
    if target.shape[1] == 2:
        true_deg = sincos_to_deg(target[:, 0], target[:, 1])
    else:
        true_deg = sincos_to_deg(target[:, 0], target[:, 1])

    diff = np.abs(pred_deg - true_deg)
    diff = np.minimum(diff, 360 - diff)
    return diff.mean()


class WeightedLogMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_log, true_log):
        """ pred_log, true_log: shape [B, 1] """
        # MSE
        mse = (pred_log - true_log) ** 2
        # log 距离对应的真实 km
        true_dist = torch.exp(true_log)
        # 分段权重（单位 km）
        weights = torch.ones_like(true_dist)
        weights = torch.where(true_dist < 100, 0.5 * weights, weights)
        weights = torch.where((true_dist >= 100) & (true_dist < 150), 1.0 * weights, weights)
        weights = torch.where(true_dist >= 150, 1.5 * weights, weights)
        # 加权 MSE
        weighted_mse = mse * weights
        return weighted_mse.mean()


# -------------------------- 1. 基础配置 --------------------------
TRAIN_START_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_DIR = os.path.abspath("")
CHECKPOINT_DIR = os.path.join(BASE_DIR, f"checkpoint_multi_{TRAIN_START_TIME}")
LOG_DIR = os.path.join(BASE_DIR, f"log_multi_{TRAIN_START_TIME}")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 数据与训练参数 - 适配H5三输入数据集
H5_FILE_PATH = r"/20251209NewTry/dataset/JKnet_300_with_FS.h5"  # 替换为你的H5文件路径
TARGET_LENGTH = 300
SPEC_LENGTH = 128
TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT = 0.80, 0.15, 0.05
BATCH_SIZE = 96
EPOCHS = 200
WEIGHT_DECAY = 1e-5
EARLY_STOP_PATIENCE = 20
EARLY_STOP_EPS = 1e-6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------- 2. 增强版日志配置 --------------------------
log_format = logging.Formatter(
    "%(asctime)s - %(name)s - %(module)s - %(levelname)s - %(message)s")
file_handler = logging.FileHandler(
    os.path.join(LOG_DIR, "multitask_train_detail.log"), encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(log_format)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(log_format)
logging.basicConfig(
    level=logging.DEBUG, handlers=[file_handler, console_handler])

logging.info("=" * 50)
logging.info(f"多任务训练启动时间：{TRAIN_START_TIME}")
logging.info(f"使用设备：{device}")
logging.info(f"训练参数：")
logging.info(f" H5数据路径：{H5_FILE_PATH}")
logging.info(f" 数据尺寸：波形(3,{TARGET_LENGTH}) | 频谱(3,{SPEC_LENGTH}) | 特征(15,)")
logging.info(f" 数据拆分：训练{TRAIN_SPLIT * 100}% | 验证{VAL_SPLIT * 100}% | 测试{TEST_SPLIT * 100}%")
logging.info(f" 批次大小：{BATCH_SIZE} | 总轮次：{EPOCHS}")
logging.info(f" 权重衰减：{WEIGHT_DECAY}")
logging.info(f" 早停阈值：{EARLY_STOP_PATIENCE}轮 | 最小改进：{EARLY_STOP_EPS}")
logging.info("=" * 50)


# -------------------------- 3. 正确的三输入H5数据集 --------------------------
class SeismicDataset(Dataset):
    def __init__(self, h5_file_path, target_length=300, spec_length=128, max_samples=None):
        self.data_list = []  # 存储格式：(wave, spec, feat, label_azi, label_dis, label_mag, key)
        self.target_length = target_length
        self.spec_length = spec_length
        with h5py.File(h5_file_path, "r") as f:
            if 'fft_spectrum' not in f.keys() or not isinstance(f['fft_spectrum'], h5py.Group):
                logging.error("H5文件中未找到有效频谱组 'fft_spectrum'！")
                return
            spec_group = f['fft_spectrum']
            waveform_keys = list(f.keys())
            spec_keys = list(spec_group.keys())
            valid_sample_keys = list(set(waveform_keys) & set(spec_keys))
            if len(valid_sample_keys) == 0:
                logging.error("无有效样本（波形与频谱key无交集）！")
                return
            if max_samples:
                valid_sample_keys = valid_sample_keys[:max_samples]
            valid_sample_keys.sort()
            for key in tqdm(valid_sample_keys, desc="加载H5三输入数据"):
                try:
                    wave_group = f[key]
                    wave = wave_group[:].astype(np.float32).T
                    if wave.shape[0] != 3:
                        logging.warning(f"样本{key}波形通道数不为3，跳过")
                        continue
                    if wave.shape[1] < self.target_length:
                        pad = self.target_length - wave.shape[1]
                        wave = np.pad(wave, ((0, 0), (0, pad)), mode='constant')
                    else:
                        wave = wave[:, :self.target_length]

                    spec_obj = spec_group[key]
                    if isinstance(spec_obj, h5py.Group):
                        if not all(ds in spec_obj.keys() for ds in ['amp_ud', 'amp_ns', 'amp_ew']):
                            logging.warning(f"样本{key}频谱缺失子数据集，跳过")
                            continue
                        amp_ud = spec_obj['amp_ud'][:].astype(np.float32)
                        amp_ns = spec_obj['amp_ns'][:].astype(np.float32)
                        amp_ew = spec_obj['amp_ew'][:].astype(np.float32)
                        spec = np.vstack([amp_ud, amp_ns, amp_ew])
                    else:
                        spec = spec_obj[:].astype(np.float32)
                        if spec.shape[-1] == 3 and spec.shape[0] != 3:
                            spec = spec.T
                    if spec.shape[1] < self.spec_length:
                        pad = self.spec_length - spec.shape[1]
                        spec = np.pad(spec, ((0, 0), (0, pad)), mode='constant')
                    else:
                        spec = spec[:, :self.spec_length]
                    if spec.shape != (3, self.spec_length):
                        logging.warning(f"样本{key}频谱形状异常，跳过")
                        continue

                    feat = []
                    for feat_name in FEATURE_NAMES:
                        feat_value = wave_group.attrs.get(feat_name, np.nan)
                        try:
                            feat_float = float(feat_value) if not np.isnan(float(feat_value)) else 0.0
                        except (ValueError, TypeError):
                            feat_float = 0.0
                        feat.append(feat_float)
                    feat = np.array(feat, dtype=np.float32)
                    if feat.shape[0] != 15:
                        logging.warning(f"样本{key}特征维度不为15，跳过")
                        continue

                    # 加载三个任务的标签
                    azi_deg = float(wave_group.attrs.get("azi", np.nan))
                    dis = float(wave_group.attrs.get("dis", np.nan))
                    mag = float(wave_group.attrs.get("mag", np.nan))

                    # 检查有效性
                    if np.isnan(azi_deg) or np.isnan(dis) or np.isnan(mag):
                        continue
                    if dis <= 0 or mag <= 0:
                        continue

                    # 转换标签
                    # 方位角 -> sin/cos
                    azi_rad = np.deg2rad(azi_deg)
                    label_azi = np.array([np.sin(azi_rad), np.cos(azi_rad)], dtype=np.float32)

                    # 震中距 -> log距离
                    eps = 1e-3
                    label_dis = np.log(dis + eps)

                    # 震级 -> 直接使用
                    label_mag = np.array([mag], dtype=np.float32)

                    wave_tensor = torch.from_numpy(wave).to(torch.float32)
                    spec_tensor = torch.from_numpy(spec).to(torch.float32)
                    feat_tensor = torch.from_numpy(feat).to(torch.float32)
                    label_azi_tensor = torch.tensor(label_azi, dtype=torch.float32)
                    label_dis_tensor = torch.tensor(label_dis, dtype=torch.float32)
                    label_mag_tensor = torch.tensor(label_mag, dtype=torch.float32)

                    self.data_list.append((wave_tensor, spec_tensor, feat_tensor, label_azi_tensor, label_dis_tensor,
                                           label_mag_tensor, key))
                except Exception as e:
                    logging.warning(f"跳过样本 {key}: {e}")
        logging.info(f"H5数据集加载完成，有效样本数：{len(self.data_list)}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if len(self.data_list) == 0:
            raise ValueError("数据集无有效样本，无法获取数据！")
        return self.data_list[idx]


# -------------------------- 4. 多任务训练/验证/测试函数 --------------------------
def train_one_epoch(model, loader, criterion_azi, criterion_dis, criterion_mag, optimizer, scheduler, backbone_params,
                    head_params, epoch):
    model.train()
    total_loss, total_azi_loss, total_dis_loss, total_mag_loss = 0, 0, 0, 0
    azi_preds, azi_labels = [], []
    dis_preds, dis_labels = [], []
    mag_preds, mag_labels = [], []

    for i, (wave, spec, feat, y_azi, y_dis, y_mag, _) in enumerate(tqdm(loader, desc=f"训练 Epoch {epoch}")):
        wave, spec, feat = wave.to(device), spec.to(device), feat.to(device)
        y_azi, y_dis, y_mag = y_azi.to(device), y_dis.to(device), y_mag.to(device)

        optimizer.zero_grad()

        # 前向传播 - 获取三个任务的输出
        pred_azi, pred_mag, pred_dis = model(wave, spec, feat)

        # 处理方位角输出 - 如果是元组则拼接为 [B, 2]
        if isinstance(pred_azi, tuple):
            pred_azi = torch.cat([pred_azi[0], pred_azi[1]], dim=1)  # [B, 1] + [B, 1] -> [B, 2]

        # 处理震中距和震级输出 - 确保是 [B, 1] 形状
        if isinstance(pred_dis, tuple):
            pred_dis = pred_dis[0] if len(pred_dis) == 1 else torch.cat(pred_dis, dim=1)
        if pred_dis.dim() == 1:
            pred_dis = pred_dis.unsqueeze(1)
        if pred_dis.shape[1] != 1:
            pred_dis = pred_dis.view(-1, 1)

        if isinstance(pred_mag, tuple):
            pred_mag = pred_mag[0] if len(pred_mag) == 1 else torch.cat(pred_mag, dim=1)
        if pred_mag.dim() == 1:
            pred_mag = pred_mag.unsqueeze(1)
        if pred_mag.shape[1] != 1:
            pred_mag = pred_mag.view(-1, 1)

        # 确保pred_azi是[B, 2]形状
        if pred_azi.shape[1] != 2:
            if pred_azi.shape[1] == 1:
                # 如果只有一个输出，这可能是一个问题，需要根据实际情况调整
                # 这里假设模型应该输出两个值（sin和cos）
                logging.error(f"方位角输出形状不正确: {pred_azi.shape}")
                continue
            else:
                # 如果有多于2个输出，取前两个
                pred_azi = pred_azi[:, :2]

        # 计算各任务损失
        loss_azi = criterion_azi(pred_azi, y_azi)
        loss_dis = criterion_dis(pred_dis, y_dis)
        loss_mag = criterion_mag(pred_mag, y_mag)

        # 总损失（等权重）
        total_task_loss = loss_azi + loss_dis + loss_mag

        total_task_loss.backward()

        # 分组梯度裁剪
        torch.nn.utils.clip_grad_norm_(backbone_params, max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(head_params, max_norm=5.0)
        optimizer.step()
        scheduler.step()  # Step-based scheduler

        # 累计损失
        total_loss += total_task_loss.item()
        total_azi_loss += loss_azi.item()
        total_dis_loss += loss_dis.item()
        total_mag_loss += loss_mag.item()

        # 收集预测和标签用于评估
        # 方位角：转换回角度
        pred_azi_cpu = pred_azi.detach().cpu().numpy()
        true_azi_cpu = y_azi.cpu().numpy()
        pred_azi_deg = sincos_to_deg(pred_azi_cpu[:, 0], pred_azi_cpu[:, 1])
        true_azi_deg = sincos_to_deg(true_azi_cpu[:, 0], true_azi_cpu[:, 1])
        azi_preds.extend(pred_azi_deg.tolist())
        azi_labels.extend(true_azi_deg.tolist())

        # 震中距：exp转换
        pred_dis_km = np.exp(pred_dis.detach().cpu().numpy().squeeze())
        true_dis_km = np.exp(y_dis.cpu().numpy().squeeze())
        dis_preds.extend(pred_dis_km.tolist())
        dis_labels.extend(true_dis_km.tolist())

        # 震级：直接使用
        mag_preds.extend(pred_mag.detach().cpu().numpy().squeeze().tolist())
        mag_labels.extend(y_mag.cpu().numpy().squeeze().tolist())

        if (i + 1) % 100 == 0:
            batch_azi_err = baz_angle_error(
                pred_azi.detach().cpu().numpy(), y_azi.cpu().numpy()
            )
            batch_dis_mae = mean_absolute_error(dis_labels[-len(y_dis):], dis_preds[-len(y_dis):])
            batch_mag_mae = mean_absolute_error(mag_labels[-len(y_mag):], mag_preds[-len(y_mag):])

            logging.debug(
                f"Epoch {epoch} 批次 {i + 1}/{len(loader)} - "
                f"总损失: {total_task_loss.item():.6f} | "
                f"方位角损失: {loss_azi.item():.6f} | "
                f"震中距损失: {loss_dis.item():.6f} | "
                f"震级损失: {loss_mag.item():.6f} | "
                f"方位角误差: {batch_azi_err:.3f}° | "
                f"震中距MAE: {batch_dis_mae:.6f} | "
                f"震级MAE: {batch_mag_mae:.6f}"
            )

    avg_total_loss = total_loss / len(loader)
    avg_azi_loss = total_azi_loss / len(loader)
    avg_dis_loss = total_dis_loss / len(loader)
    avg_mag_loss = total_mag_loss / len(loader)

    # 计算整体评估指标
    avg_azi_err = baz_angle_error(np.column_stack([np.sin(np.deg2rad(azi_preds)), np.cos(np.deg2rad(azi_preds))]),
                                  np.column_stack([np.sin(np.deg2rad(azi_labels)), np.cos(np.deg2rad(azi_labels))]))
    avg_dis_mae = mean_absolute_error(dis_labels, dis_preds)
    avg_mag_mae = mean_absolute_error(mag_labels, mag_preds)

    logging.debug(f"Epoch {epoch} 训练结束 - "
                  f"平均总损失: {avg_total_loss:.6f} | "
                  f"方位角损失: {avg_azi_loss:.6f} | "
                  f"震中距损失: {avg_dis_loss:.6f} | "
                  f"震级损失: {avg_mag_loss:.6f} | "
                  f"方位角误差: {avg_azi_err:.3f}° | "
                  f"震中距MAE: {avg_dis_mae:.6f} | "
                  f"震级MAE: {avg_mag_mae:.6f}")

    return avg_total_loss, avg_azi_loss, avg_dis_loss, avg_mag_loss, avg_azi_err, avg_dis_mae, avg_mag_mae


def evaluate(model, loader, criterion_azi, criterion_dis, criterion_mag, mode="验证"):
    model.eval()
    total_loss, total_azi_loss, total_dis_loss, total_mag_loss = 0, 0, 0, 0
    azi_preds, azi_labels = [], []
    dis_preds, dis_labels = [], []
    mag_preds, mag_labels = [], []

    with torch.no_grad():
        for wave, spec, feat, y_azi, y_dis, y_mag, _ in tqdm(loader, desc=mode):
            wave, spec, feat = wave.to(device), spec.to(device), feat.to(device)
            y_azi, y_dis, y_mag = y_azi.to(device), y_dis.to(device), y_mag.to(device)

            # 前向传播
            pred_azi, pred_mag, pred_dis = model(wave, spec, feat)

            # 处理方位角输出
            if isinstance(pred_azi, tuple):
                pred_azi = torch.cat([pred_azi[0], pred_azi[1]], dim=1)  # [B, 2]

            # 处理震中距和震级输出
            if isinstance(pred_dis, tuple):
                pred_dis = pred_dis[0] if len(pred_dis) == 1 else torch.cat(pred_dis, dim=1)
            if pred_dis.dim() == 1:
                pred_dis = pred_dis.unsqueeze(1)
            if pred_dis.shape[1] != 1:
                pred_dis = pred_dis.view(-1, 1)

            if isinstance(pred_mag, tuple):
                pred_mag = pred_mag[0] if len(pred_mag) == 1 else torch.cat(pred_mag, dim=1)
            if pred_mag.dim() == 1:
                pred_mag = pred_mag.unsqueeze(1)
            if pred_mag.shape[1] != 1:
                pred_mag = pred_mag.view(-1, 1)

            # 确保pred_azi是[B, 2]形状
            if pred_azi.shape[1] != 2:
                if pred_azi.shape[1] == 1:
                    logging.error(f"方位角输出形状不正确: {pred_azi.shape}")
                    continue
                else:
                    pred_azi = pred_azi[:, :2]

            # 计算损失
            loss_azi = criterion_azi(pred_azi, y_azi)
            loss_dis = criterion_dis(pred_dis, y_dis)
            loss_mag = criterion_mag(pred_mag, y_mag)
            total_task_loss = loss_azi + loss_dis + loss_mag

            # 累计损失
            total_loss += total_task_loss.item()
            total_azi_loss += loss_azi.item()
            total_dis_loss += loss_dis.item()
            total_mag_loss += loss_mag.item()

            # 收集预测和标签用于评估
            # 方位角：转换回角度
            pred_azi_cpu = pred_azi.cpu().numpy()
            true_azi_cpu = y_azi.cpu().numpy()
            pred_azi_deg = sincos_to_deg(pred_azi_cpu[:, 0], pred_azi_cpu[:, 1])
            true_azi_deg = sincos_to_deg(true_azi_cpu[:, 0], true_azi_cpu[:, 1])
            azi_preds.extend(pred_azi_deg.tolist())
            azi_labels.extend(true_azi_deg.tolist())

            # 震中距：exp转换
            pred_dis_km = np.exp(pred_dis.cpu().numpy().squeeze())
            true_dis_km = np.exp(y_dis.cpu().numpy().squeeze())
            dis_preds.extend(pred_dis_km.tolist())
            dis_labels.extend(true_dis_km.tolist())

            # 震级：直接使用
            mag_preds.extend(pred_mag.cpu().numpy().squeeze().tolist())
            mag_labels.extend(y_mag.cpu().numpy().squeeze().tolist())

    avg_total_loss = total_loss / len(loader)
    avg_azi_loss = total_azi_loss / len(loader)
    avg_dis_loss = total_dis_loss / len(loader)
    avg_mag_loss = total_mag_loss / len(loader)

    # 计算整体评估指标
    avg_azi_err = baz_angle_error(np.column_stack([np.sin(np.deg2rad(azi_preds)), np.cos(np.deg2rad(azi_preds))]),
                                  np.column_stack([np.sin(np.deg2rad(azi_labels)), np.cos(np.deg2rad(azi_labels))]))
    avg_dis_mae = mean_absolute_error(dis_labels, dis_preds)
    avg_mag_mae = mean_absolute_error(mag_labels, mag_preds)

    logging.info(f"{mode}结束 - "
                 f"平均总损失: {avg_total_loss:.6f} | "
                 f"方位角损失: {avg_azi_loss:.6f} | "
                 f"震中距损失: {avg_dis_loss:.6f} | "
                 f"震级损失: {avg_mag_loss:.6f} | "
                 f"方位角误差: {avg_azi_err:.3f}° | "
                 f"震中距MAE: {avg_dis_mae:.6f} | "
                 f"震级MAE: {avg_mag_mae:.6f}")

    return avg_total_loss, avg_azi_loss, avg_dis_loss, avg_mag_loss, avg_azi_err, avg_dis_mae, avg_mag_mae


def test_model(model, loader):
    model.eval()
    azi_preds, azi_labels = [], []
    dis_preds, dis_labels = [], []
    mag_preds, mag_labels = [], []
    keys = []

    with torch.no_grad():
        for wave, spec, feat, y_azi, y_dis, y_mag, k in tqdm(loader, desc="测试中"):
            wave, spec, feat = wave.to(device), spec.to(device), feat.to(device)
            y_azi, y_dis, y_mag = y_azi.to(device), y_dis.to(device), y_mag.to(device)

            # 前向传播
            pred_azi, pred_mag, pred_dis = model(wave, spec, feat)

            # 处理方位角输出
            if isinstance(pred_azi, tuple):
                pred_azi = torch.cat([pred_azi[0], pred_azi[1]], dim=1)  # [B, 2]

            # 处理震中距和震级输出
            if isinstance(pred_dis, tuple):
                pred_dis = pred_dis[0] if len(pred_dis) == 1 else torch.cat(pred_dis, dim=1)
            if pred_dis.dim() == 1:
                pred_dis = pred_dis.unsqueeze(1)
            if pred_dis.shape[1] != 1:
                pred_dis = pred_dis.view(-1, 1)

            if isinstance(pred_mag, tuple):
                pred_mag = pred_mag[0] if len(pred_mag) == 1 else torch.cat(pred_mag, dim=1)
            if pred_mag.dim() == 1:
                pred_mag = pred_mag.unsqueeze(1)
            if pred_mag.shape[1] != 1:
                pred_mag = pred_mag.view(-1, 1)

            # 确保pred_azi是[B, 2]形状
            if pred_azi.shape[1] != 2:
                if pred_azi.shape[1] == 1:
                    logging.error(f"方位角输出形状不正确: {pred_azi.shape}")
                    continue
                else:
                    pred_azi = pred_azi[:, :2]

            # 收集预测和标签
            # 方位角：转换回角度
            pred_azi_cpu = pred_azi.cpu().numpy()
            true_azi_cpu = y_azi.cpu().numpy()
            pred_azi_deg = sincos_to_deg(pred_azi_cpu[:, 0], pred_azi_cpu[:, 1])
            true_azi_deg = sincos_to_deg(true_azi_cpu[:, 0], true_azi_cpu[:, 1])
            azi_preds.extend(pred_azi_deg.tolist())
            azi_labels.extend(true_azi_deg.tolist())

            # 震中距：exp转换
            pred_dis_km = np.exp(pred_dis.cpu().numpy().squeeze())
            true_dis_km = np.exp(y_dis.cpu().numpy().squeeze())
            dis_preds.extend(pred_dis_km.tolist())
            dis_labels.extend(true_dis_km.tolist())

            # 震级：直接使用
            mag_preds.extend(pred_mag.cpu().numpy().squeeze().tolist())
            mag_labels.extend(y_mag.cpu().numpy().squeeze().tolist())

            keys.extend(k)

    # 计算各项指标
    azi_err = baz_angle_error(np.column_stack([np.sin(np.deg2rad(azi_preds)), np.cos(np.deg2rad(azi_preds))]),
                              np.column_stack([np.sin(np.deg2rad(azi_labels)), np.cos(np.deg2rad(azi_labels))]))
    dis_mae = mean_absolute_error(dis_labels, dis_preds)
    mag_mae = mean_absolute_error(mag_labels, mag_preds)

    dis_mse = mean_squared_error(dis_labels, dis_preds)
    mag_mse = mean_squared_error(mag_labels, mag_preds)

    dis_r2 = r2_score(dis_labels, dis_preds)
    mag_r2 = r2_score(mag_labels, mag_preds)

    logging.info(f"测试结果汇总:")
    logging.info(f" 方位角误差: {azi_err:.3f}°")
    logging.info(f" 震中距 MAE: {dis_mae:.6f}")
    logging.info(f" 震中距 MSE: {dis_mse:.6f}")
    logging.info(f" 震中距 R²: {dis_r2:.6f}")
    logging.info(f" 震级 MAE: {mag_mae:.6f}")
    logging.info(f" 震级 MSE: {mag_mse:.6f}")
    logging.info(f" 震级 R²: {mag_r2:.6f}")

    return (np.array(azi_labels), np.array(azi_preds)), \
        (np.array(dis_labels), np.array(dis_preds)), \
        (np.array(mag_labels), np.array(mag_preds)), \
        keys, \
        (azi_err, dis_mae, dis_mse, dis_r2, mag_mae, mag_mse, mag_r2)


def plot_baz_polar(true_deg, pred_deg, save_path):
    """ Rose / polar diagram of BAZ error """
    error = pred_deg - true_deg
    error = (error + 180) % 360 - 180  # [-180, 180]
    error_rad = np.deg2rad(error)
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, projection='polar')
    bins = np.deg2rad(np.linspace(-180, 180, 37))  # 10° bins
    ax.hist(error_rad, bins=bins, density=True, alpha=0.75)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title("Back-Azimuth Error Rose Diagram", pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def set_deterministic(seed=42):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# -------------------------- 5. 主函数 --------------------------
def main(model_name="EEW_LMBART", h5_path=None):
    set_deterministic(42)  # <-- 添加这一行
    logging.info("开始加载三输入 H5 数据集（多任务）...")
    dataset = SeismicDataset(
        h5_file_path=h5_path,
        target_length=TARGET_LENGTH,
        spec_length=SPEC_LENGTH
    )
    total = len(dataset)
    if total == 0:
        logging.error("数据集无有效样本，终止训练！")
        return

    train_size = int(total * TRAIN_SPLIT)
    val_size = int(total * VAL_SPLIT)
    test_size = total - train_size - val_size
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    logging.info(
        f"数据集拆分完成: "
        f"训练集={len(train_ds)} | 验证集={len(val_ds)} | 测试集={len(test_ds)}"
    )
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False)

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"未知模型名称: {model_name}。可选: {list(MODEL_REGISTRY.keys())}")
    model_class = MODEL_REGISTRY[model_name]
    model = model_class().to(device)
    logging.info(f"模型结构:\n{model}")

    # 定义各任务的损失函数
    criterion_azi = baz_loss
    criterion_dis = WeightedLogMSELoss()
    criterion_mag = nn.MSELoss()

    # -------------------------- 参数分组 --------------------------
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

    # -------------------------- Optimizer with discriminative LR --------------------------
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

    # -------------------------- Scheduler (HF-style step-based warmup) --------------------------
    total_steps = len(train_loader) * EPOCHS
    num_warmup_steps = int(0.05 * total_steps)  # 5% warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps,
        num_cycles=0.5,  # 标准余弦退火（降到0）
    )

    # 最佳模型保存相关参数 - 使用综合指标
    best_metrics = {
        "val_combined_score": float("inf"),  # 综合评分（越小越好）
        "val_total_loss": float("inf"),
        "val_azi_err": float("inf"),
        "val_dis_mae": float("inf"),
        "val_mag_mae": float("inf"),
        "epoch": 0,
        "model_state": None
    }
    ckpt_path = os.path.join(
        CHECKPOINT_DIR, f"best_multitask_model_{TRAIN_START_TIME}.pth"
    )
    early_stop_counter = 0

    # 记录训练过程指标
    train_losses, val_losses = [], []
    train_azi_losses, val_azi_losses = [], []
    train_dis_losses, val_dis_losses = [], []
    train_mag_losses, val_mag_losses = [], []
    train_azi_errs, val_azi_errs = [], []
    train_dis_maes, val_dis_maes = [], []
    train_mag_maes, val_mag_maes = [], []

    logging.info("开始多任务训练...")
    for epoch in range(1, EPOCHS + 1):
        logging.info(f"\n========== Epoch {epoch}/{EPOCHS} ==========")
        logging.info(
            f"当前学习率: Backbone={optimizer.param_groups[0]['lr']:.6e}, Head={optimizer.param_groups[1]['lr']:.6e}")

        train_total_loss, train_azi_loss, train_dis_loss, train_mag_loss, train_azi_err, train_dis_mae, train_mag_mae = train_one_epoch(
            model, train_loader, criterion_azi, criterion_dis, criterion_mag,
            optimizer, scheduler, backbone_params, head_params, epoch
        )
        val_total_loss, val_azi_loss, val_dis_loss, val_mag_loss, val_azi_err, val_dis_mae, val_mag_mae = evaluate(
            model, val_loader, criterion_azi, criterion_dis, criterion_mag
        )

        # 记录指标
        train_losses.append(train_total_loss)
        val_losses.append(val_total_loss)
        train_azi_losses.append(train_azi_loss)
        val_azi_losses.append(val_azi_loss)
        train_dis_losses.append(train_dis_loss)
        val_dis_losses.append(val_dis_loss)
        train_mag_losses.append(train_mag_loss)
        val_mag_losses.append(val_mag_loss)
        train_azi_errs.append(train_azi_err)
        val_azi_errs.append(val_azi_err)
        train_dis_maes.append(train_dis_mae)
        val_dis_maes.append(val_dis_mae)
        train_mag_maes.append(train_mag_mae)
        val_mag_maes.append(val_mag_mae)

        logging.info(
            f"Train Total Loss: {train_total_loss:.6f} | "
            f"Azi Err: {train_azi_err:.3f}° | Dis MAE: {train_dis_mae:.6f} | Mag MAE: {train_mag_mae:.6f} | "
            f"Val Total Loss: {val_total_loss:.6f} | "
            f"Azi Err: {val_azi_err:.3f}° | Dis MAE: {val_dis_mae:.6f} | Mag MAE: {val_mag_mae:.6f}"
        )

        # 综合评分：各任务指标的加权平均（权重均为1）
        combined_score = val_azi_err + val_dis_mae + val_mag_mae

        # ---------- Early Stop ----------
        if combined_score < best_metrics["val_combined_score"]:
            best_metrics.update({
                "val_combined_score": combined_score,
                "val_total_loss": val_total_loss,
                "val_azi_err": val_azi_err,
                "val_dis_mae": val_dis_mae,
                "val_mag_mae": val_mag_mae,
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            })
            torch.save(best_metrics, ckpt_path)
            early_stop_counter = 0
            logging.info(f"更新最佳模型 ✓ (Combined Score: {combined_score:.6f})")
        else:
            early_stop_counter += 1
            logging.info(f"未提升 ({early_stop_counter}/{EARLY_STOP_PATIENCE})")

        if early_stop_counter >= EARLY_STOP_PATIENCE:
            logging.info(f"早停触发，最佳 Epoch={best_metrics['epoch']}")
            break

    # =================== 测试最佳模型 ===================
    logging.info("\n开始测试最佳多任务模型...")
    checkpoint = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    (azi_true, azi_pred), (dis_true, dis_pred), (mag_true, mag_pred), keys, metrics = test_model(model, test_loader)

    azi_err, dis_mae, dis_mse, dis_r2, mag_mae, mag_mse, mag_r2 = metrics

    logging.info(f"最终测试结果:")
    logging.info(f" 方位角误差: {azi_err:.3f}°")
    logging.info(f" 震中距 MAE: {dis_mae:.6f} | MSE: {dis_mse:.6f} | R²: {dis_r2:.6f}")
    logging.info(f" 震级 MAE: {mag_mae:.6f} | MSE: {mag_mse:.6f} | R²: {mag_r2:.6f}")

    # 保存测试结果表格
    save_test_table(keys, azi_true, azi_pred, dis_true, dis_pred, mag_true, mag_pred, CHECKPOINT_DIR, TRAIN_START_TIME)

    # ---------- 绘制结果图 ----------
    plt.figure(figsize=(15, 20))

    # 损失曲线
    plt.subplot(4, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='训练总损失')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='验证总损失')
    plt.title('训练与验证总损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('损失值')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 2, 2)
    plt.plot(range(1, len(train_azi_losses) + 1), train_azi_losses, label='训练方位角损失')
    plt.plot(range(1, len(val_azi_losses) + 1), val_azi_losses, label='验证方位角损失')
    plt.title('方位角损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('损失值')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 2, 3)
    plt.plot(range(1, len(train_dis_losses) + 1), train_dis_losses, label='训练震中距损失')
    plt.plot(range(1, len(val_dis_losses) + 1), val_dis_losses, label='验证震中距损失')
    plt.title('震中距损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('损失值')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 2, 4)
    plt.plot(range(1, len(train_mag_losses) + 1), train_mag_losses, label='训练震级损失')
    plt.plot(range(1, len(val_mag_losses) + 1), val_mag_losses, label='验证震级损失')
    plt.title('震级损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('损失值')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 评估指标曲线
    plt.subplot(4, 2, 5)
    plt.plot(range(1, len(train_azi_errs) + 1), train_azi_errs, label='训练方位角误差')
    plt.plot(range(1, len(val_azi_errs) + 1), val_azi_errs, label='验证方位角误差')
    plt.title('方位角误差曲线')
    plt.xlabel('Epoch')
    plt.ylabel('角度误差 (°)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 2, 6)
    plt.plot(range(1, len(train_dis_maes) + 1), train_dis_maes, label='训练震中距MAE')
    plt.plot(range(1, len(val_dis_maes) + 1), val_dis_maes, label='验证震中距MAE')
    plt.title('震中距MAE曲线')
    plt.xlabel('Epoch')
    plt.ylabel('MAE (km)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 2, 7)
    plt.plot(range(1, len(train_mag_maes) + 1), train_mag_maes, label='训练震级MAE')
    plt.plot(range(1, len(val_mag_maes) + 1), val_mag_maes, label='验证震级MAE')
    plt.title('震级MAE曲线')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 散点图
    plt.subplot(4, 2, 8)
    plt.scatter(azi_true, azi_pred, s=30, alpha=0.6)
    plt.plot([0, 360], [0, 360], 'r--', alpha=0.8)
    plt.title(f"真实 vs 预测方位角 (Azi Err={azi_err:.3f}°)")
    plt.xlabel("真实方位角(°)"), plt.ylabel("预测方位角(°)")

    plt.tight_layout()
    loss_plot_path = os.path.join(CHECKPOINT_DIR, "multitask_training_curves.png")
    plt.savefig(loss_plot_path, dpi=300)
    logging.info(f"多任务训练曲线图已保存：{loss_plot_path}")

    # 震中距散点图
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(dis_true, dis_pred, s=30, alpha=0.6)
    plt.plot([min(dis_true), max(dis_true)], [min(dis_true), max(dis_true)], 'r--', alpha=0.8)
    plt.title(f"真实 vs 预测震中距 (MAE={dis_mae:.3f})")
    plt.xlabel("真实震中距(km)"), plt.ylabel("预测震中距(km)")

    # 震级散点图
    plt.subplot(1, 2, 2)
    plt.scatter(mag_true, mag_pred, s=30, alpha=0.6)
    plt.plot([min(mag_true), max(mag_true)], [min(mag_true), max(mag_true)], 'r--', alpha=0.8)
    plt.title(f"真实 vs 预测震级 (MAE={mag_mae:.3f})")
    plt.xlabel("真实震级"), plt.ylabel("预测震级")

    plt.tight_layout()
    scatter_plot_path = os.path.join(CHECKPOINT_DIR, "multitask_scatter_plots.png")
    plt.savefig(scatter_plot_path, dpi=300)
    logging.info(f"多任务散点图已保存：{scatter_plot_path}")

    # 方位角玫瑰图
    rose_path = os.path.join(CHECKPOINT_DIR, "baz_error_rose.png")
    plot_baz_polar(azi_true, azi_pred, rose_path)
    logging.info(f"BAZ Rose Diagram 已保存: {rose_path}")

    # 保存详细结果CSV
    results_df = pd.DataFrame({
        "event_id": keys,
        "true_azi_deg": azi_true,
        "pred_azi_deg": azi_pred,
        "true_dis_km": dis_true,
        "pred_dis_km": dis_pred,
        "true_mag": mag_true,
        "pred_mag": mag_pred
    })
    csv_path = os.path.join(CHECKPOINT_DIR, f"multitask_test_results_{TRAIN_START_TIME}.csv")
    results_df.to_csv(csv_path, index=False)
    logging.info(f"详细测试结果已保存到 CSV: {csv_path}")

    logging.info("多任务训练与测试全部完成！")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Multi-task Seismic Parameter Estimation Model")
    parser.add_argument("--model_name", type=str, default="EEW_LMBART",
                        choices=list(MODEL_REGISTRY.keys()),
                        help=f"Model to train. Options: {list(MODEL_REGISTRY.keys())}")
    parser.add_argument("--h5_path", type=str, default=r"D:\zjn\three\dataset\JKnet_400_with_FS.h5",
                        help="Path to the H5 dataset file")
    args = parser.parse_args()
    main(
        model_name=args.model_name,
        h5_path=args.h5_path,
    )
