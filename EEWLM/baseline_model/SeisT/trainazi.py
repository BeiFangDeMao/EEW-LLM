#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练 SeisT-M 方位角估计模型（使用 seist-m-baz）
输出为0-360度角度，损失函数使用周期性MSE损失
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
import h5py
from tqdm import tqdm
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv
import argparse
import math
from torch.optim.lr_scheduler import LambdaLR
from seist import seist_m_baz  # 修改：导入方位角估计模型

# 调整字体设置
plt.rcParams["font.family"] = ["STIXGeneral", "DejaVu Sans", "SimHei", "Microsoft YaHei"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams['axes.unicode_minus'] = False

# -------------------------- 1. 基础配置 --------------------------
TRAIN_START_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")

# 数据与训练参数
H5_FILE_PATH = r"D:\EEWLMDATASET\JKnet_300.h5"  # 替换为你的H5文件路径
TARGET_LENGTH = 300
TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT = 0.80, 0.15, 0.05
BATCH_SIZE = 500
EPOCHS = 200
WEIGHT_DECAY = 1e-5
EARLY_STOP_PATIENCE = 30
EARLY_STOP_EPS = 1e-6

# SeisT训练策略参数
MIN_LR = 8e-5
MAX_LR = 1e-3
DROPPATH_RATE = 0.2
MODEL_SIZE = 'M'

# 路径配置
BASE_DIR = os.path.abspath("")
CHECKPOINT_DIR = os.path.join(BASE_DIR, f"checkpoint_{TRAIN_START_TIME}")
LOG_DIR = os.path.join(BASE_DIR, f"log_{TRAIN_START_TIME}")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------- 2. 增强版日志配置 --------------------------
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
logging.info(f"方位角估计训练启动时间：{TRAIN_START_TIME}")
logging.info(f"使用设备：{device}")
logging.info(f"训练参数：")
logging.info(f" H5数据路径：{H5_FILE_PATH}")
logging.info(f" 数据尺寸：波形(3,{TARGET_LENGTH})")
logging.info(f" 数据拆分：训练{TRAIN_SPLIT * 100}% | 验证{VAL_SPLIT * 100}% | 测试{TEST_SPLIT * 100}%")
logging.info(f" 批次大小：{BATCH_SIZE} | 总轮次：{EPOCHS}")
logging.info(f" 学习率范围：{MIN_LR:.1e} ~ {MAX_LR:.1e} | DropPath率：{DROPPATH_RATE}")
logging.info(f" 早停阈值：{EARLY_STOP_PATIENCE}轮 | 最小改进：{EARLY_STOP_EPS}")
logging.info("=" * 50)


# -------------------------- 3. 周期性MSE损失函数 --------------------------
class AngularCosineLoss(nn.Module):
    def forward(self, pred_deg, target_deg):
        # 转为弧度
        pred = torch.deg2rad(pred_deg)
        target = torch.deg2rad(target_deg)

        # cosine loss
        loss = 1.0 - torch.cos(pred - target)
        return loss.mean()

# -------------------------- 4. 方位角角度误差计算 --------------------------
def calculate_angle_error(pred_angle, true_angle):
    """
    计算角度误差（考虑周期性）
    Args:
        pred_angle: 预测角度（度）
        true_angle: 真实角度（度）
    Returns:
        error: 最小角度误差（度）
    """
    # 计算原始差值
    diff = np.abs(pred_angle - true_angle)
    # 考虑周期性，取最小差值
    diff = np.minimum(diff, 360 - diff)
    return diff


# -------------------------- 5. 单输入H5数据集（方位角估计） --------------------------
class SeismicWaveformDataset(Dataset):
    def __init__(self, h5_file_path, target_length=300, max_samples=None):
        """
        Args:
            h5_file_path: H5数据集路径
            target_length: 波形序列长度
            max_samples: 最大加载样本数（可选）
        """
        self.data_list = []
        self.target_length = target_length
        with h5py.File(h5_file_path, "r") as f:
            keys = list(f.keys())
            if max_samples:
                keys = keys[:max_samples]
            for key in tqdm(keys, desc="加载H5数据"):
                try:
                    group = f[key]
                    waveform = group[:].astype(np.float32).T  # (3, L)
                    if waveform.shape[0] != 3:
                        continue

                    # 修改：读取方位角标签（azi属性）
                    azimuth = float(group.attrs["azi"])
                    # 检查方位角是否有效（0-360度）
                    if np.isnan(azimuth) or azimuth < 0 or azimuth > 360:
                        continue

                    # Padding / 截断
                    if waveform.shape[1] < target_length:
                        pad = target_length - waveform.shape[1]
                        waveform = np.pad(waveform, ((0, 0), (0, pad)))
                    else:
                        waveform = waveform[:, :target_length]

                    self.data_list.append((waveform, azimuth, key))
                except Exception as e:
                    logging.warning(f"跳过样本 {key}: {e}")
        logging.info(f" 加载完成，有效样本 {len(self.data_list)} 个")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        x, y, key = self.data_list[idx]
        return torch.from_numpy(x).to(torch.float32), torch.tensor(y, dtype=torch.float32), key


# -------------------------- 6. 三角循环学习率调度器 --------------------------
def create_cyclic_lr_scheduler(optimizer, min_lr, max_lr, total_epochs, step_size_ratio=0.125):
    """创建三角循环学习率调度器"""
    step_size = int(total_epochs * step_size_ratio)

    def cyclic_lr(epoch):
        cycle = math.floor(1 + epoch / (2 * step_size))
        x = abs(epoch / step_size - 2 * cycle + 1)
        lr = min_lr + (max_lr - min_lr) * max(0, (1 - x))
        return lr / max_lr  # LambdaLR期望返回缩放因子

    return LambdaLR(optimizer, lr_lambda=cyclic_lr)


# -------------------------- 7. 自定义权重初始化 --------------------------
def init_seist_weights(model):
    """SeisT风格的自定义权重初始化"""

    def init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    model.apply(init_weights)
    logging.info("模型权重初始化完成（SeisT风格）")


# -------------------------- 8. 训练/验证/测试函数 --------------------------
def train_one_epoch(model, loader, criterion, optimizer, scheduler, epoch):
    model.train()
    total_loss, preds, labels = 0, [], []

    for i, (wave, y, _) in enumerate(tqdm(loader, desc=f"训练 Epoch {epoch}")):
        wave, y = wave.to(device), y.to(device).unsqueeze(1)

        optimizer.zero_grad()
        out = model(wave)  # 输出角度（0-360度）
        loss = criterion(out, y)
        loss.backward()

        # 全局梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()
        total_loss += loss.item()

        # 收集预测结果用于计算指标
        preds.extend(out.detach().cpu().numpy().squeeze().tolist())
        labels.extend(y.cpu().numpy().squeeze().tolist())

        # 每50个batch记录一次
        if (i + 1) % 50 == 0:
            # 计算角度误差
            batch_errors = calculate_angle_error(
                np.array(preds[-BATCH_SIZE:]),
                np.array(labels[-BATCH_SIZE:])
            )
            batch_mae = np.mean(batch_errors)
            logging.debug(
                f"Epoch {epoch} 批次 {i + 1}/{len(loader)} - 批次损失: {loss.item():.6f} | 批次角度MAE: {batch_mae:.6f}")

    epoch_loss = total_loss / len(loader)

    # 计算整个epoch的角度误差
    errors = calculate_angle_error(np.array(preds), np.array(labels))
    epoch_angle_mae = np.mean(errors)

    logging.info(f"Epoch {epoch} 训练结束 - 平均损失: {epoch_loss:.6f} | 平均角度误差: {epoch_angle_mae:.6f}")

    return epoch_loss, epoch_angle_mae


def evaluate(model, loader, criterion, mode="验证"):
    model.eval()
    total_loss, preds, labels = 0, [], []

    with torch.no_grad():
        for wave, y, _ in tqdm(loader, desc=mode):
            wave, y = wave.to(device), y.to(device).unsqueeze(1)
            out = model(wave)
            loss = criterion(out, y)
            total_loss += loss.item()

            # 直接使用模型输出的角度
            pred_angle = out.cpu().numpy().squeeze()
            true_angle = y.cpu().numpy().squeeze()

            preds.extend(pred_angle.tolist())
            labels.extend(true_angle.tolist())

    avg_loss = total_loss / len(loader)

    # 计算角度误差
    errors = calculate_angle_error(np.array(preds), np.array(labels))
    avg_angle_error = np.mean(errors)

    logging.info(f"{mode}结束 - 平均损失: {avg_loss:.6f} | 平均角度误差: {avg_angle_error:.6f}")

    return avg_loss, avg_angle_error


def test_model(model, loader):
    model.eval()
    preds, trues, keys = [], [], []

    with torch.no_grad():
        for wave, y, k in tqdm(loader, desc="测试中"):
            wave, y = wave.to(device), y.to(device).unsqueeze(1)
            out = model(wave)

            # 直接使用模型输出的角度
            pred_angle = out.cpu().numpy().squeeze()
            true_angle = y.cpu().numpy().squeeze()

            preds.extend(pred_angle.tolist())
            trues.extend(true_angle.tolist())
            keys.extend(k)

    # 计算角度误差
    errors = calculate_angle_error(np.array(preds), np.array(trues))
    mae = np.mean(errors)

    # 对于角度数据，MSE可能不太适用，但我们仍然计算
    mse = mean_squared_error(trues, preds)

    # 对于角度数据，R²可能不太适用，但我们仍然计算
    r2 = r2_score(trues, preds)

    # 计算角度误差的标准差
    angle_error_std = np.std(errors)

    logging.info("=" * 40)
    logging.info(f"测试结果汇总:")
    logging.info(f" 平均角度误差(MAE): {mae:.4f}°")
    logging.info(f" 角度误差标准差: {angle_error_std:.4f}°")
    logging.info(f" 最大角度误差: {np.max(errors):.4f}°")
    logging.info(f" 最小角度误差: {np.min(errors):.4f}°")
    logging.info(f" MSE: {mse:.6f}")
    logging.info(f" R²: {r2:.6f}")
    logging.info("=" * 40)

    return np.array(trues), np.array(preds), keys, (mae, mse, r2, angle_error_std)


def save_test_table(all_keys, all_trues, all_preds, save_dir, timestamp):
    """生成测试结果表格 (事件ID, 真实方位角, 预测方位角, 角度误差)"""
    csv_path = os.path.join(save_dir, f"test_table_{timestamp}.csv")

    # 计算角度误差
    errors = calculate_angle_error(all_preds, all_trues)

    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(["事件ID", "真实方位角(°)", "预测方位角(°)", "角度误差(°)"])
        for k, t, p, err in zip(all_keys, all_trues, all_preds, errors):
            writer.writerow([k, round(float(t), 2), round(float(p), 2), round(float(err), 2)])
    logging.info(f"测试表格已保存至：{csv_path}")
    return csv_path


def test_only(checkpoint_path, h5_path):
    """使用现有最佳权重和原始 seed=42 测试集，不重新训练。"""
    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    dataset = SeismicWaveformDataset(
        h5_file_path=h5_path or H5_FILE_PATH, target_length=TARGET_LENGTH)
    total = len(dataset)
    train_size, val_size = int(total * TRAIN_SPLIT), int(total * VAL_SPLIT)
    _, _, test_ds = random_split(
        dataset, [train_size, val_size, total - train_size - val_size],
        generator=torch.Generator().manual_seed(42))
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=True)
    model = seist_m_baz().to(device)
    # 仅加载本实验生成的可信检查点；其中含 NumPy 标量和优化器状态。
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    logging.info("仅测试：最佳模型 Epoch %s，测试样本 %d",
                 checkpoint["epoch"], len(test_ds))
    trues, preds, keys, _ = test_model(model, test_loader)
    timestamp = os.path.splitext(os.path.basename(checkpoint_path))[0].removeprefix("best_model_")
    return save_test_table(keys, trues, preds,
                           os.path.dirname(checkpoint_path), timestamp)


# -------------------------- 9. 主函数 --------------------------
def main(model_name="SeisT_M_BAZ", h5_path=None, use_drop_path=True, seed=None):
    """
    主训练函数

    Args:
        model_name: 模型名称
        h5_path: H5数据文件路径
        use_drop_path: 是否使用DropPath正则化
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logging.info("固定训练随机种子: %d；数据划分种子保持 42", seed)
    logging.info("开始加载单输入H5数据集...")

    # 加载数据集
    dataset = SeismicWaveformDataset(
        h5_file_path=h5_path or H5_FILE_PATH,
        target_length=TARGET_LENGTH
    )

    total = len(dataset)
    if total == 0:
        logging.error("数据集无有效样本，终止训练！")
        return

    # 数据集拆分
    train_size, val_size = int(total * TRAIN_SPLIT), int(total * VAL_SPLIT)
    test_size = total - train_size - val_size

    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    train_fraction = float(os.environ.get("EEW_TRAIN_FRACTION", "1"))
    if not 0 < train_fraction <= 1:
        raise ValueError("EEW_TRAIN_FRACTION 必须在 (0, 1] 内")
    if train_fraction < 1:
        indices = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(42))
        train_ds = torch.utils.data.Subset(train_ds, indices[:max(1, int(len(train_ds) * train_fraction))].tolist())
    logging.info(f"训练集使用比例: {train_fraction:.0%} | 实际训练样本: {len(train_ds)}")

    logging.info(f"数据集拆分完成: 训练集{len(train_ds)} | 验证集{len(val_ds)} | 测试集{len(test_ds)}")

    # 创建数据加载器
    shuffle_generator = torch.Generator().manual_seed(seed) if seed is not None else None
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=0,
                              pin_memory=True, generator=shuffle_generator)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # 动态导入模型
    model = seist_m_baz()  # 修改：使用方位角估计模型
    model = model.to(device)

    # 模型权重初始化（SeisT风格）
    init_seist_weights(model)
    logging.info(f"模型结构:\n{model}")

    # 损失函数 - 使用周期性MSE损失
    criterion = AngularCosineLoss()


    # 优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=MAX_LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # 学习率调度器（三角循环学习率）
    scheduler = create_cyclic_lr_scheduler(
        optimizer, MIN_LR, MAX_LR, EPOCHS, step_size_ratio=0.125
    )

    # 最佳模型保存相关参数
    best_metrics = {
        "val_angle_error": float("inf"),
        "val_loss": float("inf"),
        "epoch": 0,
        "model_state": None,
        "optimizer_state": None
    }
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"best_model_{TRAIN_START_TIME}.pth")
    early_stop_counter = 0

    # 记录训练过程指标
    train_losses = []
    val_losses = []
    train_angle_errors = []
    val_angle_errors = []
    learning_rates = []

    # 开始训练
    logging.info("开始训练...")

    for epoch in range(1, EPOCHS + 1):
        logging.info(f"\n{'=' * 20} Epoch {epoch}/{EPOCHS} {'=' * 20}")

        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        logging.info(f"当前学习率: {current_lr:.6e}")

        # 训练一个epoch
        train_loss, train_angle_error = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, epoch
        )

        # 验证
        val_loss, val_angle_error = evaluate(model, val_loader, criterion)

        # 更新学习率调度器
        scheduler.step()

        # 记录指标
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_angle_errors.append(train_angle_error)
        val_angle_errors.append(val_angle_error)

        logging.info(f"训练集 - 损失: {train_loss:.6f} | 角度误差: {train_angle_error:.4f}°")
        logging.info(f"验证集 - 损失: {val_loss:.6f} | 角度误差: {val_angle_error:.4f}°")

        # 最佳模型判断与保存（基于角度误差）
        if val_angle_error < best_metrics["val_angle_error"] - EARLY_STOP_EPS:
            best_metrics.update({
                "val_angle_error": val_angle_error,
                "val_loss": val_loss,
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            })
            torch.save(best_metrics, ckpt_path)
            logging.info(
                f"更新最佳模型 (Epoch {epoch}) - 最佳角度误差: {val_angle_error:.4f}° | 最佳损失: {val_loss:.6f}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            logging.info(f"未更新最佳模型 ({early_stop_counter}/{EARLY_STOP_PATIENCE})")

        # 早停检查
        if early_stop_counter >= EARLY_STOP_PATIENCE:
            logging.info(f"早停触发！最佳模型出现在 Epoch {best_metrics['epoch']}")
            break

    # 测试最佳模型
    logging.info("\n" + "=" * 50)
    logging.info("开始测试最佳模型...")

    best_checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state"])
    logging.info(f"加载最佳模型 (Epoch {best_checkpoint['epoch']}): 角度误差={best_checkpoint['val_angle_error']:.4f}°")

    trues, preds, keys, metrics = test_model(model, test_loader)
    save_test_table(keys, trues, preds, CHECKPOINT_DIR, TRAIN_START_TIME)

    # 绘制训练过程图
    plot_training_results(
        train_losses, val_losses, train_angle_errors, val_angle_errors,
        learning_rates, trues, preds, metrics, CHECKPOINT_DIR
    )

    logging.info("所有训练流程完成！")
    return model, best_metrics


def plot_training_results(train_losses, val_losses, train_angle_errors, val_angle_errors,
                          learning_rates, trues, preds, metrics, save_dir):
    """绘制训练结果图表"""
    mae, mse, r2, angle_error_std = metrics

    plt.figure(figsize=(15, 12))

    # 1. 损失曲线
    plt.subplot(3, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='训练损失', linewidth=2)
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='验证损失', linewidth=2)
    plt.title('训练与验证损失曲线', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('损失值', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # 2. 角度误差曲线
    plt.subplot(3, 2, 2)
    plt.plot(range(1, len(train_angle_errors) + 1), train_angle_errors, label='训练角度误差', linewidth=2)
    plt.plot(range(1, len(val_angle_errors) + 1), val_angle_errors, label='验证角度误差', linewidth=2)
    plt.title('训练与验证角度误差曲线', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('角度误差 (°)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # 3. 学习率曲线
    plt.subplot(3, 2, 3)
    plt.plot(range(1, len(learning_rates) + 1), learning_rates, linewidth=2, color='purple')
    plt.title('学习率变化曲线 (Cyclic LR)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('学习率', fontsize=12)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    # 4. 预测 vs 真实散点图
    plt.subplot(3, 2, 4)
    plt.scatter(trues, preds, s=30, alpha=0.6, c='blue')
    plt.plot([0, 360], [0, 360], 'r--', linewidth=2)
    plt.title(f"真实 vs 预测方位角 (角度误差={mae:.2f}°)", fontsize=14, fontweight='bold')
    plt.xlabel("真实方位角 (°)", fontsize=12)
    plt.ylabel("预测方位角 (°)", fontsize=12)
    plt.xlim(0, 360)
    plt.ylim(0, 360)
    plt.grid(True, alpha=0.3)

    # 5. 角度误差分布图
    plt.subplot(3, 2, 5)
    errors = calculate_angle_error(preds, trues)
    plt.hist(errors, bins=30, alpha=0.7, color='green', edgecolor='black')
    plt.axvline(x=mae, color='r', linestyle='--', linewidth=2, label=f'均值={mae:.2f}°')
    plt.axvline(x=mae + angle_error_std, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label=f'±1σ')
    plt.axvline(x=mae - angle_error_std, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
    plt.title(f'角度误差分布 (标准差={angle_error_std:.2f}°)', fontsize=14, fontweight='bold')
    plt.xlabel('角度误差 (°)', fontsize=12)
    plt.ylabel('频数', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # 6. 方位角分布玫瑰图
    plt.subplot(3, 2, 6, projection='polar')

    # 计算方位角分布
    bins = np.linspace(0, 2 * np.pi, 37)  # 36个10度的区间
    true_counts, _ = np.histogram(np.deg2rad(trues), bins=bins)
    pred_counts, _ = np.histogram(np.deg2rad(preds), bins=bins)

    # 计算角度位置（每个区间的中心）
    angles = (bins[:-1] + bins[1:]) / 2

    # 绘制玫瑰图
    width = 2 * np.pi / 36
    bars_true = plt.bar(angles, true_counts, width=width, alpha=0.6, label='真实值', color='blue')
    bars_pred = plt.bar(angles, pred_counts, width=width, alpha=0.6, label='预测值', color='orange', bottom=true_counts)

    plt.title('方位角分布对比（玫瑰图）', fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"training_results_{TRAIN_START_TIME}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"训练结果图已保存：{save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练SeisT-M地震方位角估计模型")
    parser.add_argument("--model_name", type=str, default="SeisT_M_BAZ",
                        help="模型名称，默认为SeisT_M_BAZ")
    parser.add_argument("--h5_path", type=str, default=H5_FILE_PATH,
                        help="H5数据文件路径")
    parser.add_argument("--no_drop_path", action="store_true",
                        help="不使用DropPath正则化")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="批次大小")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="训练轮次")
    parser.add_argument("--test-only-checkpoint", type=str,
                        help="仅加载指定最佳权重并测试，不重新训练")
    parser.add_argument("--seed", type=int,
                        help="固定模型初始化和训练打乱；不改变 seed=42 的数据划分")

    args = parser.parse_args()

    # 更新配置参数
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs

    if args.test_only_checkpoint:
        test_only(args.test_only_checkpoint, args.h5_path)
    else:
        main(
            model_name=args.model_name,
            h5_path=args.h5_path,
            use_drop_path=not args.no_drop_path,
            seed=args.seed
        )
