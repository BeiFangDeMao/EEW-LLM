#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project : LLM
# @File : train_combined_simple.py
# @IDE : PyCharm
# @Author : 张嘉南
# @Date : 2025/11/1
"""
简化版地震震级估计模型训练脚本
只加载波形数据和标签
输入: wave(3,300)
输出: 震级值
"""
import os
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
import matplotlib.pyplot as plt
import csv
import time
import json
import sys

# 调整字体设置
plt.rcParams["font.family"] = ["STIXGeneral", "DejaVu Sans", "SimHei", "Microsoft YaHei"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams['axes.unicode_minus'] = False

# 导入模型（请确保你的模型文件正确导入）
from magnet import magnet

MODEL_REGISTRY = {
    "magnet": magnet
}


def save_test_table(all_keys, all_trues, all_preds, save_dir, timestamp):
    """生成测试结果表格 (事件ID, 真实值, 预测值, 残差)"""
    csv_path = os.path.join(save_dir, f"test_table_{timestamp}.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(["事件ID", "真实值", "预测值", "残差"])
        for k, t, p in zip(all_keys, all_trues, all_preds):
            writer.writerow([k, round(float(t), 4), round(float(p), 4), round(float(t - p), 4)])
    logging.info(f"测试表格已保存至：{csv_path}")


# -------------------------- 1. 基础配置 --------------------------
TRAIN_START_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_DIR = os.path.abspath("")
CHECKPOINT_DIR = os.path.join(BASE_DIR, f"checkpoint_simple_{TRAIN_START_TIME}")
LOG_DIR = os.path.join(BASE_DIR, f"log_simple_{TRAIN_START_TIME}")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 数据与训练参数 - 直接设置固定值
H5_FILE_PATH = r"D:\EEWLMDATASET\JKnet_300.h5"  # 直接使用固定路径
MODEL_NAME = "magnet"  # 直接使用固定模型名称
TARGET_LENGTH = 300
TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT = 0.80, 0.15, 0.05
BATCH_SIZE = 64
EPOCHS = 200
EARLY_STOP_PATIENCE = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------- 2. 增强版日志配置 --------------------------
# 清除现有的日志处理器，避免重复
root_logger = logging.getLogger()
if root_logger.handlers:
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

log_format = logging.Formatter(
    "%(asctime)s - %(name)s - %(module)s - %(levelname)s - %(message)s")
file_handler = logging.FileHandler(
    os.path.join(LOG_DIR, "train_simple.log"), encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(log_format)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(log_format)

root_logger.setLevel(logging.DEBUG)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

logging.info("=" * 50)
logging.info(f"训练启动时间：{TRAIN_START_TIME}")
logging.info(f"使用设备：{device}")
logging.info(f"训练参数：")
logging.info(f" H5数据路径：{H5_FILE_PATH}")
logging.info(f" 数据尺寸：波形(3,{TARGET_LENGTH})")
logging.info(f" 数据拆分：训练{TRAIN_SPLIT * 100}% | 验证{VAL_SPLIT * 100}% | 测试{TEST_SPLIT * 100}%")
logging.info(f" 批次大小：{BATCH_SIZE} | 总轮次：{EPOCHS}")
logging.info(f" 早停阈值：{EARLY_STOP_PATIENCE}轮")
logging.info(f" 模型名称：{MODEL_NAME}")
logging.info("=" * 50)


# -------------------------- 3. 简化的波形数据集 --------------------------
class SeismicDatasetSimple(Dataset):
    def __init__(self, h5_file_path, target_length=300, max_samples=None):
        self.data_list = []  # 存储格式：(wave, label, key)
        self.target_length = target_length

        with h5py.File(h5_file_path, "r") as f:
            # 获取所有波形数据的key
            valid_sample_keys = list(f.keys())

            # 移除可能的非波形数据组
            if 'fft_spectrum' in valid_sample_keys:
                valid_sample_keys.remove('fft_spectrum')

            # 限制最大样本数
            if max_samples:
                valid_sample_keys = valid_sample_keys[:max_samples]
            valid_sample_keys.sort()

            # 遍历加载波形数据和标签
            for key in tqdm(valid_sample_keys, desc="加载波形数据"):
                try:
                    # 跳过fft_spectrum组
                    if key == 'fft_spectrum':
                        continue

                    # 1. 加载波形数据 (3, target_length)
                    wave_group = f[key]

                    # 检查是否是数据集（不是组）
                    if isinstance(wave_group, h5py.Dataset):
                        wave = wave_group[:].astype(np.float32).T
                    else:
                        # 如果是组，尝试获取数据集
                        if 'waveform' in wave_group.keys():
                            wave = wave_group['waveform'][:].astype(np.float32).T
                        else:
                            # 假设第一个数据集是波形数据
                            dataset_keys = list(wave_group.keys())
                            if dataset_keys:
                                wave = wave_group[dataset_keys[0]][:].astype(np.float32).T
                            else:
                                continue

                    # 确保波形是3通道
                    if wave.shape[0] != 3:
                        logging.warning(f"样本{key}波形通道数不为3，跳过")
                        continue

                    # 填充/截断到目标长度
                    if wave.shape[1] < self.target_length:
                        pad = self.target_length - wave.shape[1]
                        wave = np.pad(wave, ((0, 0), (0, pad)), mode='constant')
                    else:
                        wave = wave[:, :self.target_length]

                    # 2. 加载标签（震级）
                    label = float(wave_group.attrs.get("mag", np.nan))
                    if np.isnan(label) or label <= 0:
                        continue

                    # 3. 转换为torch张量
                    wave_tensor = torch.from_numpy(wave).to(torch.float32)
                    label_tensor = torch.tensor(label, dtype=torch.float32)

                    # 存入数据列表
                    self.data_list.append((wave_tensor, label_tensor, key))

                except Exception as e:
                    logging.warning(f"跳过样本 {key}: {e}")

        logging.info(f"H5数据集加载完成，有效样本数：{len(self.data_list)}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if len(self.data_list) == 0:
            raise ValueError("数据集无有效样本，无法获取数据！")
        return self.data_list[idx]


# -------------------------- 4. 训练函数 --------------------------
def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    predictions = []
    targets = []

    progress_bar = tqdm(dataloader, desc="训练", leave=False)
    for wave, y, _ in progress_bar:
        wave, y = wave.to(device), y.to(device).unsqueeze(1)

        # 前向传播
        outputs = model(wave)  # 注意：这里假设模型接受单输入（波形）
        loss = criterion(outputs, y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录损失
        total_loss += loss.item()

        # 记录预测值和真实值
        predictions.extend(outputs.detach().cpu().numpy().flatten())
        targets.extend(y.detach().cpu().numpy().flatten())

        # 更新进度条
        progress_bar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / len(dataloader)
    predictions = np.array(predictions)
    targets = np.array(targets)

    # 计算MAE
    mae = mean_absolute_error(targets, predictions)

    return avg_loss, mae, predictions, targets


def validate_epoch(model, dataloader, criterion, device):
    """验证一个epoch"""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="验证", leave=False)
        for wave, y, _ in progress_bar:
            wave, y = wave.to(device), y.to(device).unsqueeze(1)

            # 前向传播
            outputs = model(wave)  # 注意：这里假设模型接受单输入（波形）
            loss = criterion(outputs, y)

            # 记录损失
            total_loss += loss.item()

            # 记录预测值和真实值
            predictions.extend(outputs.detach().cpu().numpy().flatten())
            targets.extend(y.detach().cpu().numpy().flatten())

            # 更新进度条
            progress_bar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / len(dataloader)
    predictions = np.array(predictions)
    targets = np.array(targets)

    # 计算MAE
    mae = mean_absolute_error(targets, predictions)

    return avg_loss, mae, predictions, targets


# -------------------------- 5. 计算评估指标 --------------------------
def calculate_metrics(predictions, targets):
    """计算评估指标"""
    # 转换为numpy数组
    preds = np.array(predictions).flatten()
    targs = np.array(targets).flatten()

    # 计算各种指标
    mse = np.mean((preds - targs) ** 2)
    mae = np.mean(np.abs(preds - targs))
    rmse = np.sqrt(mse)

    # 计算R²
    ss_res = np.sum((targs - preds) ** 2)
    ss_tot = np.sum((targs - np.mean(targs)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    return {
        "MSE": float(mse),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }


# -------------------------- 6. 测试函数 --------------------------
def test_model(model, loader):
    model.eval()
    preds, trues, keys = [], [], []
    with torch.no_grad():
        for wave, y, k in tqdm(loader, desc="测试中"):
            wave, y = wave.to(device), y.to(device).unsqueeze(1)
            out = model(wave)  # 注意：这里假设模型接受单输入（波形）
            preds.extend(out.cpu().numpy().squeeze().tolist())
            trues.extend(y.cpu().numpy().squeeze().tolist())
            keys.extend(k)
    mae = mean_absolute_error(trues, preds)
    mse = mean_squared_error(trues, preds)
    r2 = r2_score(trues, preds)
    logging.info(f"测试结果汇总:")
    logging.info(f" MAE: {mae:.6f}")
    logging.info(f" MSE: {mse:.6f}")
    logging.info(f" R²: {r2:.6f}")
    return np.array(trues), np.array(preds), keys, (mae, mse, r2)


# -------------------------- 7. 可视化函数 --------------------------
def plot_training_results(train_losses, val_losses, train_maes, val_maes,
                          test_trues, test_preds, metrics, save_dir):
    """绘制训练结果图"""
    mae, mse, r2 = metrics

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 损失曲线
    epochs = range(1, len(train_losses) + 1)
    axes[0, 0].plot(epochs, train_losses, 'b-', label='训练损失')
    axes[0, 0].plot(epochs, val_losses, 'r-', label='验证损失')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('损失')
    axes[0, 0].set_title('训练和验证损失曲线')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. MAE曲线
    axes[0, 1].plot(epochs, train_maes, 'b-', label='训练MAE')
    axes[0, 1].plot(epochs, val_maes, 'r-', label='验证MAE')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('训练和验证MAE曲线')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 预测vs真实值散点图
    axes[1, 0].scatter(test_trues, test_preds, s=30, alpha=0.6)
    axes[1, 0].plot([min(test_trues), max(test_trues)], [min(test_trues), max(test_trues)], 'r--')
    axes[1, 0].set_xlabel("真实震级")
    axes[1, 0].set_ylabel("预测震级")
    axes[1, 0].set_title(f"测试集: 真实 vs 预测 (MAE={mae:.3f}, R²={r2:.3f})")
    axes[1, 0].grid(True, alpha=0.3)

    # 4. 误差分布直方图
    errors = test_preds - test_trues
    axes[1, 1].hist(errors, bins=30, alpha=0.7, color='blue', density=True)
    axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('预测误差')
    axes[1, 1].set_ylabel('密度')
    axes[1, 1].set_title('测试集预测误差分布')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "training_test_results.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logging.info(f"训练与测试结果图已保存：{save_path}")
    plt.close()


# -------------------------- 8. 主函数 --------------------------
def main():
    """主训练函数，直接使用固定的参数"""

    # 设置随机种子
    def set_seed(seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed(42)

    # 加载简化的波形数据集
    logging.info("开始加载波形数据集...")
    dataset = SeismicDatasetSimple(
        h5_file_path=H5_FILE_PATH,
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
        dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )
    train_fraction = float(os.environ.get("EEW_TRAIN_FRACTION", "1"))
    if not 0 < train_fraction <= 1:
        raise ValueError("EEW_TRAIN_FRACTION 必须在 (0, 1] 内")
    if train_fraction < 1:
        indices = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(42))
        train_ds = torch.utils.data.Subset(train_ds, indices[:max(1, int(len(train_ds) * train_fraction))].tolist())
    logging.info(f"训练集使用比例: {train_fraction:.0%} | 实际训练样本: {len(train_ds)}")
    logging.info(f"数据集拆分完成: 训练集{len(train_ds)} | 验证集{len(val_ds)} | 测试集{len(test_ds)}")

    # 数据加载器初始化
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=0)

    # 初始化模型
    if MODEL_NAME not in MODEL_REGISTRY:
        raise ValueError(f"未知模型名称: {MODEL_NAME}。可选: {list(MODEL_REGISTRY.keys())}")
    model_class = MODEL_REGISTRY[MODEL_NAME]
    model = model_class().to(device)
    logging.info(f"模型结构:\n{model}")

    # 检查模型输入参数数量
    import inspect
    model_forward_params = inspect.signature(model.forward).parameters
    logging.info(f"模型forward函数参数: {list(model_forward_params.keys())}")

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # 最佳模型保存相关参数
    best_val_loss = float('inf')
    best_model_path = os.path.join(CHECKPOINT_DIR, f"best_model_{TRAIN_START_TIME}.pth")
    early_stop_counter = 0

    # 记录训练过程指标
    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []
    train_metrics_history = []
    val_metrics_history = []

    # 开始训练
    logging.info("开始训练...")
    for epoch in range(1, EPOCHS + 1):
        logging.info(f"\n{'=' * 20} Epoch {epoch}/{EPOCHS} {'=' * 20}")
        start_time = time.time()

        # 训练
        train_loss, train_mae, train_preds, train_targets = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        train_losses.append(train_loss)
        train_maes.append(train_mae)

        # 验证
        val_loss, val_mae, val_preds, val_targets = validate_epoch(
            model, val_loader, criterion, device
        )
        val_losses.append(val_loss)
        val_maes.append(val_mae)

        # 计算详细指标
        train_metrics = calculate_metrics(train_preds, train_targets)
        val_metrics = calculate_metrics(val_preds, val_targets)
        train_metrics_history.append(train_metrics)
        val_metrics_history.append(val_metrics)

        # 更新学习率
        scheduler.step(val_loss)

        # 输出训练信息
        epoch_time = time.time() - start_time
        logging.info(f"训练损失: {train_loss:.6f} | 验证损失: {val_loss:.6f}")
        logging.info(f"训练MAE: {train_mae:.6f} | 验证MAE: {val_mae:.6f}")
        logging.info(f"训练R²: {train_metrics['R2']:.4f} | 验证R²: {val_metrics['R2']:.4f}")
        logging.info(f"当前学习率: {optimizer.param_groups[0]['lr']:.6e}")
        logging.info(f"耗时: {epoch_time:.2f}秒")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_mae': val_mae,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'config': {
                    'model_name': MODEL_NAME,
                    'batch_size': BATCH_SIZE,
                    'learning_rate': 0.001,
                    'input_type': 'waveform_only'
                }
            }, best_model_path)
            logging.info(f"保存最佳模型到: {best_model_path}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            logging.info(f"未更新最佳模型 ({early_stop_counter}/{EARLY_STOP_PATIENCE})")

        # 早停检查
        if early_stop_counter >= EARLY_STOP_PATIENCE:
            logging.info(f"早停触发！最佳模型出现在 Epoch {epoch - EARLY_STOP_PATIENCE}")
            break

    # 最终模型保存
    final_model_path = os.path.join(CHECKPOINT_DIR, f"final_model_{TRAIN_START_TIME}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'model_name': MODEL_NAME,
            'batch_size': BATCH_SIZE,
            'learning_rate': 0.001,
            'input_type': 'waveform_only'
        }
    }, final_model_path)
    logging.info(f"保存最终模型到: {final_model_path}")

    # 测试最佳模型
    logging.info("\n" + "=" * 50)
    logging.info("开始测试最佳模型...")
    best_checkpoint = torch.load(best_model_path)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    logging.info(f"加载最佳模型 (Epoch {best_checkpoint['epoch']}): MAE={best_checkpoint['val_mae']:.6f}")

    # 测试
    trues, preds, keys, metrics = test_model(model, test_loader)
    mae, mse, r2 = metrics

    # 保存测试表格
    save_test_table(keys, trues, preds, CHECKPOINT_DIR, TRAIN_START_TIME)

    # 绘制结果图
    plot_training_results(train_losses, val_losses, train_maes, val_maes,
                          trues, preds, metrics, CHECKPOINT_DIR)

    # 保存训练历史
    history = {
        "config": {
            "model_name": MODEL_NAME,
            "h5_path": H5_FILE_PATH,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "train_split": TRAIN_SPLIT,
            "val_split": VAL_SPLIT,
            "test_split": TEST_SPLIT,
            "learning_rate": 0.001,
            "input_type": "waveform_only"
        },
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_maes": train_maes,
        "val_maes": val_maes,
        "train_metrics": train_metrics_history,
        "val_metrics": val_metrics_history,
        "test_metrics": {
            "MAE": float(mae),
            "MSE": float(mse),
            "R2": float(r2)
        },
        "best_model_path": best_model_path,
        "final_model_path": final_model_path,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    history_path = os.path.join(CHECKPOINT_DIR, "training_history.json")
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4, default=str, ensure_ascii=False)

    logging.info(f"训练历史保存到: {history_path}")
    logging.info("所有训练流程完成！")


# -------------------------- 9. 直接执行训练 --------------------------
if __name__ == "__main__":
    # 直接执行训练，不需要命令行参数
    print("=" * 60)
    print(f"开始训练 MagNet 模型")
    print(f"数据路径: {H5_FILE_PATH}")
    print(f"模型名称: magnet")
    print(f"开始时间: {TRAIN_START_TIME}")
    print("=" * 60)

    # 执行训练
    main()

    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"检查点保存到: {CHECKPOINT_DIR}")
    print(f"日志保存到: {LOG_DIR}")
    print("=" * 60)
