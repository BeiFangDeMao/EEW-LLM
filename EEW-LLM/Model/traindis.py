#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project : LM
# @File : train_dis.py
# @IDE : PyCharm
# @Author : 张嘉南
# @Date : 2025/12/28 下午6:37
"""
Chronos-T5 + LoRA 地震震中距估计模型训练脚本
适配三输入模型（wave, spec, feat），替换正确数据集
"""
from transformers import get_cosine_schedule_with_warmup
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
from transformers import get_linear_schedule_with_warmup  # 新增导入

# 调整字体设置
plt.rcParams["font.family"] = ["STIXGeneral", "DejaVu Sans", "SimHei", "Microsoft YaHei"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams['axes.unicode_minus'] = False


from EEWLMT5 import EEW_LM_t5_dis, EEW_LM_t5_mag, EEW_LM_t5_azi
from EEWLMGPT import EEW_LM_GPT2_azi, EEW_LM_GPT2_mag, EEW_LM_GPT2_dis
from EEWLMCT5 import EEW_LMChronos_t5_azi, EEW_LMChronos_t5_mag, EEW_LMChronos_t5_dis
from EEWLMBART import EEW_LMBART_azi, EEW_LMBART_mag, EEW_LMBART_dis, EEW_LMBART_pp
from EEWLMBERT import EEW_LMBERT_azi, EEW_LMBERT_mag, EEW_LMBERT_dis
from EEWLMBART_XIAORONG import EEW_LMBART_aziA, EEW_LMBART_magA, EEW_LMBART_disA
from EEWLMBART_XIAORONG import EEW_LMBART_aziB, EEW_LMBART_magB, EEW_LMBART_disB
from EEWLMBART_XIAORONG import EEW_LMBART_aziC, EEW_LMBART_magC, EEW_LMBART_disC

MODEL_REGISTRY = {
    "EEW_LMChronos_t5_dis": EEW_LMChronos_t5_dis,
    "EEW_LM_t5_dis": EEW_LM_t5_dis,
    "EEW_LM_GPT2_dis": EEW_LM_GPT2_dis,
    "EEW_LMBERT_dis": EEW_LMBERT_dis,
    "EEW_LMBART_dis": EEW_LMBART_dis,
    "EEW_LMBART_disA": EEW_LMBART_disA,
    "EEW_LMBART_disC": EEW_LMBART_disC,
    "EEW_LMBART_disB": EEW_LMBART_disB,

}

# 定义15个特征名称列表（与数据集保持一致）
FEATURE_NAMES = [
    'Pa', 'Pv', 'Pd', 'Pav', 'Pad', 'Pvd', 'IAA', 'IAV', 'IAD', 'IV2', 'Ia', 'Tc', 'TP', 'DI', 'Tva'
]

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
CHECKPOINT_DIR = os.path.join(BASE_DIR, f"checkpointdis_{TRAIN_START_TIME}")
LOG_DIR = os.path.join(BASE_DIR, f"logdis_{TRAIN_START_TIME}")
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
    os.path.join(LOG_DIR, "train_detailgai.log"), encoding="utf-8")
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
logging.info(f" 数据尺寸：波形(3,{TARGET_LENGTH}) | 频谱(3,{SPEC_LENGTH}) | 特征(15,)")
logging.info(f" 数据拆分：训练{TRAIN_SPLIT * 100}% | 验证{VAL_SPLIT * 100}% | 测试{TEST_SPLIT * 100}%")
logging.info(f" 批次大小：{BATCH_SIZE} | 总轮次：{EPOCHS}")
logging.info(f" 权重衰减：{WEIGHT_DECAY}")
logging.info(f" 早停阈值：{EARLY_STOP_PATIENCE}轮 | 最小改进：{EARLY_STOP_EPS}")
logging.info("=" * 50)

# -------------------------- 3. 加权损失函数（保留原逻辑）--------------------------
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

# -------------------------- 4. 正确的三输入H5数据集（保留原逻辑）--------------------------
class SeismicDataset(Dataset):
    def __init__(self, h5_file_path, target_length=300, spec_length=128, max_samples=None):
        self.data_list = []  # 存储格式：(wave, spec, feat, label, key)
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

                    dist = float(wave_group.attrs.get("dis", np.nan))
                    if np.isnan(dist) :
                        continue

                    # dist = float(wave_group.attrs.get("dis", np.nan))
                    # if np.isnan(dist) :
                    #     continue

                    eps = 1e-3
                    label = np.log(dist + eps)

                    wave_tensor = torch.from_numpy(wave).to(torch.float32)
                    spec_tensor = torch.from_numpy(spec).to(torch.float32)
                    feat_tensor = torch.from_numpy(feat).to(torch.float32)
                    label_tensor = torch.tensor(label, dtype=torch.float32)
                    self.data_list.append((wave_tensor, spec_tensor, feat_tensor, label_tensor, key))
                except Exception as e:
                    logging.warning(f"跳过样本 {key}: {e}")
        logging.info(f"H5数据集加载完成，有效样本数：{len(self.data_list)}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if len(self.data_list) == 0:
            raise ValueError("数据集无有效样本，无法获取数据！")
        return self.data_list[idx]

# -------------------------- 5. 适配三输入模型的训练/验证/测试函数 --------------------------
def train_one_epoch(model, loader, criterion, optimizer, scheduler, backbone_params, head_params, epoch):
    model.train()
    total_loss, preds, labels = 0, [], []
    for i, (wave, spec, feat, y, _) in enumerate(tqdm(loader, desc=f"训练 Epoch {epoch}")):
        wave, spec, feat, y = wave.to(device), spec.to(device), feat.to(device), y.to(device).unsqueeze(1)
        optimizer.zero_grad()
        out = model(wave, spec, feat)
        loss = criterion(out, y)
        loss.backward()
        # 分组梯度裁剪
        torch.nn.utils.clip_grad_norm_(backbone_params, max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(head_params, max_norm=5.0)
        optimizer.step()
        scheduler.step()  # Step-based scheduler
        total_loss += loss.item()
        preds.extend(out.detach().cpu().numpy().squeeze().tolist())
        labels.extend(y.cpu().numpy().squeeze().tolist())
        if (i + 1) % 100 == 0:
            batch_mae = mean_absolute_error(labels[-len(y):], preds[-len(y):])
            logging.debug(
                f"Epoch {epoch} 批次 {i + 1}/{len(loader)} - 批次损失: {loss.item():.6f} | 批次MAE: {batch_mae:.6f}")
    epoch_loss = total_loss / len(loader)
    epoch_mae = mean_absolute_error(labels, preds)
    logging.debug(f"Epoch {epoch} 训练结束 - 平均损失: {epoch_loss:.6f} | 平均MAE: {epoch_mae:.6f}")
    return epoch_loss, epoch_mae

def evaluate(model, loader, criterion, mode="验证"):
    model.eval()
    total_loss, preds, labels = 0, [], []
    with torch.no_grad():
        for wave, spec, feat, y, _ in tqdm(loader, desc=mode):
            wave, spec, feat, y = wave.to(device), spec.to(device), feat.to(device), y.to(device).unsqueeze(1)
            out = model(wave, spec, feat)
            loss = criterion(out, y)
            total_loss += loss.item()
            pred_log = out.cpu().numpy().squeeze()
            true_log = y.cpu().numpy().squeeze()
            pred_dist = np.exp(pred_log)
            true_dist = np.exp(true_log)
            preds.extend(pred_dist.tolist())
            labels.extend(true_dist.tolist())
    avg_loss = total_loss / len(loader)
    avg_mae = mean_absolute_error(labels, preds)
    logging.info(f"{mode}结束 - 平均损失: {avg_loss:.6f} | 平均MAE: {avg_mae:.6f}")
    return avg_loss, avg_mae

def test_model(model, loader):
    model.eval()
    preds, trues, keys = [], [], []
    with torch.no_grad():
        for wave, spec, feat, y, k in tqdm(loader, desc="测试中"):
            wave, spec, feat, y = wave.to(device), spec.to(device), feat.to(device), y.to(device).unsqueeze(1)
            out = model(wave, spec, feat)
            pred_log = out.cpu().numpy().squeeze()
            true_log = y.cpu().numpy().squeeze()
            pred_dist = np.exp(pred_log)
            true_dist = np.exp(true_log)
            preds.extend(pred_dist.tolist())
            trues.extend(true_dist.tolist())
            keys.extend(k)
    mae = mean_absolute_error(trues, preds)
    mse = mean_squared_error(trues, preds)
    r2 = r2_score(trues, preds)
    logging.info(f"测试结果汇总:")
    logging.info(f" MAE: {mae:.6f}")
    logging.info(f" MSE: {mse:.6f}")
    logging.info(f" R²: {r2:.6f}")
    return np.array(trues), np.array(preds), keys, (mae, mse, r2)

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

# -------------------------- 6. 主函数 --------------------------
def main(model_name="EEW_LMChronos_t5_dis", h5_path=None):
    set_deterministic(42)  # <-- 添加这一行
    logging.info("开始加载三输入H5数据集...")
    dataset = SeismicDataset(
        h5_file_path=h5_path,
        target_length=TARGET_LENGTH,
        spec_length=SPEC_LENGTH
    )
    total = len(dataset)
    if total == 0:
        logging.error("数据集无有效样本，终止训练！")
        return

    train_size, val_size = int(total * TRAIN_SPLIT), int(total * VAL_SPLIT)
    test_size = total - train_size - val_size
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )
    logging.info(f"数据集拆分完成: 训练集{len(train_ds)} | 验证集{len(val_ds)} | 测试集{len(test_ds)}")

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False)

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"未知模型名称: {model_name}。可选: {list(MODEL_REGISTRY.keys())}")
    model_class = MODEL_REGISTRY[model_name]
    model = model_class().to(device)
    logging.info(f"模型结构:\n{model}")

    # 使用你原有的加权损失函数
    criterion = WeightedLogMSELoss()

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
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=num_warmup_steps,
    #     num_training_steps=total_steps,
    # )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps,
        num_cycles=0.5,  # 标准余弦退火（降到0）
    )
    # 最佳模型保存相关参数
    best_metrics = {
        "val_mae": float("inf"),
        "val_loss": float("inf"),
        "epoch": 0,
        "model_state": None
    }
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"best_model_{TRAIN_START_TIME}.pth")
    early_stop_counter = 0

    # 记录训练过程指标
    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []

    # 开始训练
    logging.info("开始训练...")
    for epoch in range(1, EPOCHS + 1):
        logging.info(f"\n{'=' * 20} Epoch {epoch}/{EPOCHS} {'=' * 20}")
        logging.info(f"当前学习率: Backbone={optimizer.param_groups[0]['lr']:.6e}, Head={optimizer.param_groups[1]['lr']:.6e}")

        train_loss, train_mae = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            backbone_params, head_params, epoch
        )
        val_loss, val_mae = evaluate(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_maes.append(train_mae)
        val_maes.append(val_mae)

        logging.info(f"训练集 - 损失: {train_loss:.6f} | MAE: {train_mae:.6f}")
        logging.info(f"验证集 - 损失: {val_loss:.6f} | MAE: {val_mae:.6f}")

        # 最佳模型判断与保存
        if val_mae < best_metrics["val_mae"] - EARLY_STOP_EPS:
            best_metrics.update({
                "val_mae": val_mae,
                "val_loss": val_loss,
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            })
            torch.save(best_metrics, ckpt_path)
            logging.info(f"更新最佳模型 (Epoch {epoch}) - 最佳MAE: {val_mae:.6f} | 最佳损失: {val_loss:.6f}")
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
    best_checkpoint = torch.load(ckpt_path)
    model.load_state_dict(best_checkpoint["model_state"])
    logging.info(f"加载最佳模型 (Epoch {best_checkpoint['epoch']}): MAE={best_checkpoint['val_mae']:.6f}")
    trues, preds, keys, metrics = test_model(model, test_loader)
    save_test_table(keys, trues, preds, CHECKPOINT_DIR, TRAIN_START_TIME)

    # 绘制结果图
    mae, mse, r2 = metrics
    plt.figure(figsize=(10, 15))
    plt.subplot(3, 1, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='训练损失')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='验证损失')
    plt.title('训练与验证损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('损失值')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplot(3, 1, 2)
    plt.plot(range(1, len(train_maes) + 1), train_maes, label='训练MAE')
    plt.plot(range(1, len(val_maes) + 1), val_maes, label='验证MAE')
    plt.title('训练与验证MAE曲线')
    plt.xlabel('Epoch')
    plt.ylabel('MAE值')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplot(3, 1, 3)
    plt.scatter(trues, preds, s=30, alpha=0.6)
    plt.plot([min(trues), max(trues)], [min(trues), max(trues)], 'r--')
    plt.title(f"真实 vs 预测 (MAE={mae:.3f}, R²={r2:.3f})")
    plt.xlabel("真实震中距(km)"), plt.ylabel("预测震中距(km)")
    plt.tight_layout()
    save_path = os.path.join(CHECKPOINT_DIR, "training_test_results.png")
    plt.savefig(save_path, dpi=300)
    logging.info(f"训练与测试结果图已保存：{save_path}")
    logging.info("所有训练流程完成！")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Seismic Distance Estimation Model with configurable backbone")
    parser.add_argument("--model_name", type=str, default="EEW_LMBART_dis",
                        choices=list(MODEL_REGISTRY.keys()),
                        help=f"Model to train. Options: {list(MODEL_REGISTRY.keys())}")
    parser.add_argument("--h5_path", type=str, default=r"G:\迁移学习\LLM\20251209NewTry\dataset\JKnet_300_with_FS.h5", help="Path to the H5 dataset file")
    args = parser.parse_args()
    main(
        model_name=args.model_name,
        h5_path=args.h5_path,
    )