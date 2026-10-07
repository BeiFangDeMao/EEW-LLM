#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SeisMoLLM 对比入口：三项回归直接调用原项目骨干和任务头。

仅因 300 点波形将回归头的两层卷积改为同核大小/步长的补零卷积。
P 波拾取仍保留此前的单通道适配版，本轮不调整。文件后半部是四任务共用训练入口。
"""

from functools import partial
import importlib.util
import logging
import os
import sys
import time
import types
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from peft import LoraConfig, get_peft_model
from transformers import GPT2Model


def auto_pad_1d(x, kernel_size, stride=1):
    pad = (stride - x.size(-1) % stride) % stride + kernel_size - stride
    return F.pad(x, (pad // 2, pad - pad // 2))


class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride):
        super().__init__()
        self.in_proj = nn.Conv1d(in_dim, in_dim, 1, bias=False)
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size, stride, bias=False)
        self.norm = nn.BatchNorm1d(out_dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.in_proj(x)
        x = auto_pad_1d(x, self.conv.kernel_size[0], self.conv.stride[0])
        return self.act(self.norm(self.conv(x)))


class MultiScaleConvBlock(nn.Module):
    def __init__(self, scale_stride, in_dim, out_dim, kernel_size, stride):
        super().__init__()
        self.convs = nn.ModuleList([
            ConvBlock(in_dim, out_dim, kernel_size + scale_stride * i, stride)
            for i in range(4)
        ])
        self.out_proj = nn.Conv1d(4 * out_dim, out_dim, 1, bias=False)
        self.norm = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        return self.norm(self.out_proj(torch.cat([conv(x) for conv in self.convs], dim=1)))


class GPT2LoRABlock(nn.Module):
    def __init__(self, model_path, layers=3, patch_size=8):
        super().__init__()
        if not 1 <= layers <= 12:
            raise ValueError("GPT-2 层数应为 1–12")
        self.patch_size = patch_size
        # 输入采用 inputs_embeds，因此保留词嵌入参数不影响实际前向计算。
        # 与原实现一致，先加载预训练 GPT-2，再截取前 layers 层。
        llm = GPT2Model.from_pretrained(str(model_path), local_files_only=True).float()
        llm.h = llm.h[:layers]
        config = LoraConfig(target_modules="all-linear", r=16, lora_alpha=16,
                            lora_dropout=0.1, bias="lora_only")
        self.llm = get_peft_model(llm, config)
        for name, parameter in self.llm.named_parameters():
            parameter.requires_grad = any(s in name for s in ("lora", "ln", "wpe"))

    def forward(self, x):
        # 原模型的 unfold 会忽略不足一个 patch 的尾部，不能要求整除。
        x = x.unfold(-1, self.patch_size, self.patch_size)
        x = rearrange(x, "b c n p -> b n (c p)")
        x = self.llm(inputs_embeds=x).last_hidden_state
        return rearrange(x, "b n (c p) -> b c (n p)", p=self.patch_size)


class ScaledSigmoid(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return torch.sigmoid(x) * self.scale


class RegressionHead(nn.Module):
    def __init__(self, channels, outputs, activation):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(channels, channels, 16, 4)
                                    for _ in range(2)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.lin = nn.Linear(channels, outputs)
        self.activation = activation

    def forward(self, x, _waveform):
        for conv in self.convs:
            # 原论文使用较长波形；300 点输入时第二个 16 点卷积核会大于
            # 当前特征长度。只在此处补零，保留卷积核/步长/参数不变。
            x = conv(auto_pad_1d(x, conv.kernel_size[0], conv.stride[0]))
        return self.activation(self.lin(self.pool(x).flatten(1)))


class PPickingHead(nn.Module):
    """原 HeadDetectionPicking 的上采样结构；只输出 P 波概率。"""

    def __init__(self, channels=96):
        super().__init__()
        layer_channels = [224, 192, 160, 128]
        layer_kernels = [1, 6, 8, 16]
        outputs = [channels] + layer_channels[:-1]
        targets = layer_channels[:-1] + [2]
        self.up_layers = nn.ModuleList([
            nn.Sequential(nn.Conv1d(inc, outc, kernel), nn.BatchNorm1d(outc), nn.GELU())
            for inc, outc, kernel in zip(outputs, targets, layer_kernels)
        ])
        self.out_conv = nn.Conv1d(2, 1, 7, padding=3)

    def forward(self, x, waveform):
        size = waveform.size(-1)
        factor = (size / x.size(-1)) ** (1 / len(self.up_layers))
        sizes = [size] * len(self.up_layers)
        for i in range(len(sizes) - 2, -1, -1):
            sizes[i] = int(sizes[i + 1] / factor)
        for layer, target in zip(self.up_layers, sizes):
            x = F.interpolate(x, size=target, mode="linear")
            x = auto_pad_1d(x, layer[0].kernel_size[0])
            x = layer(x)
        return torch.sigmoid(self.out_conv(x))


class SeisMoLLM(nn.Module):
    """仅接收 [batch, 3, 300] 波形；不接收频谱或人工特征。

    task: picking / magnitude / distance / azimuth。
    输出分别为 [B,1,300]、[B,1]、[B,1]、[B,2]（sin, cos）。
    """

    def __init__(self, task, gpt2_path, llm_layers=3):
        super().__init__()
        if task not in ("picking", "magnitude", "distance", "azimuth"):
            raise ValueError(f"未知任务：{task}")
        self.task = task
        if task == "picking":
            # 拾取任务暂不修改，仍保留上一版的单 P 波实现。
            channels = [3, 16, 48, 96, 96]
            self.convs = nn.Sequential(*[
                MultiScaleConvBlock(s, channels[i], channels[i + 1], k, stride)
                for i, (s, k, stride) in enumerate(zip(
                    [8, 6, 4, 2], [16, 8, 6, 1], [2, 2, 2, 1]))
            ])
            self.llm_blocks = GPT2LoRABlock(gpt2_path, layers=llm_layers, patch_size=8)
            self.out_head = PPickingHead()
            return

        source_file = Path(__file__).resolve().parent / "original_models" / "SeisMoLLM.py"
        if not source_file.is_file():
            raise FileNotFoundError(f"原项目模型不存在：{source_file}")
        package = "seismollm_original_components"
        if package not in sys.modules:
            pkg = types.ModuleType(package)
            pkg.__path__ = [str(source_file.parent)]
            sys.modules[package] = pkg
        module_name = package + ".SeisMoLLM"
        if module_name not in sys.modules:
            spec = importlib.util.spec_from_file_location(module_name, source_file)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        original = sys.modules[module_name]
        weight_dir = Path(gpt2_path)
        if not weight_dir.is_dir():
            raise FileNotFoundError(f"本地 GPT-2 权重目录不存在：{weight_dir}")
        original.GPT_file_path = str(weight_dir.parent) + "/"

        class ShortRegressionHead(original.HeadRegression):
            def forward(self, x, _=None):
                for conv in self.convs:
                    x = conv(auto_pad_1d(x, conv.kernel_size[0], conv.stride[0]))
                return self.out_act(self.lin(self.flatten(self.pool(x))))

        class ShortBAZHead(original.HeadBAZ):
            def forward(self, x, _=None):
                for conv in self.convs:
                    x = conv(auto_pad_1d(x, conv.kernel_size[0], conv.stride[0]))
                x = self.out_act(self.lin(self.flatten(self.pool(x))))
                return x[:, :1], x[:, 1:]

        if task == "magnitude":
            head = partial(ShortRegressionHead,
                           out_act_layer=partial(original.ScaledActivation,
                                                 act_layer=nn.Sigmoid, scale_factor=8))
        elif task == "distance":
            head = partial(ShortRegressionHead,
                           out_act_layer=partial(original.ScaledActivation,
                                                 act_layer=nn.Sigmoid, scale_factor=500))
        else:
            head = partial(ShortBAZHead, out_act_layer=nn.Tanh)
        # 直接实例化原项目主模型，仅将 300 点输入下无法运行的任务头卷积补零。
        self.network = original.SeisMoLLM(
            output_head=head, llm_layers=llm_layers,
            conv_channels=[16, 48, 96], conv_scale_strides=[8, 6, 4, 2],
            conv_kernel_sizes=[16, 8, 6, 1], conv_strides=[2, 2, 2, 1])

    def forward(self, waveform):
        if waveform.ndim != 3 or waveform.shape[1:] != (3, 300):
            raise ValueError(f"需要 [B,3,300] 波形，收到 {tuple(waveform.shape)}")
        if self.task != "picking":
            output = self.network(waveform)
            if self.task == "azimuth":
                # 原项目 HeadBAZ 输出 (cos,sin)；BART 结果表使用 (sin,cos)。
                return torch.cat((output[1], output[0]), dim=1)
            return output
        x = self.convs(waveform)
        x = self.llm_blocks(x)
        return self.out_head(x, waveform)


def SeisMoLLM_picking(gpt2_path, **kwargs):
    return SeisMoLLM("picking", gpt2_path, **kwargs)


def SeisMoLLM_magnitude(gpt2_path, **kwargs):
    return SeisMoLLM("magnitude", gpt2_path, **kwargs)


def SeisMoLLM_distance(gpt2_path, **kwargs):
    return SeisMoLLM("distance", gpt2_path, **kwargs)


def SeisMoLLM_azimuth(gpt2_path, **kwargs):
    return SeisMoLLM("azimuth", gpt2_path, **kwargs)


def compute_picking_detection_metrics(truth, pred, threshold=0.5, tolerance=10):
    """由单通道 P 波概率曲线同时计算事件判别和 P 波拾取指标。"""
    import numpy as np

    if truth.ndim != 3 or pred.shape != truth.shape or truth.shape[1] != 1:
        raise ValueError("P 波标签和预测均应为相同形状的 [N, 1, T]")
    true_curve, pred_curve = truth[:, 0], pred[:, 0]
    true_event = true_curve.max(axis=1) > 0.1
    peak_probability = pred_curve.max(axis=1)
    pred_event = peak_probability > threshold
    true_p = np.where(true_event, true_curve.argmax(axis=1), -1)
    pred_p = pred_curve.argmax(axis=1)
    pick_error = np.where(true_event, np.abs(pred_p - true_p), np.nan)
    tp = int(np.sum(true_event & pred_event))
    fn = int(np.sum(true_event & ~pred_event))
    fp = int(np.sum(~true_event & pred_event))
    tn = int(np.sum(~true_event & ~pred_event))
    has_both_classes = bool(true_event.any() and (~true_event).any())
    event_metrics = {
        "threshold": threshold, "positive_samples": int(true_event.sum()),
        "negative_samples": int((~true_event).sum()),
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "recall": tp / (tp + fn) if tp + fn else None,
        # 无噪声负样本时无法评价虚警，故不报告完整检测性能。
        "accuracy": (tp + tn) / len(true_event) if has_both_classes else None,
        "precision": tp / (tp + fp) if has_both_classes and tp + fp else None,
        "F1": 2 * tp / (2 * tp + fp + fn) if has_both_classes and 2 * tp + fp + fn else None,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
    }
    picking_metrics = {
        "P_MAE_samples": float(np.nanmean(pick_error)) if true_event.any() else None,
        "P_within_tolerance": float(np.mean(pick_error[true_event] <= tolerance))
        if true_event.any() else None,
        "tolerance_samples": tolerance,
    }
    return event_metrics, picking_metrics, true_event, pred_event, true_p, pred_p, peak_probability, pick_error


# 五文件布局：以下是四个训练入口共用的数据/训练逻辑。
# 任务脚本只指定 task，保证读取、划分、优化和评估口径完全一致。
def train_task(task):
    import argparse
    import csv
    import math
    import random
    from pathlib import Path

    import h5py
    import numpy as np
    from tqdm import tqdm
    from torch.utils.data import DataLoader, Dataset, random_split

    default_h5 = Path(r"D:\EEWLMDATASET\JKnet_3001_Pphase_noise.h5" if task == "picking" else r"D:\EEWLMDATASET\JKnet_300.h5")

    class WaveDataset(Dataset):
        def __init__(self, path):
            self.path = str(path)
            self.rows = []
            with h5py.File(path, "r") as f:
                keys = sorted(k for k in f if k != "fft_spectrum" and isinstance(f[k], h5py.Dataset))
                # 与 BART 数据加载方式一致：一次性读取，训练时不再访问 H5。
                waves = np.empty((len(keys), 3, 300), dtype=np.float32)
                for key in tqdm(keys, desc=f"筛选 {task} 样本", file=sys.stdout,
                                dynamic_ncols=True, mininterval=1,
                                disable=not sys.stdout.isatty()):
                    ds = f[key]
                    if len(ds.shape) != 2 or ds.shape != (300, 3):
                        continue
                    field = {"picking": "new_pat_sample", "magnitude": "mag",
                             "distance": "dis", "azimuth": "azi"}[task]
                    try:
                        value = float(ds.attrs.get(field, np.nan))
                    except (TypeError, ValueError):
                        continue
                    if not np.isfinite(value):
                        continue
                    if task == "picking" and not (value == -1 or 0 <= value < 300):
                        continue
                    if task == "magnitude" and value <= 0:
                        continue
                    wave = np.asarray(ds[()], dtype=np.float32).T
                    if task == "picking":
                        std = wave.std(axis=1, keepdims=True)
                        wave = (wave - wave.mean(axis=1, keepdims=True)) / np.where(std == 0, 1, std)
                    waves[len(self.rows)] = wave
                    self.rows.append((key, value))
            if not self.rows:
                raise ValueError(f"{path} 未找到 {task} 可用样本")
            self.waves = waves[:len(self.rows)]

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            key, value = self.rows[i]
            wave = self.waves[i]
            if task == "picking":
                label = np.zeros((1, 300), dtype=np.float32)
                if value >= 0:
                    positions = np.arange(300)
                    label[0] = np.exp(-((positions - int(value)) ** 2) / (2 * 5 ** 2))
                    label[label < 1e-3] = 0
            elif task == "azimuth":
                radians = math.radians(value)
                label = np.array([math.sin(radians), math.cos(radians)], dtype=np.float32)
            else:
                # SeisMoLLM 原回归头直接输出震级或 km；震中距不做对数变换。
                label = np.array([value], dtype=np.float32)
            return torch.from_numpy(wave), torch.from_numpy(label), key

    parser = argparse.ArgumentParser(description=f"SeisMoLLM {task} training")
    parser.add_argument("--h5", type=Path, default=default_h5,
                        help="拾取任务请传入带 new_pat_sample 的 H5；其他任务默认 with_FS")
    parser.add_argument("--gpt2", type=Path, default=Path(r"D:\ZJN\GPT2"),
                        help="本地 GPT-2 权重目录")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "runs" / task)
    original_batch = {"magnitude": 500, "distance": 256, "azimuth": 128,
                      "picking": 96}[task]
    parser.add_argument("--batch-size", type=int, default=original_batch)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30 if task != "picking" else 20)
    parser.add_argument("--log-every-batches", type=int, default=100,
                        help="非交互运行时，每隔多少个训练 batch 记录一次进度")
    if task == "picking":
        parser.add_argument("--event-threshold", type=float, default=0.5)
        parser.add_argument("--pick-tolerance", type=int, default=10)
    args = parser.parse_args()
    if args.log_every_batches < 1:
        parser.error("--log-every-batches 必须大于 0")
    if not args.h5.is_file():
        raise FileNotFoundError(args.h5)
    if not args.gpt2.is_dir():
        raise FileNotFoundError(f"本地 GPT-2 权重目录不存在：{args.gpt2}")
    args.output.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"SeisMoLLM.{task}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    for handler in (logging.FileHandler(args.output / "train_detail.log", encoding="utf-8"),
                    logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.info("任务=%s | H5=%s | GPT-2=%s | 输出=%s", task, args.h5, args.gpt2, args.output)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset = WaveDataset(args.h5)
    n = len(dataset)
    sizes = [int(n * .80), int(n * .15)]
    sizes.append(n - sum(sizes))
    train_set, val_set, test_set = random_split(
        dataset, sizes, generator=torch.Generator().manual_seed(42))
    train_fraction = float(os.environ.get("EEW_TRAIN_FRACTION", "1"))
    if not 0 < train_fraction <= 1:
        raise ValueError("EEW_TRAIN_FRACTION 必须在 (0, 1] 内")
    if train_fraction < 1:
        indices = torch.randperm(len(train_set), generator=torch.Generator().manual_seed(42))
        train_set = torch.utils.data.Subset(train_set, indices[:max(1, int(len(train_set) * train_fraction))].tolist())
    logger.info("task=%s H5=%s split=%s seed=42", task, args.h5, sizes)
    logger.info("train_fraction=%.0f%% actual_train=%d validation=%d test=%d",
                train_fraction * 100, len(train_set), len(val_set), len(test_set))

    parts = (train_set, val_set, test_set)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    loaders = [DataLoader(part, batch_size=args.batch_size, shuffle=(i == 0),
                          num_workers=0, pin_memory=pin_memory)
               for i, part in enumerate(parts)]
    logger.info("设备=%s | batch_size=%d | pin_memory=%s", device, args.batch_size, pin_memory)
    model = SeisMoLLM(task, args.gpt2).to(device)
    if task == "picking":
        # 拾取任务暂不调整，保留上一版设置。
        from transformers import get_cosine_schedule_with_warmup
        criterion = nn.BCELoss()
        backbone = [p for name, p in model.named_parameters() if p.requires_grad and "out_head" not in name]
        head = [p for name, p in model.named_parameters() if p.requires_grad and "out_head" in name]
        optimizer = torch.optim.AdamW([{"params": backbone, "lr": 2e-4},
                                       {"params": head, "lr": 1e-3}])
        total_steps = len(loaders[0]) * args.epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, max(1, int(total_steps * .05)), total_steps)
    else:
        # 原项目 config.py: emg/dis=Huber；baz=两个 Huber 之和。
        huber = nn.HuberLoss()

        def criterion(output, target):
            if task == "azimuth":
                return huber(output[:, :1], target[:, :1]) + huber(output[:, 1:], target[:, 1:])
            return huber(output, target)

        # 原项目 training/train.py: 单参数组 Adam、weight_decay=0、CyclicLR。
        # 震中距/方位角采用原 run_scripts 的任务配置；震级采用 main.py 默认值。
        base_lr = 8e-5 if task == "magnitude" else 5e-4
        warmup = {"magnitude": 2000, "distance": 4500, "azimuth": 2500}[task]
        down = {"magnitude": 3000, "distance": 5000, "azimuth": 3000}[task]
        optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=base_lr, max_lr=1e-3,
            step_size_up=warmup, step_size_down=down, cycle_momentum=False)
    def evaluate(loader, mode):
        model.eval()
        losses, labels, predictions, keys = [], [], [], []
        with torch.no_grad():
            for wave, target, names in tqdm(loader, desc=f"{task} {mode}",
                                            file=sys.stdout, dynamic_ncols=True, mininterval=1,
                                            disable=not sys.stdout.isatty()):
                pred = model(wave.to(device, non_blocking=pin_memory)).float()
                losses.append(criterion(pred, target.to(device, non_blocking=pin_memory)).item() * len(names))
                labels.append(target.numpy())
                predictions.append(pred.cpu().numpy())
                keys.extend(names)
        return (sum(losses) / len(keys), np.concatenate(labels),
                np.concatenate(predictions), keys)

    best, stale = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_loss = 0.0
        progress = tqdm(loaders[0], desc=f"{task} 训练 Epoch {epoch}/{args.epochs}",
                        file=sys.stdout, dynamic_ncols=True, mininterval=1,
                        disable=not sys.stdout.isatty())
        for batch_index, (wave, label, _) in enumerate(progress, start=1):
            wave = wave.to(device, non_blocking=pin_memory)
            label = label.to(device, non_blocking=pin_memory)
            optimizer.zero_grad()
            loss = criterion(model(wave).float(), label)
            loss.backward()
            if task == "picking":
                torch.nn.utils.clip_grad_norm_(backbone, .5)
                torch.nn.utils.clip_grad_norm_(head, 5.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * len(wave)
            progress.set_postfix(loss=f"{loss.item():.4f}",
                                 lr=f"{optimizer.param_groups[0]['lr']:.2e}")
            if not sys.stdout.isatty() and batch_index % args.log_every_batches == 0:
                logger.info("epoch=%d batch=%d/%d loss=%.4f lr=%.2e",
                            epoch, batch_index, len(loaders[0]), loss.item(),
                            optimizer.param_groups[0]["lr"])
        val_loss = evaluate(loaders[1], "验证")[0]
        logger.info("epoch=%d train_loss=%.6f val_loss=%.6f duration_s=%.1f",
                    epoch, train_loss / len(parts[0]), val_loss,
                    time.perf_counter() - epoch_start)
        if val_loss < best:
            best, stale = val_loss, 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "val_loss": best}, args.output / "best_model.pth")
        else:
            stale += 1
            if stale > args.patience:
                break

    state = torch.load(args.output / "best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"], strict=True)
    test_loss, truth, pred, keys = evaluate(loaders[2], "测试")
    if task == "azimuth":
        true_deg = np.degrees(np.arctan2(truth[:, 0], truth[:, 1])) % 360
        pred_deg = np.degrees(np.arctan2(pred[:, 0], pred[:, 1])) % 360
        signed = (pred_deg - true_deg + 180) % 360 - 180
        metrics = {"MAAE_deg": float(np.abs(signed).mean()),
                   "MSAE_deg": float(signed.mean()),
                   "SD_SAE_deg": float(signed.std(ddof=1)),
                   "R2_sin_cos_mean": float(np.mean([
                       np.corrcoef(truth[:, j], pred[:, j])[0, 1] ** 2 for j in range(2)]))}
        header = ["event_id", "true_sin", "true_cos", "pred_sin", "pred_cos",
                  "true_deg", "pred_deg", "signed_error_deg"]
        rows = zip(keys, truth[:, 0], truth[:, 1], pred[:, 0], pred[:, 1],
                   true_deg, pred_deg, signed)
    elif task == "picking":
        (event_metrics, picking_metrics, true_event, pred_event, true_p, pred_p,
         peak_probability, pick_error) = compute_picking_detection_metrics(
            truth, pred, args.event_threshold, args.pick_tolerance)
        metrics = {"event_detection": event_metrics, "P_picking": picking_metrics}
        if not event_metrics["negative_samples"]:
            print("警告：测试集无噪声负样本；事件检测的准确率、精确率、F1 和虚警率不可评估。",
                  flush=True)
        header = ["event_id", "true_event", "pred_event", "true_p_index",
                  "pred_p_index", "peak_probability", "pick_error_samples"]
        rows = zip(keys, true_event.astype(int), pred_event.astype(int), true_p,
                   pred_p, peak_probability, pick_error)
    else:
        y, yhat = truth[:, 0], pred[:, 0]
        mse = float(np.square(y - yhat).mean())
        variance = float(np.square(y - y.mean()).sum())
        metrics = {"MAE": float(np.abs(y - yhat).mean()), "MSE": mse,
                   "RMSE": math.sqrt(mse),
                   "R2": 1 - float(np.square(y - yhat).sum()) / variance if variance else None}
        header = ["event_id", "true", "pred", "residual"]
        rows = zip(keys, y, yhat, y - yhat)
    with (args.output / "test_results.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    logger.info("best_epoch=%d test_loss=%.6f metrics=%s", state['epoch'], test_loss, metrics)
