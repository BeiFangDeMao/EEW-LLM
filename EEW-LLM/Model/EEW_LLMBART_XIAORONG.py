#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project : LLM
# @File : EEWLMBART.py
# @IDE : PyCharm
# @Author : 张嘉南
# @Date : 2026/4/1 下午7:29
# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project : LLM
# @File : EEWLMBART.py
# @IDE : PyCharm
# @Author : 张嘉南
# @Date : 2026/4/1 下午5:23
"""
地震预警大语言模型 (Earthquake Early Warning Large Language Model)

本模块实现了基于BART模型的多模态地震数据处理框架，包含：
1. 多尺度卷积输入头（处理波形、频谱、特征数据）
2. BART大语言模型主干（支持LoRA微调）
3. 多个任务输出头（P波拾取、震级估计、震中距估计、方位角估计）
"""

import warnings
from functools import partial
from packaging.version import Version
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo.config
from einops import rearrange
from transformers import BartModel, BartConfig, __version__ as transformers_version
from peft import get_peft_model, LoraConfig, __version__ as peft_version

# ============================================================================
# 版本检查与配置
# ============================================================================

REQUIRED_PEFT_VERSION = "0.6.0"
REQUIRED_TRANSFORMERS_VERSION = "4.30.0"

if Version(peft_version) < Version(REQUIRED_PEFT_VERSION):
    raise ValueError(
        f"PEFT 版本过低！需 ≥ {REQUIRED_PEFT_VERSION}，当前版本：{peft_version}\n"
        f"请执行：pip install --upgrade peft"
    )

if Version(transformers_version) < Version(REQUIRED_TRANSFORMERS_VERSION):
    warnings.warn(
        f"建议升级 Transformers 至 ≥ {REQUIRED_TRANSFORMERS_VERSION}，"
        f"当前版本：{transformers_version}"
    )

# 设置 Torch Dynamo 缓存大小
torch._dynamo.config.cache_size_limit = 1024

# 模型路径配置（请根据实际情况修改）
BART_MODEL_PATH = r"G:\迁移学习\LLM\Models\bart-large"  # 本地BART模型路径


# ============================================================================
# 辅助函数与工具类
# ============================================================================

def auto_pad_1d(
        x: torch.Tensor,
        kernel_size: int,
        stride: int = 1,
        dim: int = -1,
        padding_value: float = 0.0
) -> torch.Tensor:
    """
    自动为1维卷积计算并应用填充。

    Args:
        x: 输入张量 [B, C, T] 或更高维度
        kernel_size: 卷积核大小
        stride: 步长
        dim: 需要进行填充的维度
        padding_value: 填充值

    Returns:
        填充后的张量
    """
    assert kernel_size >= stride, f"kernel_size ({kernel_size}) 必须 ≥ stride ({stride})"

    pos_dim = dim if dim >= 0 else x.dim() + dim
    # 计算所需填充大小，确保输出长度能被 stride 整除
    pad_size = (stride - (x.size(dim) % stride)) % stride + kernel_size - stride
    padding = (0, 0) * (x.dim() - pos_dim - 1) + (pad_size // 2, pad_size - pad_size // 2)

    return F.pad(x, padding, "constant", padding_value)


class PatchRecover(nn.Module):
    """
    将 Transformer 输出的 token 序列恢复为连续的时间序列。

    输入: [B, N_tokens, d_model]
    输出: [B, N_tokens * patch_size, out_dim]
    """

    def __init__(self, patch_size: int = 4, out_dim: int = 256):
        """
        Args:
            patch_size: 每个 patch 包含的时间步数
            out_dim: 输出特征的通道数
        """
        super().__init__()
        self.patch_size = patch_size
        self.out_dim = out_dim
        self.proj = nn.Linear(patch_size * out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N_tokens, d_model]，其中 d_model = patch_size * out_dim

        Returns:
            [B, N_tokens * patch_size, out_dim]
        """
        batch_size, num_tokens, d_model = x.shape
        assert d_model % self.patch_size == 0, "d_model 必须能被 patch_size 整除"

        # 拆分 patch 并重塑
        new_dim = d_model // self.patch_size
        x = x.view(batch_size, num_tokens * self.patch_size, new_dim)
        x = x.transpose(1, 2)  # [B, new_dim, N_tokens * patch_size]
        return x


class ScaledActivation(nn.Module):
    """带缩放因子的激活函数包装器"""

    def __init__(self, act_layer: nn.Module, scale_factor: float):
        """
        Args:
            act_layer: 激活函数类
            scale_factor: 输出缩放因子
        """
        super().__init__()
        self.scale_factor = scale_factor
        self.activation = act_layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x) * self.scale_factor


# ============================================================================
# 第一部分：输入头 (Input Tokenizers)
# ============================================================================

class ConvBlock(nn.Module):
    """1维卷积块，包含投影、卷积、归一化和激活"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            activation: nn.Module,
            normalization: nn.Module
    ):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            stride: 步长
            activation: 激活函数类
            normalization: 归一化层类
        """
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False)
        self.norm = normalization(out_channels)
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = auto_pad_1d(x, self.conv.kernel_size[0], self.conv.stride[0])
        x = self.conv(x)
        x = self.norm(x)
        return self.activation(x)


class MultiScaleConvBlock(nn.Module):
    """
    多尺度卷积块，使用不同尺度的卷积核并行提取特征。
    引用自王星皓的工作。
    """

    def __init__(
            self,
            num_scales: int,
            scale_stride: int,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            activation: nn.Module,
            normalization: nn.Module
    ):
        """
        Args:
            num_scales: 尺度数量
            scale_stride: 尺度步长增量
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 基础卷积核大小
            stride: 步长
            activation: 激活函数类
            normalization: 归一化层类
        """
        super().__init__()

        self.convs = nn.ModuleList([
            ConvBlock(
                in_channels, out_channels,
                kernel_size + int(scale_stride * s), stride,
                activation, normalization
            )
            for s in range(num_scales)
        ])

        self.output_proj = nn.Conv1d(num_scales * out_channels, out_channels, kernel_size=1, bias=False)
        self.norm = normalization(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        multi_scale_features = torch.cat([conv(x) for conv in self.convs], dim=1)
        return self.norm(self.output_proj(multi_scale_features))


class TemporalPatchEmbed(nn.Module):
    """
    时间维度 Patch 嵌入模块。

    输入: [B, C, T]
    输出: [B, num_patches, C * patch_size]
    """

    def __init__(self, patch_size: int):
        """
        Args:
            patch_size: 每个 patch 包含的时间步数
        """
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
        # [B, C, num_patches, patch_size]
        x = rearrange(x, "b c n p -> b n (c p)")
        return x


class SpectrogramConvEncoder(nn.Module):
    """频谱数据的卷积编码器"""

    def __init__(self, in_channels: int = 3, out_channels: int = 256):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
        """
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, bias=False, padding=1),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),

            nn.Conv1d(32, 64, kernel_size=3, bias=False, padding=1),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),

            nn.Conv1d(64, out_channels, kernel_size=3, bias=False, padding=1),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),

            nn.Dropout(p=0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SpectrogramPatch(nn.Module):
    """频谱数据的 Patch 切分模块"""

    def __init__(self, patch_size: int = 4):
        """
        Args:
            patch_size: 每个 patch 包含的时间步数
        """
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T]

        Returns:
            [B, C * patch_size, T // patch_size]
        """
        batch_size, channels, time_steps = x.shape
        x = x.reshape(batch_size, channels, self.patch_size, time_steps // self.patch_size)
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(batch_size, channels * self.patch_size, time_steps // self.patch_size)
        return x


class MLPInputHead(nn.Module):
    """特征数据的 MLP 编码器"""

    def __init__(self, hidden_dim: int = 1024, input_dim: int = 15):
        """
        Args:
            hidden_dim: 隐藏层维度
            input_dim: 输入特征维度
        """
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 256),
            nn.GELU(),
            nn.Linear(256, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, input_dim]

        Returns:
            [B, 1, hidden_dim]
        """
        x = self.feature_extractor(x)
        x = self.norm(x)
        return x.unsqueeze(1)  # 添加 token 维度


class RegressionTokenizer(nn.Module):
    """
    多模态回归任务分词器。
    将波形、频谱和特征数据编码为 Transformer 可接受的 token 序列。
    """

    def __init__(
            self,
            d_model: int = 1024,
            in_channels: int = 3,
            wave_patch_size: int = 4,
            conv_scale_num: int = 4,
            conv_scale_strides: List[int] = [8, 6, 4, 2],
            conv_channels: List[int] = [16, 48, 96],
            conv_kernel_sizes: List[int] = [16, 12, 6, 1],
            conv_strides: List[int] = [2, 2, 2, 1],
            wave_dropout: float = 0.2,
            spec_in_channels: int = 3,
            spec_out_channels: int = 256,
            spec_temporal_patch: int = 4,
            spec_dropout: float = 0.2,
            activation: nn.Module = nn.GELU,
            normalization: nn.Module = nn.BatchNorm1d,
    ):
        """
        Args:
            d_model: Transformer 模型维度
            in_channels: 波形输入通道数
            wave_patch_size: 波形 patch 大小
            conv_scale_num: 多尺度卷积的尺度数量
            conv_scale_strides: 各尺度的步长增量
            conv_channels: 各卷积层的通道数
            conv_kernel_sizes: 各卷积层的卷积核大小
            conv_strides: 各卷积层的步长
            wave_dropout: 波形编码器的 dropout 率
            spec_in_channels: 频谱输入通道数
            spec_out_channels: 频谱编码器输出通道数
            spec_temporal_patch: 频谱 temporal patch 大小
            spec_dropout: 频谱编码器的 dropout 率
            activation: 激活函数类
            normalization: 归一化层类
        """
        super().__init__()

        self.d_model = d_model
        self.wave_patch_size = wave_patch_size
        self.spec_temporal_patch = spec_temporal_patch
        self.wave_dropout = nn.Dropout(wave_dropout)
        self.spectro_dropout = nn.Dropout(spec_dropout)

        # ==================== 1. 波形编码器 ====================
        conv_channels = conv_channels.copy()
        conv_channels.append(d_model // wave_patch_size)
        assert conv_channels[-1] * wave_patch_size == d_model, \
            f"通道数 {conv_channels[-1]} * patch_size {wave_patch_size} 必须等于 d_model {d_model}"

        self.wave_convs = nn.Sequential(*[
            MultiScaleConvBlock(
                num_scales=conv_scale_num,
                scale_stride=ss,
                in_channels=inc,
                out_channels=outc,
                kernel_size=kers,
                stride=strd,
                activation=activation,
                normalization=normalization
            )
            for ss, inc, outc, kers, strd in zip(
                conv_scale_strides,
                [in_channels] + conv_channels[:-1],
                conv_channels,
                conv_kernel_sizes,
                conv_strides
            )
        ])

        self.wave_patch = TemporalPatchEmbed(wave_patch_size)

        # ==================== 2. 特征编码器 ====================
        self.feature_head = MLPInputHead(hidden_dim=d_model)

        # ==================== 3. 频谱编码器 ====================
        self.spec_conv = SpectrogramConvEncoder(
            in_channels=spec_in_channels,
            out_channels=spec_out_channels
        )

        self.spec_patch = SpectrogramPatch(patch_size=spec_temporal_patch)

        # ==================== 4. Token 类型嵌入 ====================
        # 0: 波形, 1: 频谱, 2: 特征
        self.type_embedding = nn.Embedding(3, d_model)

    def forward(
            self,
            waveform: torch.Tensor,
            spectrogram: torch.Tensor,
            features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            waveform: [B, 3, T] 原始波形数据
            spectrogram: [B, 3, T_spec] 频谱数据
            features: [B, 15] 特征数据

        Returns:
            [B, num_tokens, d_model] token 序列
        """
        device = waveform.device

        # ---- 波形 tokens ----
        wave_features = self.wave_convs(waveform)  # [B, C, T']
        wave_tokens = self.wave_patch(wave_features)  # [B, N_wave, d_model]
        wave_tokens = wave_tokens + self.type_embedding(
            torch.zeros(wave_tokens.size(1), device=device, dtype=torch.long)
        )

        # ---- 频谱 tokens ----
        spec_features = self.spec_conv(spectrogram)  # [B, C, T_spec']
        spec_tokens = self.spec_patch(spec_features)  # [B, N_spec, d_model]
        spec_tokens = spec_tokens.transpose(1, 2)
        spec_tokens = spec_tokens + self.type_embedding(
            torch.ones(spec_tokens.size(1), device=device, dtype=torch.long)
        )

        # ---- 特征 token ----
        feature_token = self.feature_head(features)  # [B, 1, d_model]
        feature_token = feature_token + self.type_embedding(
            torch.full((1,), 2, device=device, dtype=torch.long)
        )

        # ---- 拼接所有 tokens ----
        tokens = torch.cat([feature_token, spec_tokens, wave_tokens], dim=1)

        return tokens




class WaveformTokenizer(nn.Module):
    """仅使用波形数据的轻量级分词器"""

    def __init__(
            self,
            d_model: int = 1024,
            in_channels: int = 3,
            wave_patch_size: int = 4,
            conv_scale_num: int = 4,
            conv_scale_strides: List[int] = [8, 6, 4, 2],
            conv_channels: List[int] = [16, 48, 96],
            conv_kernel_sizes: List[int] = [16, 12, 6, 1],
            conv_strides: List[int] = [2, 2, 2, 1],
            wave_dropout: float = 0.2,
            activation: nn.Module = nn.GELU,
            normalization: nn.Module = nn.BatchNorm1d,
    ):
        """
        Args:
            d_model: Transformer 模型维度
            in_channels: 波形输入通道数
            wave_patch_size: 波形 patch 大小
            conv_scale_num: 多尺度卷积的尺度数量
            conv_scale_strides: 各尺度的步长增量
            conv_channels: 各卷积层的通道数
            conv_kernel_sizes: 各卷积层的卷积核大小
            conv_strides: 各卷积层的步长
            wave_dropout: 波形编码器的 dropout 率
            activation: 激活函数类
            normalization: 归一化层类
        """
        super().__init__()

        self.d_model = d_model
        self.wave_patch_size = wave_patch_size
        self.wave_dropout = nn.Dropout(wave_dropout)

        self.type_embedding = nn.Embedding(1, d_model)

        # 波形编码器
        conv_channels = conv_channels.copy()
        conv_channels.append(d_model // wave_patch_size)
        assert conv_channels[-1] * wave_patch_size == d_model

        self.wave_convs = nn.Sequential(*[
            MultiScaleConvBlock(
                num_scales=conv_scale_num,
                scale_stride=ss,
                in_channels=inc,
                out_channels=outc,
                kernel_size=kers,
                stride=strd,
                activation=activation,
                normalization=normalization
            )
            for ss, inc, outc, kers, strd in zip(
                conv_scale_strides,
                [in_channels] + conv_channels[:-1],
                conv_channels,
                conv_kernel_sizes,
                conv_strides
            )
        ])

        self.wave_patch = TemporalPatchEmbed(wave_patch_size)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: [B, 3, T] 原始波形数据

        Returns:
            [B, num_tokens, d_model] token 序列
        """
        device = waveform.device

        wave_features = self.wave_convs(waveform)  # [B, C, T']
        wave_tokens = self.wave_patch(wave_features)  # [B, N_wave, d_model]
        wave_tokens = wave_tokens + self.type_embedding(
            torch.zeros(wave_tokens.size(1), device=device, dtype=torch.long)
        )

        return wave_tokens


class BTokenizer(nn.Module):
    """
    多模态回归任务分词器。
    将波形、频谱和特征数据编码为 Transformer 可接受的 token 序列。
    """

    def __init__(
            self,
            d_model: int = 1024,
            in_channels: int = 3,
            wave_patch_size: int = 4,
            conv_scale_num: int = 4,
            conv_scale_strides: List[int] = [8, 6, 4, 2],
            conv_channels: List[int] = [16, 48, 96],
            conv_kernel_sizes: List[int] = [16, 12, 6, 1],
            conv_strides: List[int] = [2, 2, 2, 1],
            wave_dropout: float = 0.2,

            activation: nn.Module = nn.GELU,
            normalization: nn.Module = nn.BatchNorm1d,
    ):
        """
        Args:
            d_model: Transformer 模型维度
            in_channels: 波形输入通道数
            wave_patch_size: 波形 patch 大小
            conv_scale_num: 多尺度卷积的尺度数量
            conv_scale_strides: 各尺度的步长增量
            conv_channels: 各卷积层的通道数
            conv_kernel_sizes: 各卷积层的卷积核大小
            conv_strides: 各卷积层的步长
            wave_dropout: 波形编码器的 dropout 率

            activation: 激活函数类
            normalization: 归一化层类
        """
        super().__init__()

        self.d_model = d_model
        self.wave_patch_size = wave_patch_size
        self.wave_dropout = nn.Dropout(wave_dropout)

        # ==================== 1. 波形编码器 ====================
        conv_channels = conv_channels.copy()
        conv_channels.append(d_model // wave_patch_size)
        assert conv_channels[-1] * wave_patch_size == d_model, \
            f"通道数 {conv_channels[-1]} * patch_size {wave_patch_size} 必须等于 d_model {d_model}"

        self.wave_convs = nn.Sequential(*[
            MultiScaleConvBlock(
                num_scales=conv_scale_num,
                scale_stride=ss,
                in_channels=inc,
                out_channels=outc,
                kernel_size=kers,
                stride=strd,
                activation=activation,
                normalization=normalization
            )
            for ss, inc, outc, kers, strd in zip(
                conv_scale_strides,
                [in_channels] + conv_channels[:-1],
                conv_channels,
                conv_kernel_sizes,
                conv_strides
            )
        ])

        self.wave_patch = TemporalPatchEmbed(wave_patch_size)

        # ==================== 2. 特征编码器 ====================
        self.feature_head = MLPInputHead(hidden_dim=d_model)

        # ==================== 4. Token 类型嵌入 ====================
        # 0: 波形, 1: 频谱, 2: 特征
        self.type_embedding = nn.Embedding(3, d_model)

    def forward(
            self,
            waveform: torch.Tensor,
            features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            waveform: [B, 3, T] 原始波形数据
            spectrogram: [B, 3, T_spec] 频谱数据
            features: [B, 15] 特征数据

        Returns:
            [B, num_tokens, d_model] token 序列
        """
        device = waveform.device

        # ---- 波形 tokens ----
        wave_features = self.wave_convs(waveform)  # [B, C, T']
        wave_tokens = self.wave_patch(wave_features)  # [B, N_wave, d_model]
        wave_tokens = wave_tokens + self.type_embedding(
            torch.zeros(wave_tokens.size(1), device=device, dtype=torch.long)
        )

        # ---- 特征 token ----
        feature_token = self.feature_head(features)  # [B, 1, d_model]
        feature_token = feature_token + self.type_embedding(
            torch.full((1,), 2, device=device, dtype=torch.long)
        )

        # ---- 拼接所有 tokens ----
        tokens = torch.cat([feature_token, wave_tokens], dim=1)

        return tokens

class CTokenizer(nn.Module):
    """
    多模态回归任务分词器。
    将波形、频谱和特征数据编码为 Transformer 可接受的 token 序列。
    """

    def __init__(
            self,
            d_model: int = 1024,
            in_channels: int = 3,
            wave_patch_size: int = 4,
            conv_scale_num: int = 4,
            conv_scale_strides: List[int] = [8, 6, 4, 2],
            conv_channels: List[int] = [16, 48, 96],
            conv_kernel_sizes: List[int] = [16, 12, 6, 1],
            conv_strides: List[int] = [2, 2, 2, 1],
            wave_dropout: float = 0.2,
            spec_in_channels: int = 3,
            spec_out_channels: int = 256,
            spec_temporal_patch: int = 4,
            spec_dropout: float = 0.2,
            activation: nn.Module = nn.GELU,
            normalization: nn.Module = nn.BatchNorm1d,
    ):
        """
        Args:
            d_model: Transformer 模型维度
            in_channels: 波形输入通道数
            wave_patch_size: 波形 patch 大小
            conv_scale_num: 多尺度卷积的尺度数量
            conv_scale_strides: 各尺度的步长增量
            conv_channels: 各卷积层的通道数
            conv_kernel_sizes: 各卷积层的卷积核大小
            conv_strides: 各卷积层的步长
            wave_dropout: 波形编码器的 dropout 率
            spec_in_channels: 频谱输入通道数
            spec_out_channels: 频谱编码器输出通道数
            spec_temporal_patch: 频谱 temporal patch 大小
            spec_dropout: 频谱编码器的 dropout 率
            activation: 激活函数类
            normalization: 归一化层类
        """
        super().__init__()

        self.d_model = d_model
        self.wave_patch_size = wave_patch_size
        self.spec_temporal_patch = spec_temporal_patch
        self.wave_dropout = nn.Dropout(wave_dropout)
        self.spectro_dropout = nn.Dropout(spec_dropout)

        # ==================== 1. 波形编码器 ====================
        conv_channels = conv_channels.copy()
        conv_channels.append(d_model // wave_patch_size)
        assert conv_channels[-1] * wave_patch_size == d_model, \
            f"通道数 {conv_channels[-1]} * patch_size {wave_patch_size} 必须等于 d_model {d_model}"

        self.wave_convs = nn.Sequential(*[
            MultiScaleConvBlock(
                num_scales=conv_scale_num,
                scale_stride=ss,
                in_channels=inc,
                out_channels=outc,
                kernel_size=kers,
                stride=strd,
                activation=activation,
                normalization=normalization
            )
            for ss, inc, outc, kers, strd in zip(
                conv_scale_strides,
                [in_channels] + conv_channels[:-1],
                conv_channels,
                conv_kernel_sizes,
                conv_strides
            )
        ])

        self.wave_patch = TemporalPatchEmbed(wave_patch_size)

        # ==================== 2. 特征编码器 ====================

        # ==================== 3. 频谱编码器 ====================
        self.spec_conv = SpectrogramConvEncoder(
            in_channels=spec_in_channels,
            out_channels=spec_out_channels
        )

        self.spec_patch = SpectrogramPatch(patch_size=spec_temporal_patch)

        # ==================== 4. Token 类型嵌入 ====================
        # 0: 波形, 1: 频谱, 2: 特征
        self.type_embedding = nn.Embedding(3, d_model)

    def forward(
            self,
            waveform: torch.Tensor,
            spectrogram: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            waveform: [B, 3, T] 原始波形数据
            spectrogram: [B, 3, T_spec] 频谱数据
            features: [B, 15] 特征数据

        Returns:
            [B, num_tokens, d_model] token 序列
        """
        device = waveform.device

        # ---- 波形 tokens ----
        wave_features = self.wave_convs(waveform)  # [B, C, T']
        wave_tokens = self.wave_patch(wave_features)  # [B, N_wave, d_model]
        wave_tokens = wave_tokens + self.type_embedding(
            torch.zeros(wave_tokens.size(1), device=device, dtype=torch.long)
        )

        # ---- 频谱 tokens ----
        spec_features = self.spec_conv(spectrogram)  # [B, C, T_spec']
        spec_tokens = self.spec_patch(spec_features)  # [B, N_spec, d_model]
        spec_tokens = spec_tokens.transpose(1, 2)
        spec_tokens = spec_tokens + self.type_embedding(
            torch.ones(spec_tokens.size(1), device=device, dtype=torch.long)
        )

        # ---- 拼接所有 tokens ----
        tokens = torch.cat([spec_tokens, wave_tokens], dim=1)

        return tokens


# ============================================================================
# 第二部分：BART 大语言模型模块 (支持 LoRA 微调)
# ============================================================================

def configure_lora(
        target_modules: List[str],
        rank: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        bias: str = "lora_only",
        task_type: str = "FEATURE_EXTRACTION"
) -> LoraConfig:
    """
    配置 LoRA (Low-Rank Adaptation) 参数。

    Args:
        target_modules: 需要应用 LoRA 的模块名称列表
        rank: LoRA 秩
        lora_alpha: LoRA 缩放因子
        lora_dropout: LoRA dropout 率
        bias: 偏置训练策略
        task_type: 任务类型

    Returns:
        LoRA 配置对象
    """
    return LoraConfig(
        target_modules=target_modules,
        r=rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type,
    )


def generate_bart_target_modules(target_layers: List[int]) -> List[str]:
    """
    为 BART 模型的指定层生成 LoRA 目标模块名称。

    Args:
        target_layers: 需要进行微调的层索引列表

    Returns:
        目标模块名称列表
    """
    target_modules = []
    for layer_idx in target_layers:
        # 自注意力层模块
        target_modules.extend([
            f"layers.{layer_idx}.self_attn.q_proj",
            f"layers.{layer_idx}.self_attn.k_proj",
            f"layers.{layer_idx}.self_attn.v_proj",
            f"layers.{layer_idx}.self_attn.out_proj"
        ])
        # 前馈网络层模块
        target_modules.extend([
            f"layers.{layer_idx}.fc1",
            f"layers.{layer_idx}.fc2"
        ])
    return target_modules


class BARTEncoderBlock(nn.Module):
    """
    BART 编码器模块，支持 LoRA 微调和层截取。

    该模块加载预训练的 BART 模型，可选择性地截取部分层，
    并对指定层应用 LoRA 进行参数高效微调。
    """

    def __init__(
            self,
            start_layer: int = 0,
            end_layer: int = 4,  # 默认使用4层
            d_model: int = 1024,
            lora_config: Optional[LoraConfig] = None,
            use_pretrained: bool = True,
            freeze_base: bool = True,
            model_path: str = BART_MODEL_PATH
    ):
        """
        Args:
            start_layer: 起始层索引（包含）
            end_layer: 结束层索引（不包含）
            d_model: 模型维度
            lora_config: LoRA 配置，若为 None 则不使用 LoRA
            use_pretrained: 是否使用预训练权重
            freeze_base: 是否冻结基础模型参数（仅训练 LoRA）
            model_path: 预训练模型路径
        """
        super().__init__()

        self.use_pretrained = use_pretrained
        self.freeze_base = freeze_base
        self.lora_config = lora_config
        self.hidden_size = d_model

        # 加载预训练 BART 模型（仅编码器部分）
        if use_pretrained:
            print(f"正在加载本地 BART 模型：{model_path}")
            self.bart_encoder = BartModel.from_pretrained(
                pretrained_model_name_or_path=model_path,
                local_files_only=True,
            ).encoder
        else:
            warnings.warn("使用随机初始化的 BART Encoder，建议优先使用预训练模型！")
            self.bart_encoder = BartModel(BartConfig()).encoder

        # 截取指定范围的编码器层
        self.bart_encoder.layers = self.bart_encoder.layers[start_layer:end_layer]
        print(f"已截取 BART Encoder layers：{start_layer} ~ {end_layer - 1}（共 {end_layer - start_layer} 层）")

        # 验证 LoRA 目标模块是否存在
        if lora_config is not None:
            self._validate_lora_target_modules()

        # 应用 LoRA 并冻结参数
        if freeze_base and use_pretrained and lora_config is not None:
            self.bart_encoder = get_peft_model(self.bart_encoder, lora_config)
            self._freeze_base_parameters()
            print("LoRA 配置完成：仅训练 LoRA 层，冻结主模型（除层归一化/位置嵌入）")
            self.bart_encoder.print_trainable_parameters()

    def _validate_lora_target_modules(self) -> None:
        """验证 LoRA 目标模块是否存在于 BART Encoder 中"""
        model_module_names = set()
        for name, _ in self.bart_encoder.named_parameters():
            module_name = ".".join(name.split(".")[:-1]) if name.endswith((".weight", ".bias")) else name
            model_module_names.add(module_name)

        missing_modules = []
        for target in self.lora_config.target_modules:
            if target not in model_module_names:
                missing_modules.append(target)

        if missing_modules:
            raise ValueError(
                f"LoRA 目标模块缺失！以下模块未找到：\n{missing_modules}\n"
                f"当前可用模块示例：{list(model_module_names)[:10]}...\n"
                "解决方案：1. 确认目标层索引是否正确；2. 检查层索引是否在模型范围内"
            )
        print(f"LoRA 目标模块验证通过（共 {len(self.lora_config.target_modules)} 个模块）")

    def _freeze_base_parameters(self) -> None:
        """冻结基础模型参数，仅保留 LoRA、层归一化和位置嵌入可训练"""
        total_params = 0
        trainable_params = 0

        for name, param in self.bart_encoder.named_parameters():
            total_params += param.numel()

            # 冻结词嵌入层
            if "embed_tokens" in name:
                param.requires_grad = False
            # 解冻 LoRA、层归一化、位置嵌入
            elif any(keyword in name for keyword in ["lora", "layernorm", "layer_norm", "embed_positions"]):
                param.requires_grad = True
                trainable_params += param.numel()
            else:
                param.requires_grad = False

        print(f"可训练参数: {trainable_params} / {total_params} ({trainable_params / total_params * 100:.4f}%)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: [B, T, d_model] 输入 token 序列

        Returns:
            [B, T, d_model] 编码后的特征序列
        """
        assert x.dim() == 3 and x.size(-1) == self.hidden_size, \
            f"输入维度错误！需 [B, T, {self.hidden_size}]，当前：{x.shape}"

        # 构建注意力掩码（BART 需要）
        attention_mask = torch.ones(
            x.shape[:2],
            device=x.device,
            dtype=torch.long,
        )

        # BART Encoder 前向传播
        x = self.bart_encoder(
            inputs_embeds=x,
            attention_mask=attention_mask
        ).last_hidden_state

        return x


# ============================================================================
# 第三部分：任务输出头 (Task Heads)
# ============================================================================

class PPhaseHead(nn.Module):
    """
    P 波拾取头，通过渐进上采样生成逐点概率。

    输入: [B, C, L'] (PatchUnembed 后)
    输出: [B, 1, T] 每个时间点的 P 波概率
    """

    def __init__(
            self,
            feature_channels: int = 256,
            hidden_channels: int = 64,
            out_channels: int = 1,
            patch_size: int = 4,
            num_upsample_stages: int = 4,
            dropout_prob: float = 0.1,
            out_activation = nn.Sigmoid
    ):
        """
        Args:
            feature_channels: 输入特征通道数
            hidden_channels: 隐藏层通道数
            out_channels: 输出通道数
            patch_size: patch 大小
            num_upsample_stages: 上采样阶段数
            dropout_prob: dropout 概率
        """
        super().__init__()

        self.patch_size = patch_size
        self.num_upsample_stages = num_upsample_stages

        # 通道压缩
        self.channel_reduce = nn.Conv1d(feature_channels, hidden_channels, kernel_size=1, bias=False)

        # 渐进上采样层
        self.upsample_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Dropout1d(dropout_prob)
            )
            for _ in range(num_upsample_stages)
        ])

        # 输出层
        self.output_conv = nn.Conv1d(hidden_channels, out_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def _compute_upsample_sizes(self, in_length: int, out_length: int) -> List[int]:
        """计算每个上采样阶段的目标尺寸"""
        sizes = [out_length] * self.num_upsample_stages
        if self.num_upsample_stages == 0:
            return sizes

        factor = (out_length / in_length) ** (1 / self.num_upsample_stages)
        for i in range(self.num_upsample_stages - 2, -1, -1):
            sizes[i] = int(sizes[i + 1] / factor)
        return sizes

    def forward(self, features: torch.Tensor, input_waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, C, L'] 编码后的特征
            input_waveform: [B, 3, T] 原始波形（用于获取目标长度）

        Returns:
            [B, 1, T] P 波概率
        """
        x = self.channel_reduce(features)
        target_length = input_waveform.shape[-1]
        in_length = x.size(-1)

        up_sizes = self._compute_upsample_sizes(in_length, target_length)

        for i, layer in enumerate(self.upsample_layers):
            x = F.interpolate(x, size=up_sizes[i], mode="linear", align_corners=False)
            x = layer(x)

        x = self.output_conv(x)
        return self.sigmoid(x)


class MagnitudeHead(nn.Module):
    """震级估计头，输出震级值（0-8 范围）"""

    def __init__(self, feature_channels: int, out_activation: Optional[nn.Module] = None):
        """
        Args:
            feature_channels: 输入特征通道数
            out_activation: 输出激活函数，默认使用 ScaledSigmoid(scale=8)
        """
        super().__init__()

        self.convs = nn.ModuleList([
            nn.Conv1d(feature_channels, feature_channels, kernel_size=8, stride=2)
            for _ in range(2)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten(1, -1)
        self.linear = nn.Linear(feature_channels, 1)

        # 输出激活函数
        if out_activation is None:
            out_activation = partial(ScaledActivation, act_layer=nn.Sigmoid, scale_factor=8)
        self.out_activation = out_activation()

        # 初始化
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, features: torch.Tensor, _: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            features: [B, C, L'] 编码后的特征

        Returns:
            [B, 1] 震级估计值
        """
        x = features
        for conv in self.convs:
            x = conv(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.linear(x)
        return self.out_activation(x)


class DistanceHead(nn.Module):
    """
    震中距估计头，输出对数距离（log-distance）。
    输出范围: [-8, 7]，对应实际距离 0.001 ~ 200 km。
    """

    def __init__(
            self,
            feature_channels: int,
            output_range: Tuple[float, float] = (-8.0, 7.0),
            out_activation =None
    ):
        """
        Args:
            feature_channels: 输入特征通道数
            output_range: 输出范围 (min, max)
        """
        super().__init__()

        self.min_val, self.max_val = output_range
        self.mid_val = (self.min_val + self.max_val) / 2.0
        self.half_range = (self.max_val - self.min_val) / 2.0

        self.convs = nn.ModuleList([
            nn.Conv1d(feature_channels, feature_channels, kernel_size=8, stride=2),
            nn.Conv1d(feature_channels, feature_channels, kernel_size=8, stride=2)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten(start_dim=1)
        self.linear = nn.Linear(feature_channels, 1)

        # 初始化偏置，使初始输出接近 log(10) ≈ 2.3
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 2.3)

    def forward(self, features: torch.Tensor, _: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            features: [B, C, L'] 编码后的特征

        Returns:
            [B, 1] 对数距离估计值
        """
        x = features
        for conv in self.convs:
            x = F.relu(conv(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.linear(x)

        # 使用 tanh 进行软边界约束
        return self.mid_val + self.half_range * torch.tanh(x)


class AzimuthHead(nn.Module):
    """方位角估计头，输出正弦和余弦值"""

    def __init__(self, feature_channels: int, out_activation: nn.Module = nn.Tanh):
        """
        Args:
            feature_channels: 输入特征通道数
            out_activation: 输出激活函数
        """
        super().__init__()

        self.convs = nn.ModuleList([
            nn.Conv1d(feature_channels, feature_channels, kernel_size=8, stride=4)
            for _ in range(2)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten(1, -1)
        self.linear = nn.Linear(feature_channels, 2)
        self.out_activation = out_activation()

        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, features: torch.Tensor, _: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [B, C, L'] 编码后的特征

        Returns:
            (sin, cos) 方位角的正弦和余弦值
        """
        x = features
        for conv in self.convs:
            x = conv(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.out_activation(x)
        return x[:, :1], x[:, 1:]


# ============================================================================
# 第四部分：完整模型
# ============================================================================

class EarthquakeEarlyWarningLLM(nn.Module):
    """
    地震预警大语言模型。

    该模型集成了多模态输入头、BART 编码器和任务特定输出头，
    支持地震波 P 波拾取、震级估计、震中距估计和方位角估计等任务。
    """

    def __init__(
            self,
            d_model: int = 1024,
            llm_layers: int = 4,
            use_gradient_checkpointing: bool = True,
            # 波形编码器参数
            in_channels: int = 3,
            wave_patch_size: int = 4,
            conv_scale_num: int = 4,
            conv_scale_strides: List[int] = [8, 6, 4, 2],
            conv_channels: List[int] = [16, 48, 96],
            conv_kernel_sizes: List[int] = [16, 12, 6, 1],
            conv_strides: List[int] = [2, 2, 2, 1],
            wave_dropout: float = 0.2,
            # 频谱编码器参数
            spec_in_channels: int = 3,
            spec_out_channels: int = 256,
            spec_temporal_patch: int = 4,
            spec_dropout: float = 0.2,
            # 通用参数
            activation: nn.Module = nn.GELU,
            normalization: nn.Module = nn.BatchNorm1d,
            use_waveform_only: bool = False,

            # 消融实验专属参数
            use_waveandspec = False,
            use_waveandfea=False,

            # 输出头参数
            reshape_patch_size: int = 4,
            reshape_out_dim: int = 256,
            output_head_class: nn.Module = MagnitudeHead,
            output_activation: nn.Module = nn.Tanh,

            **kwargs
    ):
        """
        Args:
            d_model: Transformer 模型维度
            llm_layers: LLM 使用的层数
            use_gradient_checkpointing: 是否使用梯度检查点（节省显存）
            in_channels: 波形输入通道数
            wave_patch_size: 波形 patch 大小
            conv_scale_num: 多尺度卷积尺度数
            conv_scale_strides: 各尺度步长增量
            conv_channels: 各卷积层通道数
            conv_kernel_sizes: 各卷积层卷积核大小
            conv_strides: 各卷积层步长
            wave_dropout: 波形编码器 dropout 率
            spec_in_channels: 频谱输入通道数
            spec_out_channels: 频谱编码器输出通道数
            spec_temporal_patch: 频谱 temporal patch 大小
            spec_dropout: 频谱编码器 dropout 率
            activation: 激活函数类
            normalization: 归一化层类
            use_waveform_only: 是否仅使用波形数据（P 波拾取任务）
            reshape_patch_size: 输出重塑的 patch 大小
            reshape_out_dim: 输出重塑的特征维度
            output_head_class: 输出头类
            output_activation: 输出激活函数
        """
        super().__init__(**kwargs)

        self.use_gradient_checkpointing = use_gradient_checkpointing

        # ==================== 输入头 ====================
        if use_waveform_only:
            self.input_tokenizer = WaveformTokenizer(
                d_model=d_model,
                in_channels=in_channels,
                wave_patch_size=wave_patch_size,
                conv_scale_num=conv_scale_num,
                conv_scale_strides=conv_scale_strides,
                conv_channels=conv_channels,
                conv_kernel_sizes=conv_kernel_sizes,
                conv_strides=conv_strides,
                wave_dropout=wave_dropout,
                activation=activation,
                normalization=normalization,
            )
        elif use_waveandfea:  # 波形 + 特征
            self.input_tokenizer = BTokenizer(
                d_model=d_model,
                in_channels=in_channels,
                wave_patch_size=wave_patch_size,
                conv_scale_num=conv_scale_num,
                conv_scale_strides=conv_scale_strides,
                conv_channels=conv_channels,
                conv_kernel_sizes=conv_kernel_sizes,
                conv_strides=conv_strides,
                wave_dropout=wave_dropout,
                activation=activation,
                normalization=normalization,
            )
        elif use_waveandspec:  # 波形 + 频谱
            self.input_tokenizer = CTokenizer(
                d_model=d_model,
                in_channels=in_channels,
                wave_patch_size=wave_patch_size,
                conv_scale_num=conv_scale_num,
                conv_scale_strides=conv_scale_strides,
                conv_channels=conv_channels,
                conv_kernel_sizes=conv_kernel_sizes,
                conv_strides=conv_strides,
                wave_dropout=wave_dropout,
                spec_in_channels=spec_in_channels,
                spec_out_channels=spec_out_channels,
                spec_temporal_patch=spec_temporal_patch,
                spec_dropout=spec_dropout,
                activation=activation,
                normalization=normalization,
            )
        else:  # 波形 + 频谱 + 特征 (默认)
            self.input_tokenizer = RegressionTokenizer(
                d_model=d_model,
                in_channels=in_channels,
                wave_patch_size=wave_patch_size,
                conv_scale_num=conv_scale_num,
                conv_scale_strides=conv_scale_strides,
                conv_channels=conv_channels,
                conv_kernel_sizes=conv_kernel_sizes,
                conv_strides=conv_strides,
                wave_dropout=wave_dropout,
                spec_in_channels=spec_in_channels,
                spec_out_channels=spec_out_channels,
                spec_temporal_patch=spec_temporal_patch,
                spec_dropout=spec_dropout,
                activation=activation,
                normalization=normalization,
            )

        # ==================== BART 编码器 ====================
        # 动态生成目标层索引（使用 llm_layers 参数）
        target_layers = list(range(llm_layers))
        target_modules = generate_bart_target_modules(target_layers)
        lora_config = configure_lora(
            target_modules=target_modules,
            rank=16,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="lora_only",
            task_type="FEATURE_EXTRACTION"
        )

        self.bart_encoder = BARTEncoderBlock(
            start_layer=0,
            end_layer=llm_layers,
            d_model=d_model,
            lora_config=lora_config,
            use_pretrained=True,
            freeze_base=True,
        )

        # ==================== 输出重塑 ====================
        self.patch_recover = PatchRecover(
            patch_size=reshape_patch_size,
            out_dim=reshape_out_dim,
        )

        # ==================== 输出头 ====================
        self.output_head = output_head_class(
            feature_channels=reshape_out_dim,
            out_activation=output_activation,
        )

    # def forward(
    #         self,
    #         waveform: torch.Tensor,
    #         spectrogram: Optional[torch.Tensor] = None,
    #         features: Optional[torch.Tensor] = None
    # ) -> torch.Tensor:
    #     """
    #     前向传播。
    #
    #     Args:
    #         waveform: [B, 3, T] 原始波形数据
    #         spectrogram: [B, 3, T_spec] 频谱数据（可选，仅多模态模式使用）
    #         features: [B, 15] 特征数据（可选，仅多模态模式使用）
    #
    #     Returns:
    #         任务特定的输出，形状取决于输出头
    #     """
    #     # 输入编码
    #     if spectrogram is None and features is None:
    #         tokens = self.input_tokenizer(waveform)
    #     else:
    #         tokens = self.input_tokenizer(waveform, spectrogram, features)
    #
    #     # BART 编码
    #     encoded = self.bart_encoder(tokens)
    #
    #     # 输出重塑
    #     recovered = self.patch_recover(encoded)
    #
    #     # 任务头
    #     output = self.output_head(recovered, waveform)
    #
    #     return output
    def forward(
            self,
            waveform: torch.Tensor,
            spectrogram: Optional[torch.Tensor] = None,
            features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播。

        Args:
            waveform: [B, 3, T] 原始波形数据
            spectrogram: [B, 3, T_spec] 频谱数据（可选，仅多模态模式使用）
            features: [B, 15] 特征数据（可选，仅多模态模式使用）

        Returns:
            任务特定的输出，形状取决于输出头
        """
        # 根据 tokenizer 类型选择输入参数
        tokenizer_type = type(self.input_tokenizer).__name__

        if tokenizer_type == 'WaveformTokenizer':
            # 仅波形，只传 waveform
            tokens = self.input_tokenizer(waveform)
        elif tokenizer_type == 'BTokenizer':
            # 波形 + 特征，传入 waveform 和 features
            tokens = self.input_tokenizer(waveform, features)
        elif tokenizer_type == 'CTokenizer':
            # 波形 + 频谱，传入 waveform 和 spectrogram
            tokens = self.input_tokenizer(waveform, spectrogram)
        else:  # RegressionTokenizer 或其他
            # 全模态，传入所有数据
            tokens = self.input_tokenizer(waveform, spectrogram, features)

        # BART 编码
        encoded = self.bart_encoder(tokens)

        # 输出重塑
        recovered = self.patch_recover(encoded)

        # 任务头
        output = self.output_head(recovered, waveform)

        return output


# ============================================================================
# 模型工厂函数
# ============================================================================

def EEW_LMBART_azi() -> EarthquakeEarlyWarningLLM:
    """创建方位角估计模型"""
    return EarthquakeEarlyWarningLLM(
        output_head_class=AzimuthHead,
        reshape_out_dim=256,
        output_activation=nn.Tanh,
    )


def EEW_LMBART_mag() -> EarthquakeEarlyWarningLLM:
    """创建震级估计模型"""
    return EarthquakeEarlyWarningLLM(
        output_head_class=MagnitudeHead,
        reshape_out_dim=256,
        output_activation=nn.Sigmoid,
    )


def EEW_LMBART_dis() -> EarthquakeEarlyWarningLLM:
    """创建震中距估计模型"""
    return EarthquakeEarlyWarningLLM(
        output_head_class=DistanceHead,
        reshape_out_dim=256,
        output_activation=nn.Tanh,
    )


def EEW_LMBART_pp() -> EarthquakeEarlyWarningLLM:
    """创建 P 波拾取模型"""
    return EarthquakeEarlyWarningLLM(
        llm_layers=4,
        use_waveform_only=True,
        output_head_class=PPhaseHead,
        reshape_out_dim=256,
    )

# ============================================================================
# 消融实验
# ============================================================================


def EEW_LMBART_aziA() -> EarthquakeEarlyWarningLLM:
    """创建方位角估计模型"""
    return EarthquakeEarlyWarningLLM(
        use_waveform_only=True,
        # 消融实验专属参数
        use_waveandspec = False,
        use_waveandfea = False,
        output_head_class=AzimuthHead,
        reshape_out_dim=256,
        output_activation=nn.Tanh,
    )


def EEW_LMBART_magA() -> EarthquakeEarlyWarningLLM:
    """创建震级估计模型"""
    return EarthquakeEarlyWarningLLM(
        use_waveform_only=True,
        # 消融实验专属参数
        use_waveandspec=False,
        use_waveandfea=False,
        output_head_class=MagnitudeHead,
        reshape_out_dim=256,
        output_activation=nn.Sigmoid,
    )


def EEW_LMBART_disA() -> EarthquakeEarlyWarningLLM:
    """创建震中距估计模型"""
    return EarthquakeEarlyWarningLLM(
        use_waveform_only=True,
        # 消融实验专属参数
        use_waveandspec=False,
        use_waveandfea=False,
        output_head_class=DistanceHead,
        reshape_out_dim=256,
        output_activation=nn.Tanh,
    )


def EEW_LMBART_aziB() -> EarthquakeEarlyWarningLLM:
    """创建方位角估计模型"""
    return EarthquakeEarlyWarningLLM(
        use_waveform_only=False,
        # 消融实验专属参数
        use_waveandspec=True,
        use_waveandfea=False,
        output_head_class=AzimuthHead,
        reshape_out_dim=256,
        output_activation=nn.Tanh,
    )


def EEW_LMBART_magB() -> EarthquakeEarlyWarningLLM:
    """创建震级估计模型"""
    return EarthquakeEarlyWarningLLM(
        use_waveform_only=False,
        # 消融实验专属参数
        use_waveandspec=True,
        use_waveandfea=False,
        output_head_class=MagnitudeHead,
        reshape_out_dim=256,
        output_activation=nn.Sigmoid,
    )


def EEW_LMBART_disB() -> EarthquakeEarlyWarningLLM:
    """创建震中距估计模型"""
    return EarthquakeEarlyWarningLLM(
        use_waveform_only=False,
        # 消融实验专属参数
        use_waveandspec=True,
        use_waveandfea=False,
        output_head_class=DistanceHead,
        reshape_out_dim=256,
        output_activation=nn.Tanh,
    )


def EEW_LMBART_aziC() -> EarthquakeEarlyWarningLLM:
    """创建方位角估计模型"""
    return EarthquakeEarlyWarningLLM(
        use_waveform_only=False,
        # 消融实验专属参数
        use_waveandspec=False,
        use_waveandfea=True,
        output_head_class=AzimuthHead,
        reshape_out_dim=256,
        output_activation=nn.Tanh,
    )


def EEW_LMBART_magC() -> EarthquakeEarlyWarningLLM:
    """创建震级估计模型"""
    return EarthquakeEarlyWarningLLM(
        use_waveform_only=False,
        # 消融实验专属参数
        use_waveandspec=False,
        use_waveandfea=True,
        output_head_class=MagnitudeHead,
        reshape_out_dim=256,
        output_activation=nn.Sigmoid,
    )


def EEW_LMBART_disC() -> EarthquakeEarlyWarningLLM:
    """创建震中距估计模型"""
    return EarthquakeEarlyWarningLLM(
        use_waveform_only=False,
        # 消融实验专属参数
        use_waveandspec=False,
        use_waveandfea=True,
        output_head_class=DistanceHead,
        reshape_out_dim=256,
        output_activation=nn.Tanh,
    )


# ============================================================================
# 测试代码
# ============================================================================
#
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"使用设备: {device}")
#
#     batch_size = 2
#
#     # 生成测试数据
#     waveform = torch.randn(batch_size, 3, 300, device=device)
#     spectrogram = torch.randn(batch_size, 3, 128, device=device)
#     features = torch.randn(batch_size, 15, device=device)
#
#     # # 测试 P 波拾取模型
#     # print("\n测试 P 波拾取模型...")
#     # model_pp =EEW_LMBART_pp().to(device)
#     # print(model_pp)
#     #
#     # model_pp.eval()
#     # print("\nP波拾取模型初始化成功")
#     #
#     # # 测试前向传播
#     # with torch.no_grad():
#     #     output_pp = model_pp(waveform)
#     #     print(f"P波拾取模型输出形状: {output_pp.shape}")
#
#     # 测试震级估计模型
#     print("\n测试震级估计模型...")
#     model_mag = EEW_LMBART_magC().to(device)
#     with torch.no_grad():
#         output_mag = model_mag(waveform, spectrogram, features)
#         print(f"震级估计模型输出形状: {output_mag.shape}")
#
#     # 测试震中距估计模型
#     print("\n测试震中距估计模型...")
#     model_dis = EEW_LMBART_disC().to(device)
#     with torch.no_grad():
#         output_dis = model_dis(waveform, spectrogram, features)
#         print(f"震中距估计模型输出形状: {output_dis.shape}")
#
#     # 测试方位角估计模型
#     print("\n测试方位角估计模型...")
#     model_azi = EEW_LMBART_aziC().to(device)
#     with torch.no_grad():
#         sin, cos = model_azi(waveform, spectrogram, features)
#         print(f"方位角估计模型输出形状: sin={sin.shape}, cos={cos.shape}")
#
#     print("\n所有测试通过！")