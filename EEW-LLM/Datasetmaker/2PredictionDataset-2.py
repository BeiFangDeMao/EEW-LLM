#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project : LLM
# @File : 加频率和特征的。.py
# @IDE : PyCharm
# @Author : 张嘉南
# @Date : 2025/12/28 下午12:07
# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project : LLM
# @File : 添加时频图和特征值的批次处理脚本（特征独立保存版）.py
# @IDE : PyCharm
# @Author : 张嘉南
# @Date : 2025/12/28 上午9:30

import os
import time
import numpy as np
import h5py
import pandas as pd
from scipy import signal
from scipy import interpolate  # 新增插值模块导入
import shutil
from tqdm import tqdm

# -------------------------- 1. 配置参数 --------------------------
INPUT_H5_PATH = r"G:\Knet\日本3级以上\JKnet_300_P40_test.h5" # 输入HDF5文件路径
OUTPUT_H5_PATH = r"G:\迁移学习\LLM\20251209NewTry\dataset\JKnet_300_P40_test_FS.h5"  # 输出HDF5文件路径
LOG_FILE = r"G:\迁移学习\LLM\20251209NewTry\dataset\JKnet_300_P40_test_FS_log.csv"  # 日志文件路径
BATCH_SIZE = 2000  # 批次处理大小
SAMPLING_RATE = 100  # 采样频率(Hz)，需满足>80Hz


# -------------------------- 2. 工具函数 --------------------------
def load_waveform_from_h5(h5_path: str, file_prefix: str, target_length: int = 300) -> tuple:
    """
    从 HDF5 文件中读取指定 file_prefix 的三分量加速度波形数据
    """
    with h5py.File(h5_path, 'r') as f:
        if file_prefix not in f:
            raise KeyError(f"Dataset '{file_prefix}' not found in {h5_path}")

        data = f[file_prefix][:]
        expected_shape = (target_length, 3)
        if data.shape != expected_shape:
            raise ValueError(f"Unexpected shape {data.shape} for dataset '{file_prefix}'. Expected {expected_shape}.")

        wave_ud = data[:, 0].astype(np.float32)
        wave_ns = data[:, 1].astype(np.float32)
        wave_ew = data[:, 2].astype(np.float32)

        return wave_ud, wave_ns, wave_ew


def fft_waveform_to_200pts(waveform_data, f_samp):
    """
    输入点波形数据和采样频率，输出0.075Hz~40Hz区间均匀采样的128个傅里叶谱幅值数据
    """
    # 1. 输入校验
    if len(waveform_data) != 300:
        raise ValueError("波形数据必须为300点！请检查输入数据长度。")
    if f_samp <= 80:
        raise ValueError("采样频率必须大于80Hz（满足奈奎斯特定理，大于2倍40Hz上限）！")

    # 2. 核心参数定义
    N = 300
    f_min = 0.075
    f_max = 40.0
    n_output = 128

    # 3. FFT计算与幅值校正
    Y_fft = np.fft.fft(waveform_data)
    freq_original = np.fft.fftfreq(N, d=1 / f_samp)
    Y_amp = 2 * np.abs(Y_fft) / N  # 幅值校正

    # 4. 筛选目标频率区间[0.075, 40]Hz的正频率数据
    positive_mask = freq_original >= 0
    freq_pos = freq_original[positive_mask]
    Y_amp_pos = Y_amp[positive_mask]

    target_mask = (freq_pos >= f_min) & (freq_pos <= f_max)
    freq_target = freq_pos[target_mask]
    Y_amp_target = Y_amp_pos[target_mask]

    if len(freq_target) == 0:
        raise RuntimeError("目标频率区间内无有效数据，请检查采样频率或输入波形。")

    # 5. 均匀采样个点
    freq_uniform = np.linspace(f_min, f_max, n_output)
    interp_fun = interpolate.interp1d(
        freq_target, Y_amp_target,
        kind='linear',
        bounds_error=False,
        fill_value=0.0
    )
    Y_amp_uniform = interp_fun(freq_uniform)

    return Y_amp_uniform


def process_fft_components(wave_ud, wave_ns, wave_ew, f_samp=SAMPLING_RATE):
    """处理三分量波形的FFT谱，返回各分量的频率轴和200点幅值谱"""
    # 计算各分量的FFT谱
    amp_ud = fft_waveform_to_200pts(wave_ud, f_samp)
    amp_ns = fft_waveform_to_200pts(wave_ns, f_samp)
    amp_ew = fft_waveform_to_200pts(wave_ew, f_samp)

    # 生成统一的频率轴（0.075~40Hz均匀分布的200点）
    freq_uniform = np.linspace(0.075, 40.0, 200)

    return freq_uniform, amp_ud, amp_ns, amp_ew


def feature_extraction(wave_ud, wave_ns, wave_ew, dt=0.01):
    """从已输入的300长度三分量波形中提取地震特征参数"""
    # 保持原特征提取逻辑不变
    sampling_rate = 1 / dt
    sos = signal.butter(
        N=4,
        Wn=0.075,
        btype='highpass',
        fs=sampling_rate,
        output='sos'
    )

    expected_len = 300
    if len(wave_ud) != expected_len or len(wave_ns) != expected_len or len(wave_ew) != expected_len:
        raise ValueError(
            f"输入波形长度需为{expected_len}！当前长度：UD={len(wave_ud)}, NS={len(wave_ns)}, EW={len(wave_ew)}"
        )

    acc_ud = wave_ud.astype(np.float32)
    acc_ns = wave_ns.astype(np.float32)
    acc_ew = wave_ew.astype(np.float32)

    vel_ud = np.cumsum(acc_ud) * dt
    vel_ns = np.cumsum(acc_ns) * dt
    vel_ew = np.cumsum(acc_ew) * dt

    vel_ud = signal.detrend(vel_ud, type='linear')
    vel_ns = signal.detrend(vel_ns, type='linear')
    vel_ew = signal.detrend(vel_ew, type='linear')

    vel_ud = vel_ud - np.mean(vel_ud)
    vel_ns = vel_ns - np.mean(vel_ns)
    vel_ew = vel_ew - np.mean(vel_ew)

    vel_ud = signal.sosfilt(sos, vel_ud)
    vel_ns = signal.sosfilt(sos, vel_ns)
    vel_ew = signal.sosfilt(sos, vel_ew)

    disp_ud = np.cumsum(vel_ud) * dt
    disp_ns = np.cumsum(vel_ns) * dt
    disp_ew = np.cumsum(vel_ew) * dt

    disp_ud = signal.detrend(disp_ud, type='linear')
    disp_ns = signal.detrend(disp_ns, type='linear')
    disp_ew = signal.detrend(disp_ew, type='linear')

    disp_ud = disp_ud - np.mean(disp_ud)
    disp_ns = disp_ns - np.mean(disp_ns)
    disp_ew = disp_ew - np.mean(disp_ew)

    disp_ud = signal.sosfilt(sos, disp_ud)
    disp_ns = signal.sosfilt(sos, disp_ns)
    disp_ew = signal.sosfilt(sos, disp_ew)

    def calc_3comp_sum(z: np.ndarray, n: np.ndarray, e: np.ndarray) -> np.ndarray:
        return np.sqrt(np.square(z) + np.square(n) + np.square(e))

    acc_3comp = calc_3comp_sum(acc_ud, acc_ns, acc_ew)
    vel_3comp = calc_3comp_sum(vel_ud, vel_ns, vel_ew)
    disp_3comp = calc_3comp_sum(disp_ud, disp_ns, disp_ew)

    Pa = np.max(np.abs(acc_3comp))
    Pv = np.max(np.abs(vel_3comp))
    Pd = np.max(np.abs(disp_3comp))
    Pav = Pa * Pv
    Pad = Pa * Pd
    Pvd = Pv * Pd

    IAA = np.trapezoid(np.abs(acc_3comp), dx=dt)
    IAV = np.trapezoid(np.abs(vel_3comp), dx=dt)
    IAD = np.trapezoid(np.abs(disp_3comp), dx=dt)
    IV2 = np.trapezoid(np.square(vel_3comp), dx=dt)

    Ia = (np.pi / (2 * 9.8 * 100)) * np.trapezoid(np.square(acc_3comp), dx=dt)

    r2 = np.trapezoid(np.square(disp_3comp), dx=dt)
    r = IV2 / r2 if r2 != 0 else 0
    Tc = 2 * np.pi / np.sqrt(r) if r > 0 else 0
    TP = Tc * Pd

    acc_vel_prod = np.abs(acc_3comp * vel_3comp) + 1e-8
    DI = np.max(np.log10(acc_vel_prod))

    Tva = 2 * np.pi * (Pv / Pa) if Pa != 0 else 0

    return np.array([
        Pa, Pv, Pd, Pav, Pad, Pvd, IAA, IAV, IAD, IV2, Ia, Tc, TP, DI, Tva
    ], dtype=np.float32)


def write_log(log_data: dict, log_path: str):
    """将处理日志写入CSV文件"""
    log_df = pd.DataFrame([log_data])
    header = not os.path.exists(log_path)
    log_df.to_csv(log_path, mode='a', index=False, header=header, encoding='utf-8-sig')


# -------------------------- 3. 主处理逻辑 --------------------------
if __name__ == "__main__":
    start_total_time = time.time()
    print(f"=== 添加FFT谱和特征值处理开始（目标长度：300点）===")

    # 创建输出文件（复制输入文件内容）
    if os.path.exists(OUTPUT_H5_PATH):
        os.remove(OUTPUT_H5_PATH)
    shutil.copyfile(INPUT_H5_PATH, OUTPUT_H5_PATH)

    # 获取所有数据集名称
    with h5py.File(INPUT_H5_PATH, 'r') as f:
        all_datasets = list(f.keys())

    total_datasets = len(all_datasets)
    total_batches = (total_datasets + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"总共有 {total_datasets} 个数据集，分 {total_batches} 批处理")

    # 准备日志文件
    log_headers = [
        "序号", "file_prefix", "处理状态", "处理时间", "处理时长(秒)",
        "FFT谱计算状态", "特征值计算状态", "失败原因"  # 日志中时频图改为FFT谱
    ]

    if not os.path.exists(LOG_FILE):
        pd.DataFrame(columns=log_headers).to_csv(LOG_FILE, index=False, encoding='utf-8-sig')

    # 创建进度条
    pbar = tqdm(total=total_datasets, desc="处理数据集", unit="个")

    # 特征名称列表
    feature_names = [
        'Pa', 'Pv', 'Pd', 'Pav', 'Pad', 'Pvd', 'IAA', 'IAV', 'IAD',
        'IV2', 'Ia', 'Tc', 'TP', 'DI', 'Tva'
    ]

    # 处理每个数据集
    for batch_idx in range(total_batches):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min((batch_idx + 1) * BATCH_SIZE, total_datasets)
        batch_datasets = all_datasets[batch_start:batch_end]

        print(f"\n=== 处理第{batch_idx + 1}/{total_batches}批（数据集{batch_start + 1}-{batch_end}）===")

        batch_success_count = 0
        batch_failed_count = 0

        for idx, file_prefix in enumerate(batch_datasets):
            global_idx = batch_start + idx + 1
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            log_data = {
                "序号": global_idx,
                "file_prefix": file_prefix,
                "处理状态": "成功",
                "处理时间": current_time,
                "处理时长(秒)": 0,
                "FFT谱计算状态": "成功",  # 日志状态更新
                "特征值计算状态": "成功",
                "失败原因": ""
            }

            try:
                # 读取波形数据
                start_time = time.time()
                wave_ud, wave_ns, wave_ew = load_waveform_from_h5(INPUT_H5_PATH, file_prefix)
                time_taken = time.time() - start_time

                # 计算FFT谱（替换原STFT计算）
                start_time = time.time()
                freq, amp_ud, amp_ns, amp_ew = process_fft_components(wave_ud, wave_ns, wave_ew)
                time_taken_fft = time.time() - start_time

                # 计算特征值
                start_time = time.time()
                features = feature_extraction(wave_ud, wave_ns, wave_ew)
                time_taken_features = time.time() - start_time

                # 写入FFT谱数据（替换原时频图写入）
                with h5py.File(OUTPUT_H5_PATH, 'r+') as h5_file:
                    # 创建FFT谱数据组
                    if 'fft_spectrum' not in h5_file:
                        h5_file.create_group('fft_spectrum')

                    # 保存FFT谱数据
                    fft_group = h5_file['fft_spectrum']
                    if file_prefix not in fft_group:
                        fft_group.create_group(file_prefix)

                    fft_prefix = fft_group[file_prefix]
                    # 保存频率轴（三分量共用同一频率轴）
                    fft_prefix.create_dataset('frequency', data=freq)
                    # 保存各分量幅值谱
                    fft_prefix.create_dataset('amp_ud', data=amp_ud)
                    fft_prefix.create_dataset('amp_ns', data=amp_ns)
                    fft_prefix.create_dataset('amp_ew', data=amp_ew)

                # 保存特征值为独立属性（保持不变）
                with h5py.File(OUTPUT_H5_PATH, 'r+') as h5_file:
                    dataset = h5_file[file_prefix]
                    for i, feature_name in enumerate(feature_names):
                        dataset.attrs[feature_name] = features[i]

                # 更新日志
                log_data["处理时长(秒)"] = time_taken + time_taken_fft + time_taken_features
                log_data["处理状态"] = "成功"
                log_data["FFT谱计算状态"] = "成功"
                log_data["特征值计算状态"] = "成功"
                batch_success_count += 1

            except Exception as e:
                log_data["处理状态"] = "失败"
                log_data["失败原因"] = str(e)[:100]
                log_data["FFT谱计算状态"] = "失败"
                log_data["特征值计算状态"] = "失败"
                batch_failed_count += 1
                print(f"处理失败: {file_prefix} - {str(e)[:100]}")

            finally:
                write_log(log_data, LOG_FILE)
                pbar.update(1)

        print(f"第{batch_idx + 1}批完成：成功{batch_success_count}个，失败{batch_failed_count}个")

    pbar.close()

    # 总结统计
    total_time = round((time.time() - start_total_time) / 60, 2)
    log_summary = pd.read_csv(LOG_FILE, encoding='utf-8-sig')
    total_success = len(log_summary[log_summary["处理状态"] == "成功"])
    total_failed = len(log_summary[log_summary["处理状态"] == "失败"])

    print(f"\n=== 所有数据集处理完成 ===")
    print(f"总耗时：{total_time}分钟 | 成功：{total_success}个 | 失败：{total_failed}个")
    print(f"输出HDF5文件：{OUTPUT_H5_PATH}")
    print(f"日志文件：{LOG_FILE}")