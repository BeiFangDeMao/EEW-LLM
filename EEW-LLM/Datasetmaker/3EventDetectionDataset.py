#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project : LLM
# @File : 新不固定窗长带噪声.py
# @IDE : PyCharm
# @Author : 张嘉南
# @Date : 2026/4/1 上午10:20
# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project : LLM
# @File : 新不固定窗长.py
# @IDE : PyCharm
# @Author : 张嘉南
# @Date : 2026/3/1 下午2:39
"""
不固定窗长波形截取（保留完整元数据 + new_pat_sample）
- 从Excel读取文件名和P波到时
- 随机截取固定长度窗口（TARGET_LENGTH），要求P波前≥MIN_PRE_P点，后≥MIN_POST_P点
- 转换为物理单位(gal)、去趋势、带通滤波(0.75-40Hz)
- 计算SNR、震中距、方位角
- 保存波形至H5 (shape=(300,3), dtype=float32)，并将所有元数据存入attrs
- 同时将元数据写入CSV
- 记录详细处理日志
- 当P波到时≥6秒时，额外截取前300点作为纯噪声样本（new_pat_sample=-1）
"""

import os
import time
import pandas as pd
import numpy as np
import h5py
from scipy.signal import detrend
# 导入工具函数（需确保utils.py在同级或Python路径中）
from utils import butterworth_filter, cal_snr, cal_dis_haversine, cal_azimuth

# -------------------------- 1. 配置参数 --------------------------
EXCEL_PATH = r"G:\迁移学习\LLM\AAAAAAAAA\test_results\P波到时-3级以上日本-zjn.xlsx"
UD_FOLDER = r"G:\Knet\日本3级以上\UD"  # Z方向
NS_FOLDER = r"G:\Knet\日本3级以上\NS"  # N方向
EW_FOLDER = r"G:\Knet\日本3级以上\EW"  # E方向
OUTPUT_CSV = r"G:\迁移学习\LLM\20251209NewTry\2636dataset\JKnet_3001_Pphase_noise.csv"
OUTPUT_H5 = r"G:\迁移学习\LLM\20251209NewTry\2636dataset\JKnet_3001_Pphase_noise.h5"
LOG_FILE = r"G:\迁移学习\LLM\20251209NewTry\2636dataset\processing_log_noise.csv"  # 修正日志文件路径

TARGET_LENGTH = 300  # 固定窗口总长度
MIN_PRE_P = 20  # P波前最少点数（噪声）
MIN_POST_P = 20  # P波后最少点数（信号）
assert MIN_PRE_P + MIN_POST_P <= TARGET_LENGTH, "约束冲突：MIN_PRE_P + MIN_POST_P > TARGET_LENGTH"

BATCH_SIZE = 2000  # 每批处理条数

# CSV输出列（与固定长度标签信息.py保持一致，增加new_pat_sample）
CSV_HEADERS = [
    "file_prefix", "origin_time", "lat", "lon", "depth_km", "mag",
    "station_code", "station_lat", "station_lon", "station_height_m",
    "record_time", "sampling_freq_hz", "duration_time_s", "direction",
    "scale_factor", "max_acc_gal", "last_correction", "pt_sec",
    "pat_sample_original", "new_pat_sample", "crop_start", "crop_end", "zne_array_shape",
    "snr_ud_p5s", "dis", "azi"
]


# -------------------------- 2. 工具函数 --------------------------
def parse_ud_metadata(ud_file_path: str) -> dict:
    """从UD头文件解析完整元数据（震源、台站、仪器参数）"""
    with open(ud_file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    try:
        sampling_freq = int(lines[10].split()[2].rstrip('Hz'))
    except (ValueError, IndexError):
        raise Exception(f"采样频率解析错误：{lines[10] if len(lines) > 10 else '行不足'}")

    # 解析 Scale Factor，如 "2000(gal)/8388608"
    scale_line = lines[13]
    scale_str = scale_line.split()[-1]
    scale_clean = scale_str.replace('(gal)', '')  # → "2000/8388608"
    try:
        num_str, den_str = scale_clean.split('/')
        sf_num = float(num_str)
        sf_den = float(den_str)
    except Exception as e:
        raise ValueError(f"无法解析 scale factor '{scale_str}': {e}")

    metadata = {
        "origin_time": str(lines[0].split()[-1]),
        "lat": float(lines[1].split()[-1]),
        "lon": float(lines[2].split()[-1]),
        "depth_km": float(lines[3].split()[-1]),
        "mag": float(lines[4].split()[-1]),
        "station_code": str(lines[5].split()[-1]),
        "station_lat": float(lines[6].split()[-1]),
        "station_lon": float(lines[7].split()[-1]),
        "station_height_m": float(lines[8].split()[-1]),
        "record_time": str(lines[9].split()[-1]),
        "sampling_freq_hz": int(sampling_freq),
        "duration_time_s": float(lines[11].split()[-1]),
        "direction": str(lines[12].split()[-1]),
        "scale_factor": (sf_num, sf_den),
        "max_acc_gal": float(lines[14].split()[-1].rstrip('gal')),
        "last_correction": str(lines[15].split()[-1])
    }
    return metadata


def read_waveform_data(file_path: str) -> np.ndarray:
    """读取波形数据（原始整型计数）"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f]

    memo_idx = next((i for i, line in enumerate(lines) if line == "Memo."), None)
    if memo_idx is None:
        raise Exception(f"未找到 'Memo.' 行：{file_path}")

    waveform = []
    for line in lines[memo_idx + 1:]:
        if line.strip():
            waveform.extend([int(x) for x in line.split()])
    return np.array(waveform, dtype=np.int32)


def apply_scale_factor(wave: np.ndarray, scale_factor: tuple) -> np.ndarray:
    """将原始计数转换为物理单位 (gal)"""
    sf_num, sf_den = scale_factor
    if sf_den == 0:
        raise ValueError("Scale factor denominator is zero!")
    return wave.astype(np.float32) * (sf_num / sf_den)


def write_log(log_data: dict, log_path: str):
    """写入日志文件（每行一条记录）"""
    log_df = pd.DataFrame([log_data])
    header = not os.path.exists(log_path)
    log_df.to_csv(log_path, mode='a', index=False, header=header, encoding='utf-8-sig')


def process_waveform(wave_raw, scale_factor, fs):
    """处理单个方向波形：转物理单位、去趋势、滤波"""
    phys = apply_scale_factor(wave_raw, scale_factor)
    detrended = detrend(phys, type='linear')
    filt = butterworth_filter(detrended, fs=fs, ftype="bandpass", cutoff=[0.75, 40], order=4)
    return filt


# -------------------------- 3. 主处理逻辑 --------------------------
if __name__ == "__main__":
    start_total_time = time.time()
    print(f"=== 随机P波位置截取（窗长={TARGET_LENGTH}，P前≥{MIN_PRE_P}，P后≥{MIN_POST_P}，保留完整元数据）===")
    print(f"=== 新增功能：P波到时≥6秒时，额外截取前300点作为纯噪声样本（标签=-1）===")

    # 读取Excel总行数
    excel_total_rows = len(pd.read_excel(EXCEL_PATH, header=None))
    total_batches = (excel_total_rows + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Excel总计{excel_total_rows}条数据，分{total_batches}批处理")

    # 初始化CSV文件（若不存在则写入表头）
    if not os.path.exists(OUTPUT_CSV):
        pd.DataFrame(columns=CSV_HEADERS).to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    # 噪声样本计数器
    noise_counter = 1

    # 打开H5文件准备写入
    with h5py.File(OUTPUT_H5, 'w') as h5_file:
        for batch_idx in range(total_batches):
            batch_start = batch_idx * BATCH_SIZE
            batch_end = min((batch_idx + 1) * BATCH_SIZE, excel_total_rows)
            print(f"\n=== 处理第{batch_idx + 1}/{total_batches}批（行{batch_start + 1}-{batch_end}）===")

            # 读取当前批次的Excel数据
            batch_excel = pd.read_excel(EXCEL_PATH, header=None, skiprows=batch_start, nrows=BATCH_SIZE)
            batch_excel.columns = ["file_prefix", "pt_sec"]
            batch_excel["file_prefix"] = batch_excel["file_prefix"].astype(str).str.strip()

            batch_metadata = []  # 用于暂存本批成功数据的元数据（最后写入CSV）
            batch_success_count = 0
            batch_discarded_count = 0

            for row_idx, (_, row) in enumerate(batch_excel.iterrows()):
                global_idx = batch_start + row_idx + 1
                file_prefix = row["file_prefix"]
                pt_sec = row["pt_sec"]
                current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

                # 初始化日志记录
                log_data = {
                    "序号": global_idx,
                    "file_prefix": file_prefix,
                    "处理状态": "失败",
                    "处理时间": current_time,
                    "P波时间（秒）": pt_sec,
                    "原始P波采样点": "",
                    "修改后P波采样点": "",
                    "原始波形长度": "",
                    "截取起始位置": "",
                    "截取结束位置": "",
                    "失败原因": "",
                    "备注": ""
                }

                try:
                    # 检查三方向文件是否存在
                    ud_file = os.path.join(UD_FOLDER, f"{file_prefix}.UD")
                    ns_file = os.path.join(NS_FOLDER, f"{file_prefix}.NS")
                    ew_file = os.path.join(EW_FOLDER, f"{file_prefix}.EW")
                    if not (os.path.exists(ud_file) and os.path.exists(ns_file) and os.path.exists(ew_file)):
                        log_data["失败原因"] = "文件不完整"
                        write_log(log_data, LOG_FILE)
                        continue

                    # 解析UD元数据（包含震源、台站等信息）
                    ud_metadata = parse_ud_metadata(ud_file)
                    fs = ud_metadata["sampling_freq_hz"]
                    scale_factor = ud_metadata["scale_factor"]

                    # 读取原始整型波形
                    ud_raw = read_waveform_data(ud_file)
                    ns_raw = read_waveform_data(ns_file)
                    ew_raw = read_waveform_data(ew_file)
                    min_len = min(len(ud_raw), len(ns_raw), len(ew_raw))
                    log_data["原始波形长度"] = min_len

                    # 计算P波原始采样点
                    pat_original = round(pt_sec * fs)
                    log_data["原始P波采样点"] = pat_original

                    # ==================== 新增功能：截取纯噪声样本 ====================
                    # 当P波到时≥6秒时，额外截取前300点作为纯噪声样本
                    if pt_sec >= 6.0:
                        try:
                            # 检查波形长度是否足够（至少300点）
                            if min_len >= TARGET_LENGTH:
                                # 处理三个方向的波形（前300点）
                                ud_filt_noise = process_waveform(ud_raw[:TARGET_LENGTH], scale_factor, fs)
                                ns_filt_noise = process_waveform(ns_raw[:TARGET_LENGTH], scale_factor, fs)
                                ew_filt_noise = process_waveform(ew_raw[:TARGET_LENGTH], scale_factor, fs)

                                # 组合波形 (3, 300) → (300, 3)
                                zne_array_noise = np.stack([ud_filt_noise, ns_filt_noise, ew_filt_noise], axis=0)

                                # 生成噪声样本名称（A000001格式）
                                noise_name = f"A{noise_counter:06d}"
                                noise_counter += 1

                                # 写入H5 dataset
                                h5_noise_dataset = h5_file.create_dataset(
                                    name=noise_name,
                                    data=zne_array_noise.T,
                                    dtype=np.float32
                                )

                                # 计算震中距和方位角
                                dis_val = cal_dis_haversine(
                                    lat1=ud_metadata["station_lat"],
                                    lon1=ud_metadata["station_lon"],
                                    lat2=ud_metadata["lat"],
                                    lon2=ud_metadata["lon"],
                                    unit="km"
                                )
                                azi_val = cal_azimuth(
                                    lat1=ud_metadata["station_lat"],
                                    lon1=ud_metadata["station_lon"],
                                    lat2=ud_metadata["lat"],
                                    lon2=ud_metadata["lon"]
                                )

                                # 准备噪声数据的元数据（new_pat_sample = -1）
                                noise_meta = {
                                    "file_prefix": noise_name,
                                    "origin_time": ud_metadata["origin_time"],
                                    "lat": ud_metadata["lat"],
                                    "lon": ud_metadata["lon"],
                                    "depth_km": ud_metadata["depth_km"],
                                    "mag": ud_metadata["mag"],
                                    "station_code": ud_metadata["station_code"],
                                    "station_lat": ud_metadata["station_lat"],
                                    "station_lon": ud_metadata["station_lon"],
                                    "station_height_m": ud_metadata["station_height_m"],
                                    "record_time": ud_metadata["record_time"],
                                    "sampling_freq_hz": ud_metadata["sampling_freq_hz"],
                                    "duration_time_s": ud_metadata["duration_time_s"],
                                    "direction": ud_metadata["direction"],
                                    "scale_factor": f"{scale_factor[0]},{scale_factor[1]}",
                                    "max_acc_gal": ud_metadata["max_acc_gal"],
                                    "last_correction": ud_metadata["last_correction"],
                                    "pt_sec": round(pt_sec, 3),
                                    "pat_sample_original": int(pat_original),
                                    "new_pat_sample": -1,  # 噪声样本标签设为-1
                                    "crop_start": 0,
                                    "crop_end": TARGET_LENGTH,
                                    "zne_array_shape": f"3,{TARGET_LENGTH}",
                                    "snr_ud_p5s": np.nan,
                                    "dis": float(dis_val),
                                    "azi": float(azi_val)
                                }

                                # 将元数据写入H5 dataset的attrs
                                for key, val in noise_meta.items():
                                    if isinstance(val, str):
                                        h5_noise_dataset.attrs[key] = val.encode('utf-8')
                                    else:
                                        h5_noise_dataset.attrs[key] = val

                                # 收集噪声元数据（用于写入CSV）
                                batch_metadata.append(noise_meta)

                                print(f"第{global_idx}条：{file_prefix} → 额外截取噪声样本 {noise_name}")
                            else:
                                print(f"第{global_idx}条：{file_prefix} 波形太短({min_len}点)，无法截取噪声样本")
                        except Exception as e:
                            print(f"第{global_idx}条：{file_prefix} 噪声截取失败: {str(e)[:50]}")
                            log_data["备注"] = f"噪声截取失败: {str(e)[:50]}"

                    # ==================== 原有功能：截取带P波的波形样本 ====================
                    # 波形长度必须至少为 TARGET_LENGTH
                    if min_len < TARGET_LENGTH:
                        log_data["处理状态"] = "丢弃"
                        log_data["失败原因"] = f"波形太短 (<{TARGET_LENGTH})"
                        batch_discarded_count += 1
                        write_log(log_data, LOG_FILE)
                        continue

                    # 处理三个方向的波形（完整波形）
                    ud_filt = process_waveform(ud_raw, scale_factor, fs)
                    ns_filt = process_waveform(ns_raw, scale_factor, fs)
                    ew_filt = process_waveform(ew_raw, scale_factor, fs)

                    # 计算信噪比 (使用滤波后的UD波形，P波前后各5秒)
                    snr_val = np.nan
                    try:
                        snr_val = cal_snr(
                            wave=ud_filt,
                            p_pre=5.0,
                            p_post=5.0,
                            p_arrival_time=pt_sec,
                            f=fs
                        )
                    except Exception as e:
                        log_data["备注"] = f"SNR计算失败: {str(e)[:80]}"

                    # 计算震中距和方位角
                    dis_val = cal_dis_haversine(
                        lat1=ud_metadata["station_lat"],
                        lon1=ud_metadata["station_lon"],
                        lat2=ud_metadata["lat"],
                        lon2=ud_metadata["lon"],
                        unit="km"
                    )
                    azi_val = cal_azimuth(
                        lat1=ud_metadata["station_lat"],
                        lon1=ud_metadata["station_lon"],
                        lat2=ud_metadata["lat"],
                        lon2=ud_metadata["lon"]
                    )

                    # 计算合法起始点范围（确保P前≥MIN_PRE_P，P后≥MIN_POST_P）
                    start_min = max(0, pat_original - (TARGET_LENGTH - MIN_POST_P))
                    start_max = min(min_len - TARGET_LENGTH, pat_original - MIN_PRE_P)

                    if start_min > start_max:
                        log_data["处理状态"] = "丢弃"
                        log_data["失败原因"] = f"P位置非法（前<{MIN_PRE_P}或后<{MIN_POST_P}）"
                        batch_discarded_count += 1
                        write_log(log_data, LOG_FILE)
                        continue

                    # 随机选择起始点
                    start = np.random.randint(start_min, start_max + 1)
                    end = start + TARGET_LENGTH

                    # 截取滤波后的波形
                    ud_crop = ud_filt[start:end]
                    ns_crop = ns_filt[start:end]
                    ew_crop = ew_filt[start:end]

                    # 新P波位置（在300点窗口内）
                    new_pat = pat_original - start  # 0-based

                    # 组合波形 (3, 300) → 存储为 (300, 3)
                    zne_array = np.stack([ud_crop, ns_crop, ew_crop], axis=0)  # shape (3, 300)
                    if zne_array.shape != (3, TARGET_LENGTH):
                        raise ValueError("波形形状错误")

                    # 写入H5 dataset
                    h5_dataset = h5_file.create_dataset(
                        name=file_prefix,
                        data=zne_array.T,  # (300, 3)
                        dtype=np.float32
                    )

                    # 准备元数据字典（用于CSV和H5 attrs）
                    current_meta = {
                        "file_prefix": file_prefix,
                        "origin_time": ud_metadata["origin_time"],
                        "lat": ud_metadata["lat"],
                        "lon": ud_metadata["lon"],
                        "depth_km": ud_metadata["depth_km"],
                        "mag": ud_metadata["mag"],
                        "station_code": ud_metadata["station_code"],
                        "station_lat": ud_metadata["station_lat"],
                        "station_lon": ud_metadata["station_lon"],
                        "station_height_m": ud_metadata["station_height_m"],
                        "record_time": ud_metadata["record_time"],
                        "sampling_freq_hz": ud_metadata["sampling_freq_hz"],
                        "duration_time_s": ud_metadata["duration_time_s"],
                        "direction": ud_metadata["direction"],
                        "scale_factor": f"{scale_factor[0]},{scale_factor[1]}",
                        "max_acc_gal": ud_metadata["max_acc_gal"],
                        "last_correction": ud_metadata["last_correction"],
                        "pt_sec": round(pt_sec, 3),
                        "pat_sample_original": int(pat_original),
                        "new_pat_sample": int(new_pat),
                        "crop_start": int(start),
                        "crop_end": int(end),
                        "zne_array_shape": f"3,{TARGET_LENGTH}",
                        "snr_ud_p5s": float(snr_val) if not np.isnan(snr_val) else np.nan,
                        "dis": float(dis_val),
                        "azi": float(azi_val)
                    }

                    # 将元数据写入H5 dataset的attrs
                    for key, val in current_meta.items():
                        if isinstance(val, str):
                            h5_dataset.attrs[key] = val.encode('utf-8')
                        else:
                            h5_dataset.attrs[key] = val

                    # 收集本批成功数据的元数据（用于写入CSV）
                    batch_metadata.append(current_meta)

                    # 更新日志
                    log_data["处理状态"] = "成功"
                    log_data["修改后P波采样点"] = new_pat
                    log_data["截取起始位置"] = start
                    log_data["截取结束位置"] = end
                    batch_success_count += 1
                    print(f"第{global_idx}条：{file_prefix} → new_pat={new_pat}, SNR={snr_val:.2f}, dis={dis_val:.1f}km")

                except Exception as e:
                    log_data["失败原因"] = str(e)[:100]
                    print(f"第{global_idx}条：{file_prefix}（失败：{str(e)[:50]}...）")
                finally:
                    # 无论成功/失败，都写入日志
                    write_log(log_data, LOG_FILE)

            # 本批处理结束后，将成功数据的元数据追加到CSV
            if batch_metadata:
                pd.DataFrame(batch_metadata).to_csv(
                    OUTPUT_CSV, mode='a', index=False, header=False, encoding='utf-8-sig'
                )

            print(
                f"第{batch_idx + 1}批完成：成功{batch_success_count}条，丢弃{batch_discarded_count}条，"
                f"失败{len(batch_excel) - batch_success_count - batch_discarded_count}条"
            )

    # 汇总统计
    total_time = round((time.time() - start_total_time) / 60, 2)
    log_summary = pd.read_csv(LOG_FILE, encoding='utf-8-sig')
    total_success = len(log_summary[log_summary["处理状态"] == "成功"])
    total_discarded = len(log_summary[log_summary["处理状态"] == "丢弃"])
    total_fail = len(log_summary) - total_success - total_discarded

    print(f"\n=== 所有批次处理完成 ===")
    print(f"总耗时：{total_time}分钟 | 成功：{total_success}条 | 丢弃：{total_discarded}条 | 失败：{total_fail}条")
    print(f"输出波形：{OUTPUT_H5}（shape=(300, 3)，单位：gal）")
    print(f"元数据CSV：{OUTPUT_CSV}（包含震源、台站、处理信息等）")
    print(f"噪声样本名称格式：A000001、A000002...（new_pat_sample=-1标识）")
    print(f"日志：{LOG_FILE}")