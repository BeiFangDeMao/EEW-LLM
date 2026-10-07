#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project : LLM
# @File : 预警800点_多进程版.py
# @IDE : PyCharm
# @Author : 张嘉南
# @Date : 2025/10/12 下午12:59
import os
import time
import pandas as pd
import numpy as np
import h5py
from scipy.signal import detrend
from utils import butterworth_filter, cal_snr, cal_dis_haversine, cal_azimuth
from multiprocessing import Pool, Manager, cpu_count
import queue
from concurrent.futures import ProcessPoolExecutor
import traceback

# -------------------------- 1. 配置参数（修正为真正的800点：前300+后500） --------------------------
EXCEL_PATH = r"G:\Knet\日本3级以上\P波到时-3级以上日本-zjn.xlsx"
UD_FOLDER = r"G:\Knet\日本3级以上\UD"  # Z方向
NS_FOLDER = r"G:\Knet\日本3级以上\NS"  # N方向
EW_FOLDER = r"G:\Knet\日本3级以上\EW"  # E方向
OUTPUT_CSV = r"G:\迁移学习\LLM\20251209NewTry\2636dataset\JKnet_300_Regression.csv"
OUTPUT_H5 = r"G:\迁移学习\LLM\20251209NewTry\2636dataset\JKnet_300_Regression.h5"
LOG_FILE = r"G:\迁移学习\LLM\20251209NewTry\2636dataset\JKnet_300_Regression.csv"

TARGET_LENGTH = 300  # 总长度
PRE_PT_LENGTH = 0  # P波前
POST_PT_LENGTH = 300  # P波后
assert PRE_PT_LENGTH + POST_PT_LENGTH == TARGET_LENGTH
BATCH_SIZE = 3000
NUM_PROCESSES = min(cpu_count(), 8)  # 限制进程数量，避免过多进程影响性能


# -------------------------- 2. 工具函数 --------------------------
def parse_ud_metadata(ud_file_path: str) -> dict:
    with open(ud_file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    try:
        sampling_freq = int(lines[10].split()[2].rstrip('Hz'))
    except (ValueError, IndexError):
        raise Exception(f"采样频率解析错误：{lines[10] if len(lines) > 10 else '行不足'}")

    # --- 关键修正：正确解析 Scale Factor 如 "2000(gal)/8388608" ---
    scale_line = lines[13]
    scale_str = scale_line.split()[-1]  # e.g., "2000(gal)/8388608"
    scale_clean = scale_str.replace('(gal)', '')  # → "2000/8388608"
    try:
        num_str, den_str = scale_clean.split('/')
        sf_num = float(num_str)
        sf_den = float(den_str)
    except Exception as e:
        raise ValueError(f"无法解析 scale factor '{scale_str}': {e}")
    # ------------------------------------------------------------

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
        "scale_factor": (sf_num, sf_den),  # (numerator, denominator)
        "max_acc_gal": float(lines[14].split()[-1].rstrip('gal')),
        "last_correction": str(lines[15].split()[-1])
    }
    return metadata


def read_waveform_data(file_path: str) -> np.ndarray:
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f]

    memo_idx = next((i for i, line in enumerate(lines) if line == "Memo.") or [None])
    if memo_idx is None:
        raise Exception(f"未找到 'Memo.' 行：{file_path}")

    waveform = []
    for line in lines[memo_idx + 1:]:
        if line.strip():
            waveform.extend([int(x) for x in line.split()])
    return np.array(waveform, dtype=np.int32)


def apply_scale_factor(wave: np.ndarray, scale_factor: tuple) -> np.ndarray:
    """将原始计数转换为 gal 单位的加速度"""
    sf_num, sf_den = scale_factor
    if sf_den == 0:
        raise ValueError("Scale factor denominator is zero!")
    return wave.astype(np.float32) * (sf_num / sf_den)


def process_single_record(args):
    """处理单条记录的函数，用于多进程调用"""
    row_idx, global_idx, file_prefix, pt_sec, batch_start = args

    result = {
        "success": False,
        "data": None,
        "metadata": None,
        "error_msg": "",
        "global_idx": global_idx
    }

    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

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
        ud_file = os.path.join(UD_FOLDER, f"{file_prefix}.UD")
        ns_file = os.path.join(NS_FOLDER, f"{file_prefix}.NS")
        ew_file = os.path.join(EW_FOLDER, f"{file_prefix}.EW")
        if not (os.path.exists(ud_file) and os.path.exists(ns_file) and os.path.exists(ew_file)):
            log_data["失败原因"] = "文件不完整"
            result["error_msg"] = "文件不完整"
            return result, log_data

        ud_metadata = parse_ud_metadata(ud_file)
        sampling_freq = ud_metadata["sampling_freq_hz"]
        scale_factor = ud_metadata["scale_factor"]
        log_data["采样频率（Hz）"] = sampling_freq

        pat_sample_original = round(pt_sec * sampling_freq)
        log_data["原始P波采样点"] = pat_sample_original

        # 1. 读取原始整型波形
        ud_raw = read_waveform_data(ud_file)
        ns_raw = read_waveform_data(ns_file)
        ew_raw = read_waveform_data(ew_file)
        min_len = min(len(ud_raw), len(ns_raw), len(ew_raw))
        log_data["原始波形长度"] = min_len

        # 2. 转换为物理单位（gal）
        ud_phys = apply_scale_factor(ud_raw, scale_factor)
        ns_phys = apply_scale_factor(ns_raw, scale_factor)
        ew_phys = apply_scale_factor(ew_raw, scale_factor)

        # 3. 去趋势
        ud_detrended = detrend(ud_phys, type='linear')
        ns_detrended = detrend(ns_phys, type='linear')
        ew_detrended = detrend(ew_phys, type='linear')

        # 4. 巴特沃斯带通滤波 (0.75–40 Hz, 4阶, 零相位)
        cutoff = [0.75, 40]
        order = 4
        fs = sampling_freq
        ud_filtered = butterworth_filter(ud_detrended, fs=fs, ftype="bandpass", cutoff=cutoff, order=order)
        ns_filtered = butterworth_filter(ns_detrended, fs=fs, ftype="bandpass", cutoff=cutoff, order=order)
        ew_filtered = butterworth_filter(ew_detrended, fs=fs, ftype="bandpass", cutoff=cutoff, order=order)

        # 5. 计算 SNR、震中距、方位角
        snr_val = np.nan
        try:
            snr_val = cal_snr(
                wave=ud_filtered,
                p_pre=5.0,
                p_post=5.0,
                p_arrival_time=pt_sec,
                f=sampling_freq
            )
        except Exception as e:
            log_data["备注"] = f"SNR计算失败: {str(e)[:80]}"

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

        # 6. 截取波形（使用滤波后、物理单位数据）
        start = pat_sample_original - PRE_PT_LENGTH
        end = pat_sample_original + POST_PT_LENGTH

        if start < 0 or end > min_len or (end - start) != TARGET_LENGTH:
            log_data["处理状态"] = "丢弃"
            log_data["失败原因"] = f"无法截取{TARGET_LENGTH}点（长度{min_len}, P点{pat_sample_original}）"
            result["error_msg"] = f"无法截取{TARGET_LENGTH}点"
            return result, log_data

        ud_cropped = ud_filtered[start:end]
        ns_cropped = ns_filtered[start:end]
        ew_cropped = ew_filtered[start:end]

        zne_array = np.stack([ud_cropped, ns_cropped, ew_cropped], axis=0)

        modified_pat_sample = PRE_PT_LENGTH

        # 7. 整合元数据（scale_factor 转为字符串以便CSV兼容）
        current_meta = {
            "file_prefix": str(file_prefix),
            "pt_sec": float(round(pt_sec, 3)),
            "pat_sample_original": int(pat_sample_original),
            "modified_pat_sample": int(modified_pat_sample),
            "crop_start": int(start),
            "crop_end": int(end),
            "zne_array_shape": f"3,{TARGET_LENGTH}",
            "snr_ud_p5s": float(snr_val),
            "dis": float(dis_val),
            "azi": float(azi_val),
            **ud_metadata
        }
        current_meta["scale_factor"] = f"{scale_factor[0]},{scale_factor[1]}"

        result["success"] = True
        result["data"] = zne_array
        result["metadata"] = current_meta

        log_data["处理状态"] = "成功"
        log_data["截取起始位置"] = start
        log_data["截取结束位置"] = end
        log_data["修改后P波采样点"] = modified_pat_sample

        print(f"第{global_idx}条：{file_prefix}（SNR={snr_val:.2f}, dis={dis_val:.1f}km, azi={azi_val:.1f}°）")

    except Exception as e:
        error_msg = str(e)[:100]
        log_data["失败原因"] = error_msg
        result["error_msg"] = error_msg
        print(f"第{global_idx}条：{file_prefix}（失败：{str(e)[:50]}...）")
        print(f"详细错误信息: {traceback.format_exc()}")

    return result, log_data


def write_log_batch(log_data_list: list, log_path: str):
    """批量写入日志"""
    if not log_data_list:
        return
    log_df = pd.DataFrame(log_data_list)
    header = not os.path.exists(log_path)
    log_df.to_csv(log_path, mode='a', index=False, header=header, encoding='utf-8-sig')


def process_batch_data(batch_excel, batch_start):
    """处理一个批次的数据"""
    tasks = []
    for row_idx, (_, row) in enumerate(batch_excel.iterrows()):
        global_idx = batch_start + row_idx + 1
        file_prefix = row["file_prefix"]
        pt_sec = row["pt_sec"]
        tasks.append((row_idx, global_idx, file_prefix, pt_sec, batch_start))

    results = []
    logs = []

    # 使用进程池处理数据
    with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
        # 提交所有任务
        future_to_task = {executor.submit(process_single_record, task): task for task in tasks}

        # 收集结果
        for future in future_to_task:
            result, log_data = future.result()
            results.append(result)
            logs.append(log_data)

    return results, logs


# -------------------------- 3. 主处理逻辑 --------------------------
if __name__ == "__main__":
    start_total_time = time.time()
    print(f"=== 波形统一长度处理开始（目标长度：{TARGET_LENGTH}点，P波前{PRE_PT_LENGTH}点+后{POST_PT_LENGTH}点）===")
    print(f"使用 {NUM_PROCESSES} 个进程进行并行处理")

    excel_total_rows = len(pd.read_excel(EXCEL_PATH, header=None))
    total_batches = (excel_total_rows + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Excel总计{excel_total_rows}条数据，分{total_batches}批处理")

    csv_headers = [
        "file_prefix", "origin_time", "lat", "lon", "depth_km", "mag",
        "station_code", "station_lat", "station_lon", "station_height_m",
        "record_time", "sampling_freq_hz", "duration_time_s", "direction",
        "scale_factor", "max_acc_gal", "last_correction", "pt_sec",
        "pat_sample_original", "modified_pat_sample", "crop_start", "crop_end", "zne_array_shape",
        "snr_ud_p5s", "dis", "azi"
    ]
    if not os.path.exists(OUTPUT_CSV):
        pd.DataFrame(columns=csv_headers).to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    # 创建HDF5文件
    h5_file = h5py.File(OUTPUT_H5, 'w')

    try:
        for batch_idx in range(total_batches):
            batch_start = batch_idx * BATCH_SIZE
            batch_end = min((batch_idx + 1) * BATCH_SIZE, excel_total_rows)
            print(f"\n=== 处理第{batch_idx + 1}/{total_batches}批（行{batch_start + 1}-{batch_end}）===")

            batch_excel = pd.read_excel(EXCEL_PATH, header=None, skiprows=batch_start, nrows=BATCH_SIZE)
            batch_excel.columns = ["file_prefix", "pt_sec"]
            batch_excel["file_prefix"] = batch_excel["file_prefix"].astype(str).str.strip()

            # 处理当前批次数据
            results, logs = process_batch_data(batch_excel, batch_start)

            # 统计结果
            batch_success_count = sum(1 for r in results if r["success"])
            batch_discarded_count = sum(1 for r in results if not r["success"] and "丢弃" in r["error_msg"])
            batch_failed_count = len(results) - batch_success_count - batch_discarded_count

            # 写入日志
            write_log_batch(logs, LOG_FILE)

            # 准备CSV数据
            csv_data = []
            for result in results:
                if result["success"]:
                    current_meta = result["metadata"]
                    csv_row = {col: current_meta.get(col, "") for col in csv_headers}
                    csv_data.append(csv_row)

            # 写入CSV（追加模式）
            if csv_data:
                pd.DataFrame(csv_data).to_csv(
                    OUTPUT_CSV, mode='a', index=False, header=False, encoding='utf-8-sig'
                )

            # 写入H5文件
            for result in results:
                if result["success"]:
                    file_prefix = result["metadata"]["file_prefix"]
                    zne_array = result["data"]

                    # 写入H5（物理单位 gal，float32）
                    h5_dataset = h5_file.create_dataset(
                        name=file_prefix,
                        data=zne_array.T,  # shape: (300, 3)
                        dtype=np.float32
                    )
                    # 添加属性
                    for key, val in result["metadata"].items():
                        if isinstance(val, str):
                            h5_dataset.attrs[key] = val.encode('utf-8')
                        else:
                            h5_dataset.attrs[key] = val

            print(
                f"第{batch_idx + 1}批完成：成功{batch_success_count}条，丢弃{batch_discarded_count}条，"
                f"失败{batch_failed_count}条")

    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        print(traceback.format_exc())
    finally:
        # 关闭H5文件
        h5_file.close()

    total_time = round((time.time() - start_total_time) / 60, 2)
    if os.path.exists(LOG_FILE):
        log_summary = pd.read_csv(LOG_FILE, encoding='utf-8-sig')
        total_success = len(log_summary[log_summary["处理状态"] == "成功"])
        total_discarded = len(log_summary[log_summary["处理状态"] == "丢弃"])
        total_fail = len(log_summary) - total_success - total_discarded
    else:
        total_success = total_discarded = total_fail = 0

    print(f"\n=== 所有批次处理完成 ===")
    print(f"总耗时：{total_time}分钟 | 成功：{total_success}条 | 丢弃：{total_discarded}条 | 失败：{total_fail}条")
    print("输出波形单位：gal（重力加速度单位）")
    print("新增字段：snr_ud_p5s（P±5s信噪比）, dis（震中距/km）, azi（方位角/°）")
    print(f"元数据CSV：{OUTPUT_CSV}")
    print(f"波形H5：{OUTPUT_H5}")
    print(f"日志：{LOG_FILE}")