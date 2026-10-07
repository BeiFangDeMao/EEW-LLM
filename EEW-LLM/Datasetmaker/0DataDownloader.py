#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===================================================================
Project: P-Wave Automatic Picker System (P波自动捡拾与分类系统)
Author : [闫伊淳/YYC]
Date   : 2026-03
Copyright (c) 2026 [闫伊淳]. All rights reserved.

未经作者明确授权，禁止将本代码用于商业用途或二次发布。
Unauthorized copying of this file, via any medium is strictly prohibited.
===================================================================
"""
import requests
from bs4 import BeautifulSoup
import os
import time
import sys
from tqdm import tqdm
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ================== 自动化配置区 ==================
# 如果终端持续出现中断问题，请将下方的 True 改为 False。
# 设为 False 后，程序将跳过所有手动输入，直接使用下方的默认配置运行。
USE_INTERACTIVE = True

DEFAULT_NETWORK = "1"                   # 1代表knet, 2代表kik
DEFAULT_START_DATE = "202501"           # 开始年月 (YYYYMM)
DEFAULT_END_DATE = "202501"             # 结束年月 (YYYYMM)
DEFAULT_SAVE_PATH = "earthquake_data"   # 默认保存在当前目录的 earthquake_data 文件夹中
# ==================================================

def safe_input(prompt, default_val=""):
    """带有抗幽灵中断机制的输入函数"""
    if not USE_INTERACTIVE:
        print(f"{prompt} [自动模式已开启，使用预设值]")
        return default_val
        
    for _ in range(5):  # 最多拦截5次异常信号
        try:
            return input(prompt).strip()
        except KeyboardInterrupt:
            time.sleep(0.3)
            print("\n[系统拦截] 捕获到异常的中断信号，已自动忽略。请继续输入: ", end="", flush=True)
        except EOFError:
            time.sleep(0.3)
            print("\n[系统拦截] 捕获到流异常，已自动忽略。请继续输入: ", end="", flush=True)
    
    print("\n[警告] 连续多次中断，将使用默认值。")
    return default_val

def get_valid_date(prompt, default_val):
    """获取用户输入的有效日期（YYYYMM格式）"""
    while True:
        date_str = safe_input(prompt, default_val)
        if not date_str:
            date_str = default_val
        try:
            date = datetime.strptime(date_str, "%Y%m")
            return date.year, date.month
        except ValueError:
            print("输入格式错误，请使用YYYYMM格式(例如200701)")

def get_valid_path(prompt, default_val, create_if_not_exists=True):
    """获取用户输入的有效路径"""
    while True:
        path = safe_input(prompt, default_val)
        if not path:
            path = default_val

        if os.path.exists(path):
            return path

        if create_if_not_exists:
            try:
                os.makedirs(path)
                print(f"已自动创建目录: {path}")
                return path
            except Exception as e:
                print(f"无法创建目录: {path} ({str(e)})")
        else:
            print(f"路径不存在: {path}")

def create_session():
    """创建一个带有重试机制和请求头伪装的会话，防止被反爬虫拦截"""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    
    retry_strategy = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[403, 429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def download_file(session, username, password, url, save_path):
    """流式下载单个文件"""
    try:
        response = session.get(url, auth=(username, password), stream=True, timeout=30)
        if response.status_code == 200:
            with open(save_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
            return True
        else:
            return False
    except Exception as e:
        print(f"\n文件下载异常 {url}: {str(e)}", file=sys.stderr)
        return False

def process_month(session, username, password, year, month, processed_base_dir, network):
    """处理单个月份的数据"""
    base_url = f"https://www.kyoshin.bosai.go.jp/kyoshin/download/{network}/data/"
    current_url = f"{base_url}{year}/{month:02d}/"
    processed_month_dir = os.path.join(processed_base_dir, str(year), f"{year}{month:02d}")

    os.makedirs(processed_month_dir, exist_ok=True)

    print(f"\n处理 {year}年{month:02d}月数据")
    print(f"URL: {current_url}")
    print(f"保存路径: {processed_month_dir}")

    try:
        response = session.get(current_url, auth=(username, password), timeout=30)
        if response.status_code != 200:
            print(f"未能获取该月数据或该月数据不存在 (状态码: {response.status_code})")
            return 0, 0

        soup = BeautifulSoup(response.text, 'html.parser')

        eq_dirs = [link.get('href') for link in soup.find_all('a')
                   if link.get('href', '').endswith('/') and link.get('href', '')[:-1].isdigit()]

        total_eqs = len(eq_dirs)
        success_eqs = 0

        if total_eqs == 0:
            print("该月没有找到具体的地震事件文件夹")
            return 0, 0

        print(f"找到 {total_eqs} 个地震事件，开始扫描下载...")

        for eq_dir in tqdm(eq_dirs, desc="事件进度"):
            eq_id = eq_dir.strip('/')
            eq_url = current_url + eq_dir
            eq_save_dir = os.path.join(processed_month_dir, eq_id)
            os.makedirs(eq_save_dir, exist_ok=True)

            eq_response = session.get(eq_url, auth=(username, password), timeout=30)
            if eq_response.status_code != 200:
                continue

            eq_soup = BeautifulSoup(eq_response.text, 'html.parser')

            files = [link.get('href') for link in eq_soup.find_all('a')
                     if not link.get('href', '').endswith('/') and not link.get('href', '').startswith('?')]

            all_files_success = True

            for file_name in tqdm(files, desc=f"下载 {eq_id}", leave=False):
                file_url = eq_url + file_name
                save_path = os.path.join(eq_save_dir, file_name)

                if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                    continue

                success = download_file(session, username, password, file_url, save_path)
                if not success:
                    all_files_success = False

                time.sleep(0.1)

            if all_files_success:
                success_eqs += 1

        print(f"\n{'-' * 30}")
        print(f"{year}年{month:02d}月已完成处理 ({success_eqs}/{total_eqs} 成功)")
        print(f"{'-' * 30}")

        return success_eqs, total_eqs

    except Exception as e:
        print(f"处理月份数据时发生错误: {str(e)}", file=sys.stderr)
        return 0, 0


def main():
    """主函数：控制整个数据下载和处理流程"""
    print("=" * 50)
    print("    地震数据下载与处理系统 (抗干扰增强版)")
    print("=" * 50)

    username = "Your username"
    password = "Your password"

    print("\n=== 选择要下载的数据网络 ===")
    print("1. K-NET (knet)")
    print("2. KiK-net (kik)")
    net_choice = safe_input("请输入选项(1或2，直接回车默认1): ", DEFAULT_NETWORK)
    network = "kik" if net_choice == "2" else "knet"

    print("\n=== 设置下载日期范围 ===")
    start_year, start_month = get_valid_date("请输入开始年月(YYYYMM): ", DEFAULT_START_DATE)
    end_year, end_month = get_valid_date("请输入结束年月(YYYYMM): ", DEFAULT_END_DATE)

    if start_year > end_year or (start_year == end_year and start_month > end_month):
        print("错误: 开始年月不能晚于结束年月!")
        return

    print("\n=== 设置保存路径 ===")
    processed_dir = get_valid_path("请输入数据保存路径: ", DEFAULT_SAVE_PATH)

    print("\n" + "=" * 50)
    print("配置信息确认:")
    print(f"数据网络: {network.upper()}")
    print(f"日期范围: {start_year}{start_month:02d} - {end_year}{end_month:02d}")
    print(f"数据路径: {os.path.abspath(processed_dir)}")
    print("=" * 50)

    confirm = safe_input("确认开始处理? (y/n，直接回车默认y): ", "y").lower()
    if confirm not in ['y', 'yes', '']:
        print("操作已取消")
        return

    session = create_session()
    months_to_process = []
    current_year = start_year
    current_month = start_month

    while current_year < end_year or (current_year == end_year and current_month <= end_month):
        months_to_process.append((current_year, current_month))
        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1

    print(f"\n准备处理 {len(months_to_process)} 个月份的数据")

    total_success = 0
    total_eqs = 0

    for year, month in months_to_process:
        success, count = process_month(session, username, password, year, month, processed_dir, network)
        total_success += success
        total_eqs += count

    print("\n" + "=" * 50)
    print(f"所有操作已完成! 总共成功下载 {total_success} / {total_eqs} 个地震事件的数据。")
    print("=" * 50)

if __name__ == "__main__":
    main()
