# data_loader.py
import os
import numpy as np
import scipy.io as sio
import pandas as pd
from config import DOG_DATA_ROOT, RADAR_SUBDIR, BIONET_SUBDIR

def load_reference(subject):
    """加载 BIONET 参考信号（呼吸率、心率）"""
    bionet_path = os.path.join(DOG_DATA_ROOT, subject, BIONET_SUBDIR)
    br_file = os.path.join(bionet_path, f"Reference_BR_{subject}.csv")
    hr_file = os.path.join(bionet_path, f"Reference_HR_{subject}.csv")
    br = pd.read_csv(br_file, header=None).values.flatten()
    hr = pd.read_csv(hr_file, header=None).values.flatten()
    return br, hr

def load_radar_raw(subject):
    """加载雷达原始 .mat 数据"""
    radar_path = os.path.join(DOG_DATA_ROOT, subject, RADAR_SUBDIR)
    mat_files = [f for f in os.listdir(radar_path) if f.endswith('.mat')]
    if not mat_files:
        raise FileNotFoundError(f"未找到 .mat 文件于 {radar_path}")
    mat_file = mat_files[0]
    mat_path = os.path.join(radar_path, mat_file)
    mat = sio.loadmat(mat_path)
    return mat, mat_path

def extract_radar_signal(mat):
    """从 .mat 字典中提取雷达时序信号（自动尝试常见变量名）"""
    keys = [k for k in mat.keys() if not k.startswith('__')]
    print(f"  可用变量: {keys}")

    # 常见变量名
    for key in ['data', 'radar', 'RawData', 'signal', 'IQ', 'RawData_No1', 'RawData']:
        if key in mat:
            data = mat[key]
            print(f"  使用变量 '{key}', 形状: {data.shape}, 类型: {data.dtype}")
            return data

    # 如果以上都没有，返回第一个非元数据变量
    data = mat[keys[0]]
    print(f"  使用默认变量 '{keys[0]}', 形状: {data.shape}")
    return data