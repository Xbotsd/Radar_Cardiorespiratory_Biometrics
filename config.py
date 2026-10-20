# config.py
import os

# 项目根目录
PROJECT_ROOT = os.path.join(os.path.expanduser("~"), "Desktop", "Biometric_Project")

# 数据根目录（真实路径）
DOG_DATA_ROOT = os.path.join(PROJECT_ROOT, "dog_dataset")
SIVIS_DATA_ROOT = os.path.join(PROJECT_ROOT, "SiViS_Database")
SHI_DATA_ROOT = os.path.join(PROJECT_ROOT, "Shi_Dataset")

# 受试者列表
DOG_SUBJECTS = [f"No{i}" for i in range(1, 11)]

# 子文件夹名
RADAR_SUBDIR = "Radar"
BIONET_SUBDIR = "BIONET"

# 雷达参数（后续根据探索结果修正）
FS_RADAR = 20.0             # 暂设，待探索后修正
BREATH_BAND = (0.1, 0.5)    # 呼吸频率 Hz
HEART_BAND = (0.8, 3.0)     # 心跳频率 Hz

# 打印确认
if __name__ == "__main__":
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"DOG_DATA_ROOT: {DOG_DATA_ROOT}")
    print(f"No1 Radar 存在: {os.path.exists(os.path.join(DOG_DATA_ROOT, 'No1', RADAR_SUBDIR))}")