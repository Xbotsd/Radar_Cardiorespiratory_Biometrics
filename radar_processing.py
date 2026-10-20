# radar_processing.py
import numpy as np
from scipy.signal import butter, filtfilt, welch

def bandpass_filter(signal, fs, lowcut, highcut, order=4):
    """巴特沃斯带通滤波器"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def extract_phase_signal(radar_data):
    """
    从雷达原始数据中提取相位信号。
    假设数据为复数 IQ 或实数矩阵，沿慢时间维平均。
    """
    if np.iscomplexobj(radar_data):
        phase = np.unwrap(np.angle(radar_data))
        if phase.ndim > 1:
            phase = np.mean(phase, axis=1)
        return phase
    else:
        if radar_data.ndim > 1:
            return np.mean(radar_data, axis=1)
        return radar_data

def estimate_vital_signs(signal, fs, breath_band=(0.1, 0.5), heart_band=(0.8, 3.0)):
    """从信号中估计呼吸和心跳频率（次/分）"""
    signal = signal - np.mean(signal)  # 去直流

    # 分离呼吸和心跳
    breath_sig = bandpass_filter(signal, fs, breath_band[0], breath_band[1])
    heart_sig = bandpass_filter(signal, fs, heart_band[0], heart_band[1])

    # 频谱分析
    nperseg = min(256, len(signal))
    f_breath, P_breath = welch(breath_sig, fs, nperseg=nperseg)
    f_heart, P_heart = welch(heart_sig, fs, nperseg=nperseg)

    # 峰值频率
    breath_freq = f_breath[np.argmax(P_breath)]
    heart_freq = f_heart[np.argmax(P_heart)]

    # 转换为 次/分
    breath_bpm = breath_freq * 60
    heart_bpm = heart_freq * 60

    return breath_bpm, heart_bpm, (f_breath, P_breath), (f_heart, P_heart)