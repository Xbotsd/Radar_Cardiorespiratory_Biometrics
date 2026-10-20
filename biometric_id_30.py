# biometric_id_30.py
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch, find_peaks, hilbert
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
import seaborn as sns

FS_RADAR = 50.0
BR_BAND = (0.15, 0.45)
HR_BAND = (1.6, 2.6)
WINDOW_SEC = 15
STRIDE_SEC = 3
MAX_WINDOWS_PER_SUBJECT = 56   # 与 Scenario1 对齐

SCEN1_ROOT = r"C:\Users\lei'shen\Desktop\Biometric_Project\dog_dataset"
SCEN2_ROOT = r"C:\Users\lei'shen\Desktop\Biometric_Project\Scenario2_AwakeDogs"


def bandpass(sig, fs, low, high, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, sig)


def extract_features(window_data, fs=FS_RADAR):
    """从窗口提取呼吸+心跳特征"""
    feats = {}

    # 呼吸频率统计
    br_freqs, br_ampls = [], []
    for col in range(window_data.shape[1]):
        sig = window_data[:, col] - np.mean(window_data[:, col])
        br_sig = bandpass(sig, fs, *BR_BAND)
        f, P = welch(br_sig, fs, nperseg=min(512, len(br_sig)))
        mask = (f >= BR_BAND[0]) & (f <= BR_BAND[1])
        if mask.sum() < 3:
            continue
        br_freqs.append(f[mask][np.argmax(P[mask])])
        br_ampls.append(np.sqrt(np.mean(br_sig**2)))

    if br_freqs:
        br_freqs = np.array(br_freqs)
        feats['br_mean'] = np.median(br_freqs) * 60
        feats['br_std'] = np.std(br_freqs) * 60
        feats['br_iqr'] = (np.percentile(br_freqs, 75) -
                           np.percentile(br_freqs, 25)) * 60
        feats['br_amp'] = np.median(br_ampls)
    else:
        feats['br_mean'] = feats['br_std'] = feats['br_iqr'] = feats['br_amp'] = 0

    # 心跳频率统计
    hr_freqs, hr_snrs = [], []
    for col in range(window_data.shape[1]):
        sig = window_data[:, col] - np.mean(window_data[:, col])
        hr_sig = bandpass(sig, fs, *HR_BAND)
        f, P = welch(hr_sig, fs, nperseg=min(512, len(hr_sig)))
        mask = (f >= HR_BAND[0]) & (f <= HR_BAND[1])
        if mask.sum() < 3:
            continue
        snr = P[mask].max() / (np.median(P[mask]) + 1e-20)
        if snr > 1.5:
            hr_freqs.append(f[mask][np.argmax(P[mask])])
            hr_snrs.append(snr)

    if hr_freqs:
        hr_freqs = np.array(hr_freqs)
        feats['hr_mean'] = np.median(hr_freqs) * 60
        feats['hr_std'] = np.std(hr_freqs) * 60
        feats['hr_iqr'] = (np.percentile(hr_freqs, 75) -
                           np.percentile(hr_freqs, 25)) * 60
        feats['hr_snr_mean'] = np.median(hr_snrs)
    else:
        feats['hr_mean'] = feats['hr_std'] = feats['hr_iqr'] = feats['hr_snr_mean'] = 0

    # 融合
    if feats['br_mean'] > 0 and feats['hr_mean'] > 0:
        feats['hr_br_ratio'] = feats['hr_mean'] / feats['br_mean']
    else:
        feats['hr_br_ratio'] = 0

    # 最优列波形特征
    best_col, best_snr = 0, 0
    for col in range(window_data.shape[1]):
        sig = window_data[:, col] - np.mean(window_data[:, col])
        hr_sig = bandpass(sig, fs, *HR_BAND)
        f, P = welch(hr_sig, fs, nperseg=min(512, len(hr_sig)))
        mask = (f >= HR_BAND[0]) & (f <= HR_BAND[1])
        if mask.sum() < 3:
            continue
        snr = P[mask].max() / (np.median(P[mask]) + 1e-20)
        if snr > best_snr:
            best_snr = snr
            best_col = col

    best_sig = window_data[:, best_col] - np.mean(window_data[:, best_col])

    # 呼吸波形
    br_sig = bandpass(best_sig, fs, *BR_BAND)
    br_peaks, _ = find_peaks(br_sig, distance=int(fs * 1.5))
    if len(br_peaks) >= 2:
        br_periods = np.diff(br_peaks) / fs
        feats['br_interval_mean'] = np.mean(br_periods)
        feats['br_interval_std'] = np.std(br_periods)
        feats['br_skew'] = float(pd.Series(br_sig).skew())
        feats['br_kurt'] = float(pd.Series(br_sig).kurt())
    else:
        feats['br_interval_mean'] = feats['br_interval_std'] = 0
        feats['br_skew'] = feats['br_kurt'] = 0

    # 心跳波形
    hr_sig = bandpass(best_sig, fs, *HR_BAND)
    hr_peaks, _ = find_peaks(hr_sig, distance=int(fs * 0.3))
    if len(hr_peaks) >= 2:
        hr_periods = np.diff(hr_peaks) / fs
        feats['hr_interval_mean'] = np.mean(hr_periods)
        feats['hr_interval_std'] = np.std(hr_periods)
        if len(hr_periods) >= 2:
            feats['hrv_rmssd'] = np.sqrt(np.mean(np.diff(hr_periods)**2))
        else:
            feats['hrv_rmssd'] = 0
        feats['hr_skew'] = float(pd.Series(hr_sig).skew())
        feats['hr_kurt'] = float(pd.Series(hr_sig).kurt())
    else:
        feats['hr_interval_mean'] = feats['hr_interval_std'] = feats['hrv_rmssd'] = 0
        feats['hr_skew'] = feats['hr_kurt'] = 0

    # 心肺耦合
    br_phase = np.angle(hilbert(br_sig))
    hr_env = np.abs(hilbert(hr_sig))
    if len(br_phase) == len(hr_env) and len(br_phase) > 1:
        feats['cardioresp_coupling'] = float(np.abs(
            np.corrcoef(np.cos(br_phase), hr_env)[0, 1]))
    else:
        feats['cardioresp_coupling'] = 0

    return feats


def load_subject(subject, root, max_windows=MAX_WINDOWS_PER_SUBJECT):
    """加载单个受试者的所有窗口特征"""
    radar_path = os.path.join(root, subject, 'Radar')
    csv_files = [f for f in os.listdir(radar_path) if f.endswith('.csv')]
    if not csv_files:
        return None
    csv_file = csv_files[0]
    csv_path = os.path.join(radar_path, csv_file)

    data = pd.read_csv(csv_path, header=None).values
    n_samples = data.shape[0]
    window_size = int(WINDOW_SEC * FS_RADAR)
    stride = int(STRIDE_SEC * FS_RADAR)

    features_list = []
    for start in range(0, n_samples - window_size, stride):
        window = data[start:start + window_size, :]
        features_list.append(extract_features(window))
        if len(features_list) >= max_windows:
            break

    return features_list


def build_dataset():
    """构建 30 类数据集"""
    X, y = [], []
    feature_names = None
    subjects = []

    # Scenario1 (No1-No10)
    for i in range(1, 11):
        subj = f'No{i}'
        print(f"  加载 {subj} (Scenario1) ...")
        feats_list = load_subject(subj, SCEN1_ROOT)
        if not feats_list:
            continue
        if feature_names is None:
            feature_names = list(feats_list[0].keys())
        for f in feats_list:
            X.append([f[k] for k in feature_names])
            y.append(len(subjects))
        subjects.append(subj)
        print(f"    {len(feats_list)} 个窗口")

    # Scenario2 (No11-No30)
    for i in range(11, 31):
        subj = f'No{i}'
        print(f"  加载 {subj} (Scenario2) ...")
        feats_list = load_subject(subj, SCEN2_ROOT)
        if not feats_list:
            continue
        for f in feats_list:
            X.append([f[k] for k in feature_names])
            y.append(len(subjects))
        subjects.append(subj)
        print(f"    {len(feats_list)} 个窗口")

    return np.array(X), np.array(y), feature_names, subjects


def main():
    print("=" * 70)
    print("30 类个体识别")
    print(f"窗口: {WINDOW_SEC}s, 步长: {STRIDE_SEC}s")
    print("=" * 70)

    print("\n构建数据集 ...")
    X, y, feature_names, subjects = build_dataset()
    n_classes = len(subjects)
    print(f"\n总样本: {len(X)}, 特征: {len(feature_names)}, 类别: {n_classes}")
    print(f"每类样本: {np.bincount(y).tolist()}")

    # 特征分组
    br_feats = [i for i, n in enumerate(feature_names) if n.startswith('br_')]
    hr_feats = [i for i, n in enumerate(feature_names)
                if n.startswith('hr_') or n.startswith('hrv_')]
    fusion_feats = [i for i, n in enumerate(feature_names)
                    if 'ratio' in n or 'coupling' in n]

    classifiers = {
        'SVM': make_pipeline(StandardScaler(),
                              SVC(kernel='rbf', C=10, gamma='scale')),
        'RF': make_pipeline(StandardScaler(),
                             RandomForestClassifier(n_estimators=300,
                                                     random_state=42)),
    }

    feature_sets = {
        '仅呼吸': br_feats,
        '仅心跳': hr_feats,
        '呼吸+心跳': br_feats + hr_feats,
        '全部': list(range(len(feature_names))),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"\n{'='*70}")
    print(f"{'特征组合':<20} {'SVM':<20} {'RF':<20}")
    print("-" * 70)

    results = {}
    for name, feats in feature_sets.items():
        X_sub = X[:, feats]
        line = f"{name:<20}"
        results[name] = {}
        for clf_name, clf in classifiers.items():
            scores = cross_val_score(clf, X_sub, y, cv=cv, n_jobs=-1)
            results[name][clf_name] = scores
            line += f"{scores.mean():.3f}±{scores.std():.3f}        "
        print(line)

    # 画对比图
    fig, ax = plt.subplots(figsize=(12, 6))
    names = list(results.keys())
    x = np.arange(len(names))
    width = 0.35

    svm_means = [results[n]['SVM'].mean() for n in names]
    svm_stds = [results[n]['SVM'].std() for n in names]
    rf_means = [results[n]['RF'].mean() for n in names]
    rf_stds = [results[n]['RF'].std() for n in names]

    ax.bar(x - width/2, svm_means, width, yerr=svm_stds,
           label='SVM', color='steelblue', capsize=5)
    ax.bar(x + width/2, rf_means, width, yerr=rf_stds,
           label='Random Forest', color='forestgreen', capsize=5)

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel('5-fold CV Accuracy')
    ax.set_title(f'30-Class Individual Identification (chance = {1/n_classes:.3f})')
    ax.axhline(1/n_classes, color='r', linestyle='--',
               label=f'Chance ({1/n_classes:.3f})', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('fig_30class.png', dpi=300, bbox_inches='tight')
    print(f"\n图已保存: fig_30class.png")

    # 保存结果
    summary = {
        'subjects': subjects,
        'n_samples': len(X),
        'n_classes': n_classes,
        'feature_names': feature_names,
        'results': {k: {c: list(v) for c, v in r.items()}
                    for k, r in results.items()}
    }
    with open('biometric_id_30_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"已保存 biometric_id_30_results.json")


if __name__ == "__main__":
    main()