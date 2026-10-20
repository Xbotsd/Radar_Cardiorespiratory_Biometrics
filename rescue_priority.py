# rescue_priority.py
import os
import sys
import json
import zlib
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch, find_peaks, hilbert, resample_poly
from math import gcd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

import numpy.core.numeric as _ncn
sys.modules['numpy.core.numeric'] = _ncn

P_ROOT = r"C:\Users\lei'shen\Desktop\Biometric_Project"
TARGET_FS = 20.0


def bandpass(sig, fs, low, high, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, sig)


def resample(sig, fs_orig, fs_target=TARGET_FS):
    if abs(fs_orig - fs_target) < 0.01:
        return sig, fs_orig
    a, b = int(round(fs_orig)), int(round(fs_target))
    g = gcd(a, b)
    return resample_poly(sig, b // g, a // g), fs_target


def extract_priority_features(sig, fs):
    """
    救援优先级特征——强调生理异常，而非年龄预测
    新增：心率/呼吸的异常标志
    """
    feats = {}
    sig = sig - np.mean(sig)
    sig, fs = resample(sig, fs, TARGET_FS)
    f, P = welch(sig, fs, nperseg=min(512, len(sig)))

    # ============ 呼吸 ============
    br_mask = (f >= 0.1) & (f <= 0.5)
    if br_mask.sum() == 0:
        return None
    br_freq = f[br_mask][np.argmax(P[br_mask])]
    br_bpm = br_freq * 60
    feats['br_bpm'] = br_bpm
    # 呼吸异常标志
    feats['br_brady'] = float(br_bpm < 8)     # 呼吸过缓
    feats['br_tachy'] = float(br_bpm > 24)    # 呼吸过速
    feats['br_apnea'] = float(br_bpm < 4)     # 呼吸暂停

    br_sig = bandpass(sig, fs, 0.1, 0.5)
    br_peaks, _ = find_peaks(br_sig, distance=int(fs * 1.5))
    if len(br_peaks) >= 3:
        periods = np.diff(br_peaks) / fs
        feats['br_period_cv'] = float(np.std(periods) / (np.mean(periods) + 1e-6))
        feats['br_irregularity'] = float(np.std(periods))
    else:
        feats['br_period_cv'] = 0
        feats['br_irregularity'] = 0

    # ============ 心跳 ============
    hr_mask = (f >= 0.8) & (f <= 3.0)
    if hr_mask.sum() == 0:
        return None
    hr_freq = f[hr_mask][np.argmax(P[hr_mask])]
    hr_bpm = hr_freq * 60
    feats['hr_bpm'] = hr_bpm
    # 心跳异常标志
    feats['hr_brady'] = float(hr_bpm < 50)    # 心动过缓
    feats['hr_tachy'] = float(hr_bpm > 120)   # 心动过速
    feats['hr_critical'] = float(hr_bpm < 40 or hr_bpm > 150)

    hr_sig = bandpass(sig, fs, 0.8, 3.0)
    hr_peaks, _ = find_peaks(hr_sig, distance=int(fs * 0.3))
    if len(hr_peaks) >= 3:
        hr_periods = np.diff(hr_peaks) / fs
        feats['hr_period_cv'] = float(np.std(hr_periods) / (np.mean(hr_periods) + 1e-6))
        if len(hr_periods) >= 3:
            feats['hrv_rmssd'] = float(np.sqrt(np.mean(np.diff(hr_periods) ** 2)))
        else:
            feats['hrv_rmssd'] = 0
    else:
        feats['hr_period_cv'] = 0
        feats['hrv_rmssd'] = 0

    # ============ 心肺耦合（异常时降低）============
    if len(br_sig) == len(hr_sig) and len(br_sig) > 10:
        br_phase = np.angle(hilbert(br_sig))
        hr_env = np.abs(hilbert(hr_sig))
        if np.std(hr_env) > 1e-10:
            feats['cardioresp_coupling'] = float(
                np.abs(np.corrcoef(np.cos(br_phase), hr_env)[0, 1]))
        else:
            feats['cardioresp_coupling'] = 0
    else:
        feats['cardioresp_coupling'] = 0

    # ============ 信号稳定性（置信度代理）============
    feats['hr_snr'] = float(P[hr_mask].max() / (np.median(P) + 1e-20))
    feats['br_snr'] = float(P[br_mask].max() / (np.median(P) + 1e-20))

    return feats


def load_features():
    print("加载 P 数据集特征...")
    subject_features = {}
    fs = 10.0
    WINDOW_SEC = 30    # 用 30 秒窗口（比之前长）

    for i in range(1, 111):
        subj = f"P{i:03d}"
        windows = []
        for scen, state in [("Lying", "Rest"), ("Sitting", "Rest")]:
            path = os.path.join(P_ROOT, subj, scen, state, "radar_rFFTs.zlib")
            if not os.path.exists(path):
                continue
            try:
                with open(path, 'rb') as fh:
                    obj = pickle.loads(zlib.decompress(fh.read()))
                data = obj[0]
            except Exception:
                continue
            data_sum = data.sum(axis=1)
            phases = np.unwrap(np.angle(data_sum), axis=0)
            stds = np.std(phases, axis=0)
            sig = phases[:, np.argmax(stds)]

            ws = int(WINDOW_SEC * fs)
            st = ws // 2
            for start in range(0, len(sig) - ws, st):
                feats = extract_priority_features(sig[start:start+ws], fs)
                if feats is not None:
                    windows.append(feats)
                break   # 每人每场景只取 1 个窗口

        if windows:
            keys = list(windows[0].keys())
            subject_features[subj] = {k: float(np.median([w[k] for w in windows]))
                                       for k in keys}
    return subject_features


def main():
    print("=" * 80)
    print("救援优先级分类：基于心肺特征")
    print("=" * 80)

    features = load_features()
    print(f"加载 {len(features)} 个受试者")

    labels_file = os.path.join(P_ROOT, "ParticipantsInfo.xlsx")
    labels_df = pd.read_excel(labels_file)
    age_map = dict(zip(labels_df['PARTICIPANT ID'], labels_df['AGE']))

    feature_names = list(next(iter(features.values())).keys())
    X_list, y_age, y_priority, ids = [], [], [], []

    for subj, feats in features.items():
        if subj not in age_map:
            continue
        age = age_map[subj]
        X_list.append([feats[k] for k in feature_names])
        y_age.append(age)
        # 救援优先级三级（简化 START 检伤）：
        # P1（立即）：年龄 <8 或 >70，或生命体征严重异常
        # P2（延迟）：年龄 8-15 或 55-70
        # P3（轻伤）：年龄 16-55 且生命体征正常
        if age < 8 or age > 70 or feats['br_apnea'] > 0 or feats['hr_critical'] > 0:
            y_priority.append(0)   # P1
        elif age < 16 or age > 55:
            y_priority.append(1)   # P2
        else:
            y_priority.append(2)   # P3
        ids.append(subj)

    X = np.array(X_list)
    y_age = np.array(y_age)
    y_priority = np.array(y_priority)

    print(f"\n数据集: X={X.shape}")
    print(f"年龄: {y_age.min()}-{y_age.max()} (均值 {y_age.mean():.1f})")
    print(f"\n救援优先级分布:")
    print(f"  P1 (立即): {sum(y_priority==0)}")
    print(f"  P2 (延迟): {sum(y_priority==1)}")
    print(f"  P3 (轻伤): {sum(y_priority==2)}")

    # ============ 分类 ============
    clf = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=200, random_state=42,
                                class_weight='balanced', n_jobs=-1)
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X, y_priority, cv=cv, n_jobs=-1)

    acc = (y_pred == y_priority).mean()
    print(f"\n{'='*80}")
    print(f"救援优先级三分类准确率: {acc:.3f}")
    print(f"随机基线: 1/3 = 0.333, 倍数: {acc/(1/3):.2f}×")
    print(f"{'='*80}")

    # 混淆矩阵
    cm = confusion_matrix(y_priority, y_pred)
    class_names = ['P1 (立即)', 'P2 (延迟)', 'P3 (轻伤)']
    print(f"\n混淆矩阵:")
    print(f"{'':15} {'P1':>6} {'P2':>6} {'P3':>6}")
    for i, name in enumerate(class_names):
        print(f"{name:15} {cm[i,0]:>6} {cm[i,1]:>6} {cm[i,2]:>6}")

    print(f"\n每类准确率:")
    for i, name in enumerate(class_names):
        if cm[i].sum() > 0:
            print(f"  {name}: {cm[i,i]/cm[i].sum():.3f}")

    # 特征重要性
    clf.fit(X, y_priority)
    importances = clf.named_steps['randomforestclassifier'].feature_importances_
    order = np.argsort(importances)[::-1]
    print(f"\nTop 10 特征:")
    for i in order[:10]:
        print(f"  {feature_names[i]:<25} {importances[i]:.4f}")

    # 画图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Rescue Priority Classification (Acc = {acc:.3f})')

    ax = axes[1]
    top_n = 10
    top_idx = order[:top_n]
    ax.barh(range(top_n), importances[top_idx][::-1], color='crimson')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in top_idx][::-1])
    ax.set_xlabel('Importance')
    ax.set_title('Top Features for Rescue Priority')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('fig_rescue_priority.png', dpi=300, bbox_inches='tight')
    print(f"\n图已保存: fig_rescue_priority.png")

    with open('rescue_priority_results.json', 'w') as f:
        json.dump({
            'n_subjects': len(y_priority),
            'accuracy': float(acc),
            'chance': 1/3,
            'class_distribution': {
                'P1': int(sum(y_priority==0)),
                'P2': int(sum(y_priority==1)),
                'P3': int(sum(y_priority==2)),
            },
            'per_class_acc': {class_names[i]: float(cm[i,i]/max(cm[i].sum(),1))
                              for i in range(3)},
            'feature_importance': {feature_names[i]: float(importances[i])
                                    for i in order},
        }, f, indent=2)
    print(f"已保存: rescue_priority_results.json")


if __name__ == "__main__":
    main()