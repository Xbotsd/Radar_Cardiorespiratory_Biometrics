# age_estimation.py
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
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, confusion_matrix, classification_report, r2_score
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


def extract_age_features(sig, fs):
    """提取 16 个心肺特征（同 cross_species_v5.py）"""
    feats = {}
    sig = sig - np.mean(sig)
    sig, fs = resample(sig, fs, TARGET_FS)
    f, P = welch(sig, fs, nperseg=min(512, len(sig)))

    # 呼吸
    br_mask = (f >= 0.15) & (f <= 0.5)
    if br_mask.sum() == 0:
        return None
    br_freq = f[br_mask][np.argmax(P[br_mask])]
    feats['br_freq_hz'] = br_freq

    br_sig = bandpass(sig, fs, 0.15, 0.5)
    br_peaks, _ = find_peaks(br_sig, distance=int(fs * 1.5))
    br_troughs, _ = find_peaks(-br_sig, distance=int(fs * 1.5))

    if len(br_peaks) >= 3:
        periods = np.diff(br_peaks) / fs
        feats['br_period_s'] = float(np.mean(periods))
        feats['br_period_cv'] = float(np.std(periods) / (np.mean(periods) + 1e-6))

        # I:E ratio
        i_e = []
        for p in br_peaks:
            prev = br_troughs[br_troughs < p]
            nxt = br_troughs[br_troughs > p]
            if len(prev) > 0 and len(nxt) > 0:
                ti = (p - prev[-1]) / fs
                te = (nxt[0] - p) / fs
                if te > 0.1:
                    i_e.append(ti / te)
        feats['ie_ratio'] = float(np.mean(i_e)) if i_e else 0
        feats['br_skew'] = float(pd.Series(br_sig).skew())
        feats['br_kurt'] = float(pd.Series(br_sig).kurt())
    else:
        for k in ['br_period_s', 'br_period_cv', 'ie_ratio', 'br_skew', 'br_kurt']:
            feats[k] = 0

    # 心跳
    hr_mask = (f >= 0.8) & (f <= 2.6)
    if hr_mask.sum() == 0:
        return None
    hr_freq = f[hr_mask][np.argmax(P[hr_mask])]
    feats['hr_freq_hz'] = hr_freq

    hr_sig = bandpass(sig, fs, 0.8, 2.6)
    hr_peaks, _ = find_peaks(hr_sig, distance=int(fs * 0.3))
    if len(hr_peaks) >= 3:
        hr_periods = np.diff(hr_peaks) / fs
        feats['hr_period_cv'] = float(np.std(hr_periods) / (np.mean(hr_periods) + 1e-6))
        if len(hr_periods) >= 3:
            feats['hrv_rmssd_s'] = float(np.sqrt(np.mean(np.diff(hr_periods) ** 2)))
        else:
            feats['hrv_rmssd_s'] = 0
        feats['hr_skew'] = float(pd.Series(hr_sig).skew())
        feats['hr_kurt'] = float(pd.Series(hr_sig).kurt())
    else:
        for k in ['hr_period_cv', 'hrv_rmssd_s', 'hr_skew', 'hr_kurt']:
            feats[k] = 0

    # 比值
    feats['hr_br_ratio'] = hr_freq / (br_freq + 1e-10)
    br_power = P[br_mask].sum()
    hr_power = P[hr_mask].sum()
    total = P.sum() + 1e-20
    feats['br_power_frac'] = float(br_power / total)
    feats['hr_power_frac'] = float(hr_power / total)
    feats['br_hr_power_ratio'] = float(np.log10((br_power + 1e-20) / (hr_power + 1e-20)))

    # 耦合
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

    return feats


def load_features():
    """加载所有 P 受试者的特征"""
    print("加载 P 数据集特征...")
    subject_features = {}
    fs = 10.0
    WINDOW_SEC = 20

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
            n = 0
            for start in range(0, len(sig) - ws, st):
                feats = extract_age_features(sig[start:start+ws], fs)
                if feats is not None:
                    windows.append(feats)
                    n += 1
                if n >= 5:
                    break

        if windows:
            # 每个人的特征取中位数
            keys = list(windows[0].keys())
            subject_features[subj] = {k: float(np.median([w[k] for w in windows]))
                                       for k in keys}
    return subject_features


def main():
    print("=" * 80)
    print("年龄估计：基于心肺特征的 P 数据集")
    print("=" * 80)

    # 加载特征
    features = load_features()
    print(f"加载 {len(features)} 个受试者")

    # 加载年龄标签
    labels_file = os.path.join(P_ROOT, "ParticipantsInfo.xlsx")
    labels_df = pd.read_excel(labels_file)
    age_map = dict(zip(labels_df['PARTICIPANT ID'], labels_df['AGE']))
    print(f"加载年龄标签: {len(age_map)} 条")

    # 对齐
    feature_names = list(next(iter(features.values())).keys())
    X_list, y_list, ids = [], [], []
    for subj, feats in features.items():
        if subj in age_map:
            X_list.append([feats[k] for k in feature_names])
            y_list.append(age_map[subj])
            ids.append(subj)

    X = np.array(X_list)
    y = np.array(y_list)
    print(f"\n最终数据集: X={X.shape}, y={y.shape}")
    print(f"年龄范围: {y.min()} - {y.max()} (平均 {y.mean():.1f})")

    # ============ 任务 1：年龄回归 ============
    print(f"\n{'='*80}")
    print("任务 1：年龄回归 (Random Forest)")
    print(f"{'='*80}")

    reg = make_pipeline(
        StandardScaler(),
        RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(reg, X, y, cv=cv, n_jobs=-1)

    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f"MAE = {mae:.2f} 年")
    print(f"R²  = {r2:.3f}")
    print(f"误差 ≤ 5 年: {sum(1 for a,b in zip(y,y_pred) if abs(a-b)<=5)}/{len(y)}")
    print(f"误差 ≤ 10 年: {sum(1 for a,b in zip(y,y_pred) if abs(a-b)<=10)}/{len(y)}")

    # ============ 任务 2：年龄组分类 ============
    print(f"\n{'='*80}")
    print("任务 2：年龄组三分类 (<35, 35-55, >55)")
    print(f"{'='*80}")

    def age_group(a):
        if a < 35:
            return 0
        elif a <= 55:
            return 1
        else:
            return 2

    y_class = np.array([age_group(a) for a in y])
    class_names = ['<35', '35-55', '>55']
    print(f"类别分布: {np.bincount(y_class).tolist()}")

    clf = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    )
    cv_s = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_class_pred = cross_val_predict(clf, X, y_class, cv=cv_s, n_jobs=-1)

    acc = (y_class_pred == y_class).mean()
    print(f"\n准确率: {acc:.3f} (随机基线: {1/3:.3f}, 倍数: {acc/(1/3):.2f}×)")

    # 混淆矩阵
    cm = confusion_matrix(y_class, y_class_pred)
    print(f"\n混淆矩阵:")
    print(f"{'':10} {'<35':>6} {'35-55':>6} {'>55':>6}")
    for i, name in enumerate(class_names):
        print(f"{name:10} {cm[i,0]:>6} {cm[i,1]:>6} {cm[i,2]:>6}")

    # 每类准确率
    print(f"\n每类准确率:")
    for i, name in enumerate(class_names):
        if cm[i].sum() > 0:
            print(f"  {name}: {cm[i,i]/cm[i].sum():.3f}")

    # ============ 特征重要性 ============
    print(f"\n{'='*80}")
    print("特征重要性 (回归)")
    print(f"{'='*80}")

    reg.fit(X, y)
    importances = reg.named_steps['randomforestregressor'].feature_importances_
    order = np.argsort(importances)[::-1]
    for i in order[:10]:
        print(f"  {feature_names[i]:<25} {importances[i]:.4f}")

    # ============ 画图 ============
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 散点
    ax = axes[0]
    ax.scatter(y, y_pred, alpha=0.6, c='steelblue', edgecolors='navy', s=50)
    lim = [y.min()-5, y.max()+5]
    ax.plot(lim, lim, 'k--', alpha=0.5)
    ax.set_xlabel('True Age (years)')
    ax.set_ylabel('Predicted Age (years)')
    ax.set_title(f'Age Regression (MAE = {mae:.2f} years, R² = {r2:.3f})')
    ax.grid(True, alpha=0.3)

    # 混淆矩阵
    ax = axes[1]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Age Group Classification (Acc = {acc:.3f})')

    # 特征重要性
    ax = axes[2]
    top_n = 10
    top_idx = order[:top_n]
    ax.barh(range(top_n), importances[top_idx][::-1], color='forestgreen')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in top_idx][::-1])
    ax.set_xlabel('Importance')
    ax.set_title('Top 10 Features for Age Estimation')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('fig_age_estimation.png', dpi=300, bbox_inches='tight')
    print(f"\n图已保存: fig_age_estimation.png")

    # 保存
    summary = {
        'n_subjects': len(y),
        'regression_mae': float(mae),
        'regression_r2': float(r2),
        'classification_acc': float(acc),
        'classification_chance': 1/3,
        'per_class_acc': {class_names[i]: float(cm[i,i]/max(cm[i].sum(),1))
                          for i in range(3)},
        'feature_importance': {feature_names[i]: float(importances[i]) for i in order},
    }
    with open('age_estimation_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"已保存: age_estimation_results.json")


if __name__ == "__main__":
    main()