# cross_species_v5.py
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


def extract_clean_features(sig, fs):
    """
    只保留采样率无关的特征：
    - 所有频率特征（Hz，不是 bin）
    - 所有时间特征（秒，不是样本数）
    - 所有比值特征（无量纲）
    """
    sig = sig - np.mean(sig)
    feats = {}

    # 统一重采样到 20 Hz
    sig, fs = resample(sig, fs, TARGET_FS)
    f, P = welch(sig, fs, nperseg=min(512, len(sig)))

    # ============ 呼吸（物理量）============
    br_mask = (f >= 0.15) & (f <= 0.5)
    if br_mask.sum() == 0:
        return None
    br_freq = f[br_mask][np.argmax(P[br_mask])]
    feats['br_freq_hz'] = br_freq

    br_sig = bandpass(sig, fs, 0.15, 0.5)
    br_peaks, _ = find_peaks(br_sig, distance=int(fs * 1.5))
    br_troughs, _ = find_peaks(-br_sig, distance=int(fs * 1.5))

    if len(br_peaks) >= 3 and len(br_troughs) >= 2:
        # 呼吸周期（秒）
        periods = np.diff(br_peaks) / fs
        feats['br_period_s'] = float(np.mean(periods))
        feats['br_period_cv'] = float(np.std(periods) / (np.mean(periods) + 1e-6))

        # 吸气/呼气时间比
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

        # 波形形态（无量纲）
        feats['br_skew'] = float(pd.Series(br_sig).skew())
        feats['br_kurt'] = float(pd.Series(br_sig).kurt())
    else:
        for k in ['br_period_s', 'br_period_cv', 'ie_ratio', 'br_skew', 'br_kurt']:
            feats[k] = 0

    # ============ 心跳（物理量）============
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

    # ============ 比值特征（无量纲）============
    feats['hr_br_ratio'] = hr_freq / (br_freq + 1e-10)

    # 频谱能量分布（归一化后的比例，采样率无关）
    br_power = P[br_mask].sum()
    hr_power = P[hr_mask].sum()
    total = P.sum() + 1e-20
    feats['br_power_frac'] = float(br_power / total)
    feats['hr_power_frac'] = float(hr_power / total)
    feats['br_hr_power_ratio'] = float(np.log10((br_power + 1e-20) / (hr_power + 1e-20)))

    # ============ 心肺耦合 ============
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


def load_dog(max_subs=30, max_win=20):
    from biometric_id_30 import SCEN1_ROOT, SCEN2_ROOT
    X = []
    WINDOW_SEC = 20

    for root, subs in [(SCEN1_ROOT, [f'No{i}' for i in range(1, 11)]),
                        (SCEN2_ROOT, [f'No{i}' for i in range(11, 31)])]:
        for subj in subs[:max_subs]:
            radar_path = os.path.join(root, subj, 'Radar')
            if not os.path.exists(radar_path):
                continue
            csvs = [f for f in os.listdir(radar_path) if f.endswith('.csv')]
            if not csvs:
                continue
            data = pd.read_csv(os.path.join(radar_path, csvs[0]),
                                header=None, nrows=15000).values
            fs = 50.0
            stds = np.std(data, axis=0)
            sig = data[:, np.argmax(stds)]
            ws = int(WINDOW_SEC * fs)
            st = ws // 2
            n = 0
            for start in range(0, len(sig) - ws, st):
                feats = extract_clean_features(sig[start:start+ws], fs)
                if feats is not None:
                    X.append(feats)
                    n += 1
                if n >= max_win:
                    break
    return X


def load_p(max_subs=30, max_win=10):
    X = []
    fs = 10.0
    WINDOW_SEC = 20

    for i in range(1, max_subs + 1):
        subj = f"P{i:03d}"
        for scen, state in [("Lying", "Rest"), ("Sitting", "Rest")]:
            path = os.path.join(P_ROOT, subj, scen, state, "radar_rFFTs.zlib")
            if not os.path.exists(path):
                continue
            try:
                with open(path, 'rb') as f:
                    obj = pickle.loads(zlib.decompress(f.read()))
            except Exception:
                continue
            data = obj[0]
            data_sum = data.sum(axis=1)
            phases = np.unwrap(np.angle(data_sum), axis=0)
            stds = np.std(phases, axis=0)
            sig = phases[:, np.argmax(stds)]
            ws = int(WINDOW_SEC * fs)
            st = ws // 2
            n = 0
            for start in range(0, len(sig) - ws, st):
                feats = extract_clean_features(sig[start:start+ws], fs)
                if feats is not None:
                    X.append(feats)
                    n += 1
                if n >= max_win:
                    break
    return X


def main():
    print("=" * 80)
    print("跨物种验证 v5：只保留采样率无关的特征")
    print(f"所有信号统一重采样到 {TARGET_FS} Hz")
    print("=" * 80)

    print("\n[1] 加载狗...")
    dog_list = load_dog()
    feature_names = list(dog_list[0].keys())
    X_dog = np.array([[f[k] for k in feature_names] for f in dog_list])
    print(f"  {len(X_dog)} 样本, {len(feature_names)} 特征")

    print("\n[2] 加载 P 人...")
    p_list = load_p()
    X_p = np.array([[f[k] for k in feature_names] for f in p_list])
    print(f"  {len(X_p)} 样本")

    print(f"\n{'='*80}")
    print(f"{'特征':<24} {'狗':<20} {'P人':<20} {'可分性':<8}")
    print(f"{'='*80}")
    for i, name in enumerate(feature_names):
        d_mu, d_sd = X_dog[:, i].mean(), X_dog[:, i].std()
        p_mu, p_sd = X_p[:, i].mean(), X_p[:, i].std()
        pooled = np.sqrt((d_sd**2 + p_sd**2) / 2) + 1e-10
        sep = abs(d_mu - p_mu) / pooled
        marker = "***" if sep > 1.0 else ("**" if sep > 0.5 else " ")
        print(f"{name:<24} {d_mu:7.3f}±{d_sd:<7.3f} {p_mu:7.3f}±{p_sd:<7.3f} "
              f"{sep:.2f} {marker}")

    # 分类
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    X_all = np.concatenate([X_dog, X_p], axis=0)
    y_all = np.concatenate([np.zeros(len(X_dog)), np.ones(len(X_p))])

    # 剔除异常值
    mask = np.ones(len(X_all), dtype=bool)
    for i in range(X_all.shape[1]):
        q1, q3 = np.percentile(X_all[:, i], [25, 75])
        iqr = q3 - q1
        mask &= (X_all[:, i] >= q1 - 3*iqr) & (X_all[:, i] <= q3 + 3*iqr)

    X_clean, y_clean = X_all[mask], y_all[mask]
    print(f"\n剔除异常后: {len(X_clean)} 样本")

    clf = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_clean, y_clean, cv=cv, n_jobs=-1)

    print(f"\n{'='*80}")
    print(f"5折交叉验证 (狗 vs P人): {scores.mean():.3f} ± {scores.std():.3f}")
    print(f"随机基线: 0.5")
    print(f"{'='*80}")

    # 特征重要性
    clf.fit(X_clean, y_clean)
    rf = clf.named_steps['randomforestclassifier']
    importances = rf.feature_importances_
    order = np.argsort(importances)[::-1]

    print(f"\n特征重要性 Top 10:")
    for i in order[:10]:
        print(f"  {feature_names[i]:<25} {importances[i]:.4f}")

    with open('cross_species_v5_results.json', 'w') as f:
        json.dump({
            'n_dog': len(X_dog), 'n_p': len(X_p),
            'cv_mean': float(scores.mean()),
            'cv_std': float(scores.std()),
            'importance': {feature_names[i]: float(importances[i]) for i in order},
        }, f, indent=2)
    print(f"\n已保存 cross_species_v5_results.json")


if __name__ == "__main__":
    main()