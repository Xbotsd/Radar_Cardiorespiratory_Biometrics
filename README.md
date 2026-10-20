# Radar-Based Cardiorespiratory Biometrics

This repository contains the core processing pipeline and feature extraction code for the paper:
**"Radar-Based Cardiorespiratory Biometrics: From Vital Sign Estimation to Cross-Species Open-Set Identification"** (Under Review at *IEEE J-BHI*).

## 📊 Overview
This work presents a unified pipeline across three public datasets (30 dogs and 134 humans). It covers:
- Vital-sign estimation using FMCW radar.
- 30-class closed-set identification and open-set verification via Mahalanobis distance.
- Cross-species discrimination using 16-dimensional sampling-rate-invariant features.

## 📂 Repository Structure
- `data_loader.py`: Code for parsing raw radar data (CSV, MAT, zlib-pickle).
- `signal_processing.py`: Butterworth band-pass filtering, phase unwrapping, and resampling to 20 Hz.
- `cross_species_features.py`: Extraction of the 16-D physical features (power fractions, HRV RMSSD, etc.).
- `model_inference.py`: Interface for the classification models. *(Note: Model weights and centroids are withheld due to ongoing peer review and IP protection).*

## 📦 Datasets
The datasets used in this study are publicly available:
1. Dogs Vital Sign Dataset: https://doi.org/10.6084/m9.figshare.23820915.v1
2. VITALSENSE 120 GHz Dataset: https://doi.org/10.34810/data2962
3. Age-Balanced mmWave Radar Dataset: (Available upon request from original authors)

## 📧 Contact
**Yudong Xu**
School of Mechanical Engineering, Heilongjiang University of Science and Technology
Email: 3468858545@qq.com
