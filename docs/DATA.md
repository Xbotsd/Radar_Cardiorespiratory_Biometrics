# Data manifest

All experiments use three **public** datasets.  None of the raw files are
redistributed in this repository; the loaders read them from
`RCB_DATA_ROOT` (default `~/Desktop/Biometric_Project`).

```
RCB_DATA_ROOT/
├── dog_dataset/              Scenario 1 – 10 anaesthetised dogs (180 s each)
│   └── No1..No10/
│       ├── Radar/RawData_NoX.csv          9000 x 467  (50 Hz, real-valued)
│       └── BIONET/Reference_{HR,BR}_NoX.csv   1 Hz reference, 180 samples
├── Scenario2_AwakeDogs/      Scenario 2 – 20 awake dogs (1800 s each)
│   └── No11..No30/
│       └── Radar/RawData_NoX.csv          90000 x 312 (50 Hz)
├── VS01..VS24/               VITALSENSE 120 GHz, 24 subjects x {Resting, Apnea}
│   ├── VSxx_{Resting|Apnea}.mat           VitalSig: 40000 samples @ 333.33 Hz
│   └── VSxx_{Resting|Apnea}_Mindray.mat   HR/RR @ 1 Hz, ECG @ 500 Hz, pleth, resp
├── P001..P110/               60 GHz FMCW age-balanced set, 110 subjects
│   └── {Lying,Sitting}/{Rest,Post-exercise}/
│       ├── radar_rFFTs.zlib               (500, 8, 64) complex range-FFT cube @ 10 Hz
│       ├── movesense_ecg.csv              ECG @ 256 Hz
│       └── radar_chirpConfig.json         60.25 GHz start, B = 480 MHz, 64 ADC samples
└── ParticipantsInfo.xlsx     110 rows: PARTICIPANT ID, AGE, GENDER, HR, SpO2, ...
```

## Sources

| # | Dataset | Radar | Subjects | Reference | DOI / location |
|---|---|---|---|---|---|
| 1 | Dogs Vital Sign (Ahmed *et al* 2024) | UWB 7.3 GHz, 50 Hz | 30 dogs (10 anaesthetised + 20 awake) | BIONET HR/BR | `10.6084/m9.figshare.23820915.v1` |
| 2 | VITALSENSE (Wu *et al* 2026) | FMCW 120 GHz, 333.33 Hz | 24 adults, rest + apnea | ECG, resp. belt, SpO₂ | `10.34810/data2962` |
| 3 | Age-balanced mmWave (Parralejo *et al* 2026) | FMCW 60 GHz, 10 Hz frames | 110 adults, 22–76 y | Movesense ECG | Zenodo record 18599983 |

## Verified record counts

The numbers below are asserted by `tests/test_data_manifest.py` when the data
root is present, so a silently truncated download fails loudly instead of
biasing a result.

| Item | Count |
|---|---|
| Dog recordings, Scenario 1 | 10 (9 000 samples each) |
| Dog recordings, Scenario 2 | 20 (90 000 samples each) |
| VITALSENSE recordings | 48 (24 subjects × 2 conditions) |
| 60 GHz recordings | 440 (110 subjects × 4 conditions) |
| 60 GHz frames per recording | 400–600 (median 500) |
| Dog windows, 15 s / 3 s, 56 cap | 12 450 |
| Dog windows, 20 s / 10 s, 20 cap | 560 |
| 60 GHz windows, 20 s / 10 s, 10 cap | 1 408 |
| VITALSENSE windows, 20 s / 10 s, 10 cap | 480 |

## Known quirks (handled in `src/rcb/io.py`)

* The 60 GHz pickles were written with NumPy < 2 and reference
  `numpy.core.numeric`; the loader aliases that module before unpickling.
* The 120 GHz `VitalSig` channel is a *displacement* signal, not a complex
  range-FFT.  Its respiratory band is weak by construction (the release
  emphasises cardiac motion), which is why the 120 GHz BR result
  (MAE 4.37 bpm) is reported as a dataset limitation and not as algorithm
  performance.
* The 7.3 GHz radar CSV holds real-valued samples of the mixed signal, so the
  respiratory and cardiac components are separated in the frequency domain
  rather than by phase demodulation.  No phase unwrapping is applied to this
  dataset.
* `ParticipantsInfo.xlsx` is required for the age experiment only.

## Licences

Each dataset keeps its own licence.  Check the original record before
redistributing derived feature tables; the tables in `results/features/` are
derived data, not raw recordings.
