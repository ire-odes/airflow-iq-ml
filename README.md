# AirFlow IQ — acoustic ML

Detects HVAC filter loading from sound. Each device uploads a short clip; a
poller on this machine scores it against that device's own learned baseline
and writes a verdict the web and mobile apps display.

**How the pipeline and wake intervals work:** [docs/pipeline.md](docs/pipeline.md)

## Layout

```
ML/
├── drift_detector.py         core: per-device baseline, Mahalanobis drift, alert rules
├── extract_features_v2.py    core: audio -> feature vector
├── spectrum_viz.py           core: 72-bin log spectrum (silence gate + app chart)
│
├── service/                  the live system -- runs as a scheduled task
│   ├── poll_and_infer.py         the poller: gate, score, burst, persist
│   ├── recalibrate_all.py        reset a device's baseline
│   ├── status.py / check_*.py    inspect devices and baselines
│   ├── run_migration.py          apply SQL to the live database
│   └── save_audio_by_gate.py, spectral_gate_log.py   collectors for experiments
│
├── models/                   artifacts the service loads
│   ├── model_bundle.pkl          classifier (used only before a device is calibrated)
│   └── pooled_calibration.json   default threshold for a brand-new device
│
├── training/                 rebuilds what is in models/
│   └── legacy/                   superseded v1 feature extractor and trainer
├── research/                 one-off analyses and plots behind past decisions
├── tests/                    smoke test and offline drift test
├── results/                  generated outputs -- safe to regenerate
│   ├── features/  tables/  figures/
├── firmware/                 LoRa slot-sync build and the TTN ingest function
├── docs/                     pipeline write-up, reference papers
├── data/                     labelled audio corpora (not in git)
├── backups/                  exports taken before destructive DB cleanups (not in git)
└── tools/                    portable Node for building the web app (not in git)
```

The three core modules stay at the root on purpose: the service, the
collectors and every script import them from here. Scripts in `training/`,
`research/` and `tests/` start with a short shim that puts the root on the
import path and makes it the working directory, so run them from anywhere:

```
venv\Scripts\python research\plot_drift_features.py
```

## Running the service

The poller runs as the Windows scheduled task **Airflow IQ inference poller**
(`service\poll_and_infer.py --loop 30`, working directory this folder).

**Never start a second copy by hand while the task is running.** Two pollers
score every clip twice. Since 2026-09-18 the database rejects the duplicate
(`filter_ml_readings_one_per_clip`), so a second copy now shows up as an
`ERROR ... duplicate key` line instead of silently corrupting history -- but
stop the task first if you want to run it yourself:

```powershell
Stop-ScheduledTask  'Airflow IQ inference poller'
venv\Scripts\python service\poll_and_infer.py          # one pass
Start-ScheduledTask 'Airflow IQ inference poller'
```

This machine must stay awake for the poller to run. While it sleeps, clips
are overwritten in storage before they can be scored.
