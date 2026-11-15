# Reproduce KROC on a new server

Target: Linux x86-64, NVIDIA GPU/driver compatible with CUDA 12.8, git, ffmpeg,
and a working Python 3 with venv support. Use Python 3.12 for the training venv.
The bootstrap uses uv's Python discovery/download support if needed.

```bash
cd /workspace
git clone https://github.com/YWhero/KROC.git KROC
cd KROC
bash scripts/bootstrap.sh
```

Authenticate a private Git clone on your own trusted PC or use a temporary
repository-scoped method; never commit credentials. The download source below is
public and does not require a Hugging Face token on the GPU server.

Bootstrap fetches the exact LeRobot fork commit, creates .venv, installs pinned
training packages, runs tests, downloads the source dataset at a fixed revision,
converts it, and validates the study/splits and frame counts. It never starts
training or uploads. Use `bash scripts/bootstrap.sh --env-only` to skip the data
download/conversion. Existing complete data is preserved; partial exports fail
instead of overwriting files. Network, GPU-driver compatibility and sufficient
disk space remain server prerequisites; this is not a container image snapshot.

## Source pins and data

- LeRobot: https://github.com/ROBOTIS-GIT/lerobot-cyclo.git
- Commit: 240b4a0314ae0879cdd928c7f4bdc1eee9a01b3b
- HF dataset: HYHero/red_pen_adaptive_handover_202611141907
- HF revision: 8953cd81be67e3605f3b0ecdcc3e1875464250d6
- Python dependency constraints: requirements-lock.txt (original validated Linux environment).

Git tracks source, configs, tests, documentation, study_annotations.json and the
frozen splits.json. Raw videos/MCAP/HDF5, converted Parquet/MP4/NPZ, outputs,
virtualenvs, caches, tokens and machine-specific logs are not versioned. These
are downloaded/rebuilt or restored from the separate checkpoint repository.

The converter writes into a staging directory because data/ already contains
tracked annotations and split files. Only the generated LeRobot/EMG artifacts
are promoted into data/. Its historical random split never replaces the approved
stratified split. With the pinned source and current rules, expect 15038 training
frames and 3854 validation frames after common EMG exclusions.

## Data flow

MCAP follower joints become state[6]; leader target joint positions become
action[6]. Camera MP4/receive timestamps become two RGB 320x180 video streams.
These are aligned causally to a 30 Hz decision grid and stored as LeRobot v3.

EMG remains a separate 2000 Hz sidecar, not a 30 Hz four-value signal embedded
in the LeRobot Parquet. The source's causal bandpass (20-450 Hz) plus 60 Hz notch
signal is copied from processed/bandpass_notch60_counts. Acquisition times are
mapped to the robot host clock, preserving availability times and quality masks.
Each episode NPZ stores time_ns, available_ns, values[M,4], valid[M]. At each
30 Hz robot row the loader selects only available past samples, forms [400,4]
for a 200 ms window and computes per-channel RMS over its last 100 samples
(50 ms). Baseline ignores EMG values but shares the same frame exclusions.
Concat uses RMS[4]; window ACT learns an encoder over [400,4]. Train-only
normalization is computed when starting each new experiment.

## PC relay and checkpoints

On a trusted Windows PC, clone the same private repo, then:

```powershell
python -m venv .relay_venv
.\.relay_venv\Scripts\python.exe -m pip install -r requirements-relay.txt
.\.relay_venv\Scripts\python.exe scripts\relay_client.py --host root@NEW_SSH_HOST --port NEW_PORT
```

Token input stays in the PC process only. The remote relay path currently requires
the project at /workspace/KROC. Use the new server's host/port, not the original
server defaults. See TRAINING_UPDATES_20261115.md for verified cleanup and resume.

After setup, start one experiment at a time using scripts/train.sh. No automatic
Vast stop hook is enabled. Stop the correct current instance only after all
experiments and final checkpoint receipts are complete. Never hard-code the old
instance ID into a new server's script.
