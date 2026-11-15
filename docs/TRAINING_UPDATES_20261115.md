# Training changes, 2026-11-15

Only study provenance, a frozen stratified split, complete validation, checkpoint
selection and archival changed. ACT architectures, batch 8, LR 1e-5, 100000 steps,
chunk 30, n_action_steps 15, seed 1000, EMG RMS 50 ms and window 200 ms are unchanged.

## Study and split

All production configs load `data/study_annotations.json`. The loader validates
episode coverage and train/validation disjointness. Each experiment stores a copy
and SHA256 of the annotations and split. Condition labels are evaluation metadata,
never policy inputs. The participant is yw, the recipient, right arm; rotation is
defined from the recipient's perspective. Public metadata uses participant_01.

Seed 1000, two validation episodes per condition, fixed before training:

- train: 1,2,3,4,6,7,8,10,11,12,13,15,16,17,19,20,21,22,23,24,25,26,27,28
- validation: 0,5,9,14,18,29
- counts: train as-is/right/left = 8/6/10; validation = 2/2/2.

This preserves the previous 24/6 size. There is no independent test set; final
claims require additional held-out trials. Fit all statistics on the new train
split, never resume the old verification experiments into these production runs.

## Validation and checkpoints

Every 1000 steps (and at a clean stop), evaluate ALL eligible validation frames.
Report normalized zero-latent L1, per-action MAE in original units, per-episode
L1 and condition-macro L1. Select best by full-validation zero-latent L1, preserving
the previous metric definition. Valid target elements, not padded actions, weight
the overall L1. Save every 10000 steps, on strict best improvement, and on clean
final/stop-after. A best improvement can therefore create an extra checkpoint
between the 10000-step milestones. `best.json` and `latest.json` identify canonical
checkpoint files without duplicating the large bundle. The best metric/step are
also saved inside checkpoints and restored on resume.

## Token-free server archival

Target: public model repository `HYHero/kroc-act-emg-checkpoints`, separate unique
run folders. The PC client creates the repository if missing. It refuses to alter
an existing repository with a different visibility. The server never needs an HF
token or HF login. Token input is hidden on the personal PC and retained only in
the relay process. Use a revocable token scoped to this repository when possible.

On the personal PC, from the delivered local working folder:

```powershell
.\.relay_venv\Scripts\python.exe scripts\relay_client.py --inspect
.\.relay_venv\Scripts\python.exe scripts\relay_client.py --repo HYHero/kroc-act-emg-checkpoints
```

Keep the PC awake and relay running throughout training. At each checkpoint:

1. Server atomically writes the bundle and a metadata snapshot, then queues it.
2. PC downloads the two explicitly allowed files over SSH and verifies SHA256.
3. PC commits checkpoint, metadata, latest pointer and (when improved) best pointer.
4. PC reads back file sizes and LFS SHA256 / Git blob IDs at that exact commit.
5. Only after success does the server receive a matching receipt, recheck its
   checkpoint, persist the receipt, and unlink that single checkpoint.pt.

Receipts, metadata and metrics remain on the server. No recursive checkpoint
deletion is used. Raw videos, EMG arrays, credentials and verification probes are
not uploaded. Failures retain the server bundle; relay retries. Training waits at
the checkpoint rather than generating an unbounded backlog. After one hour
without confirmation it stops with a recoverable checkpoint. Restart the relay,
restore the checkpoint if it was already archived, then resume.

Public token-free restore on the server, pinned to the verified commit:

```bash
PYTHONPATH=src .venv/bin/python scripts/restore_checkpoint.py --run outputs/01_baseline --which latest
bash scripts/train.sh --config configs/01_baseline.json --resume outputs/01_baseline/checkpoints/00010000/checkpoint.pt
```

Use the actual restored path printed by the restore command; best can occur at
any evaluation step. Resume restores optimizer, RNG and best-selection state, but
starts the DataLoader at a new epoch boundary, as before.

## Warmup

Keep the existing causal filters, masks and common exclusions. Copying the first
resting samples to a fictitious earlier time cannot reconstruct the continuous
filter state before recording. It can repeat noise, create seams, or use samples
that were not yet available. History padding for a future validated model is a
separate experiment, not a fix for the filter warmup. Do not relabel transient
samples as valid. Future recordings should include about 5 seconds of genuine
pre-motion history to cover filter settling plus the EMG window.
