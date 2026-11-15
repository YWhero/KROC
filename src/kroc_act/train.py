"""Single-GPU ACT training with full validation and verified PC-relay archival."""
import argparse
import json
import math
import random
from pathlib import Path

from .checkpoints import save_checkpoint
from .paths import PROJECT_ROOT, atomic_json, project_path
from .study import sha256_file


def resolve_config(path):
    path = project_path(path)
    cfg = json.loads(path.read_text(encoding="utf-8"))
    for key in ["dataset_root", "manifest", "splits", "output_dir", "annotations"]:
        if cfg.get(key) is not None:
            cfg[key] = str(project_path(path.parent / cfg[key]))
    if cfg["keep_checkpoints"] < 1 or cfg["save_freq"] < 1:
        raise ValueError("Positive checkpoint frequency/retention required")
    for key in ["steps", "batch_size", "eval_freq", "log_freq"]:
        if cfg[key] < 1:
            raise ValueError(f"Positive {key} required")
    if cfg.get("val_batches") is not None and cfg["val_batches"] < 1:
        raise ValueError("val_batches must be null (full validation) or positive")
    if cfg.get("checkpoint_archive", {}).get("mode", "relay") != "relay":
        raise ValueError("Only token-free PC relay archival is supported")
    if cfg.get("checkpoint_archive") and cfg.get("verify_checkpoints"):
        raise ValueError("Public archived checkpoints must not include raw verification probes")
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", type=Path, help="Trusted local checkpoint.pt; new epoch boundary")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    cfg = resolve_config(args.config)
    end_step = min(cfg["steps"], args.stop_after) if args.stop_after is not None else cfg["steps"]
    if end_step <= 0:
        raise ValueError("Positive --stop-after required")
    if args.dry_run:
        print(json.dumps(cfg, indent=2, ensure_ascii=False))
        return

    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from .data import SharedDataset
    from .evaluation import evaluate
    from .model import ACTExperiment

    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    train, val = SharedDataset(cfg, "train"), SharedDataset(cfg, "val")
    cfg.update(state_dim=train.features["observation.state"]["shape"][0],
               action_dim=train.features["action"]["shape"][0],
               emg_channels=len(train.manifest["channels"]),
               camera_shapes={k: train.features[k]["shape"] for k in train.camera_keys})
    provenance = {key: sha256_file(cfg[key]) for key in ["manifest", "splits", "annotations"] if cfg.get(key)}
    provenance.update(lerobot_commit="240b4a0314ae0879cdd928c7f4bdc1eee9a01b3b",
                      train_frames=len(train), val_frames=len(val),
                      train_excluded=train.excluded, val_excluded=val.excluded)
    run = project_path(cfg["output_dir"])
    resume = None
    if args.resume:
        resume = torch.load(project_path(args.resume), map_location="cpu", weights_only=False)
        if resume["experiment"] != cfg or resume["provenance"] != provenance:
            raise ValueError("Resume configuration/data provenance differs")
        statistics = resume["statistics"]
    else:
        if run.exists():
            raise FileExistsError(f"Choose a new output_dir: {run}")
        statistics = train.train_statistics()
        run.mkdir(parents=True)
    model = ACTExperiment(cfg, statistics).to(device)
    optimizer = torch.optim.AdamW(model.optimizer_groups(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    step = 0
    best_value, best_step = None, None
    if resume:
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        step = resume["step"]
        torch.set_rng_state(resume["rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(resume["cuda_rng"])
        random.setstate(resume["python_rng"])
        np.random.set_state(resume["numpy_rng"])
        generator.set_state(resume["loader_rng"])
        best_value = resume.get("training_state", {}).get("best_val_l1")
        best_step = resume.get("training_state", {}).get("best_step")
    records = [("experiment.json", cfg), ("statistics.json", statistics), ("provenance.json", provenance),
               ("splits.json", json.loads(project_path(cfg["splits"]).read_text()))]
    if train.annotations:
        records += [("study_annotations.json", train.annotations),
                    ("split_summary.json", {"episode_counts": train.split_condition_counts,
                     "selection_metric": "val_l1_zero_latent", "validation": "all eligible frames",
                     "test_set": "not present; validation selects models and is not independent testing"})]
    sources = sorted((PROJECT_ROOT / "src/kroc_act").glob("*.py"))
    records.append(("code_provenance.json", {str(p.relative_to(PROJECT_ROOT)): sha256_file(p) for p in sources}))
    for name, content in records:
        atomic_json(run / name, content)
    loader = DataLoader(train, batch_size=cfg["batch_size"], shuffle=True, generator=generator,
                        num_workers=0, drop_last=False)
    val_loader = DataLoader(val, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
    def move(batch):
        return {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
    iterator = iter(loader)
    with project_path(run / "metrics.jsonl").open("a", encoding="utf-8") as log:
        while step < end_step:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = model(move(batch))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at step {step}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            step += 1
            record = {"step": step, "loss": loss.item(), "grad_norm": float(norm), **metrics}
            if cfg["variant"] == "window":
                record["emg_grad_norm"] = sum(p.grad.detach().square().sum().item()
                    for p in model.emg_encoder.parameters() if p.grad is not None) ** 0.5
            validation, improved = None, False
            if step % cfg["eval_freq"] == 0 or step == end_step:
                validation = evaluate(model, val_loader, device, cfg.get("val_batches"))
                value = validation["val_l1_zero_latent"]
                if not math.isfinite(value):
                    raise FloatingPointError("Nonfinite validation metric")
                improved = best_value is None or value < best_value
                if improved:
                    best_value, best_step = value, step
                record.update(validation)
                record.update(best_val_l1=best_value, best_step=best_step)
            if step % cfg["log_freq"] == 0 or validation is not None:
                line = json.dumps(record, allow_nan=False)
                print(line, flush=True)
                log.write(line + "\n")
                log.flush()
            if step % cfg["save_freq"] == 0 or step == end_step or improved:
                save_checkpoint(run, step, model, optimizer, cfg, statistics, provenance, generator,
                                probe=move(batch) if cfg.get("verify_checkpoints", False) else None,
                                training_state={"best_val_l1": best_value, "best_step": best_step},
                                is_best=improved, validation=validation)


if __name__ == "__main__":
    main()
