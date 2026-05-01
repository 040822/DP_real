# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

`DP_real` is a PyTorch / Lightning / Hydra training & deployment framework for a real bimanual mobile manipulator (two 7-DoF arms with grippers, head + wrist RGB cameras, ROS-based runtime). It hosts several action policies that share the same dataset, callback, and Hydra plumbing, but have independent model code:

| Policy | Backbone | Scheduler | Source |
|--------|----------|-----------|--------|
| `DP2`  | 1D UNet over multi-view RGB (ResNet18) | DDPM | `source/policy/dp2.py` |
| `DP3`  | 1D UNet over point cloud (PointNet++)  | DDIM | `source/policy/dp3.py` |
| `DDP2` | Hierarchical Transformer + UNet (RGB)  | DDIM | `source/policy/ddp2{,_coarse,_fine}.py` |
| `DDP3` | Hierarchical Transformer + UNet (PCD)  | DDIM | `source/policy/ddp3{,_coarse,_fine}.py` |
| `CDP2` | Causal Transformer (RGB), KV-cached    | DDIM | `source/policy/cdp2.py` |
| `CDP3` | Causal Transformer (PCD), KV-cached    | DDIM | `source/policy/cdp3.py` |
| `FP3`  | 1D UNet (PCD), Consistency Flow Matching | —  | `source/policy/fp3.py` |
| `ACT`  | DETR Transformer + CVAE (RGB)          | —    | `source/policy/act.py` |
| `ARP`  | Causal Transformer autoregressive (PCD)| —    | `source/policy/arp.py` |

The README (in Chinese) is the most authoritative reference for end-user workflows; this file focuses on the things you'd otherwise have to discover by reading multiple files.

## Environment

- The repo uses **direnv + conda** (see `.envrc` → `layout anaconda robofactory`). The default conda env is `robofactory`. There is no `setup.py`/`pyproject.toml`; install deps from `requirements.txt`.
- `inference.py`, `inference_pi0.5.py`, and `replay_dataset.py` all hard-shebang `/home/lin/software/miniconda3/envs/aloha/bin/python` (legacy). Run them through your active env explicitly (`python inference.py …`).
- ROS (rospy + cv_bridge + sensor/geometry/nav/std_msgs) is required for `inference*.py` and `replay_dataset.py` only — not for training.
- The PI0.5 path is intentionally **split across two Python versions**: a Py3.10+ websocket server (`scripts/pi05_policy_server.py`, uses `lerobot`) and a Py3.8 ROS client (`inference_pi0.5.py`, uses `openpi-client`). Don't try to install both stacks in one env.

## Commands

### Train

```bash
# Recommended wrapper — it sets CUDA_VISIBLE_DEVICES, HYDRA_FULL_ERROR, and tees logs.
bash scripts/train.sh <policy_name> <task_name> <gpu_id> [info] [hydra_overrides...]

# Examples
bash scripts/train.sh dp2 collecting_objects 0
bash scripts/train.sh ddp2 stacking_playing_card 1 v2 training.lr=5e-5
```

`policy_name` resolves to `config/<policy_name>.yaml`, `task_name` to `config/task/<task_name>.yaml`. With `info` non-empty the output goes to `_outputs/<POLICY_UPPER>/<task_simple>/<info>/` instead of a timestamp dir, and a `debug.out` log file is created upfront so the dir exists even if the run crashes early.

Direct invocation (no logging wrapper):
```bash
python -u train.py --config-name=dp3 task=collecting_objects
```

Two extra Hydra-level cfg keys understood by `train.py`:
- `resume_ckpt=<path>` — passed straight to `trainer.fit(ckpt_path=...)`.
- `two_train_ckpt=<path>` — *load weights only* (`strict=False`) into a fresh model, used for two-stage training of DDP2/DDP3 (coarse → coarse+fine).

Batch / sweep across multiple `(policy, task)` combos with GPU-memory-aware tmux dispatch:
```bash
python scripts/train_many_tasks.py        # reads scripts/train_many_tasks.yaml
```
Edit `scripts/train_many_tasks.yaml` to set `scheduler.gpus`, `mem_per_task_gb`, and the `tasks` list (list-valued fields are Cartesian-product expanded).

### Inference (real robot, ROS)

```bash
python inference.py --ckpt_dir <full_path_to.ckpt> [--max_publish_step 250 --publish_rate 40 ...]
```
`--ckpt_dir` is a **file path**, not a directory — the script does `torch.load(args.ckpt_dir, pickle_module=dill)`. The checkpoint must contain the `cfg` dict written by `SaveConfigCallback` (all training in this repo does so by default); the policy is rebuilt via `hydra.utils.instantiate(payload['cfg']['policy'])` and weights are loaded with `strict=False`.

PI0.5 (split client/server):
```bash
# Server (Python 3.10+)
pip install -r requirements-pi05-server.txt
python scripts/pi05_policy_server.py --lerobot_model lerobot/pi05_base --lerobot_task "..." --port 8000

# Connectivity smoke test
python scripts/pi05_server_ping.py --host 127.0.0.1 --port 8000

# ROS client (Python 3.8)
pip install -r requirements-pi05-client.txt
python inference_pi0.5.py --pi05_server_host 127.0.0.1 --pi05_server_port 8000 --lerobot_task "..."
```

### Replay an HDF5 episode through ROS

```bash
python replay_dataset.py --dataset_dir <path/to/task.h5> --episode_idx 1 [--max_publish_step 250]
```

### Inspect data

```bash
python scripts/list_items.py --hdf5_file data/<task>.h5    # prints HDF5 tree with shapes/dtypes
python scripts/read_hdf5.py  --hdf5_file data/<task>.h5    # JSON summary / full export
```

### Other

```bash
bash scripts/zip.sh                  # bundle repo into DP_real.zip, excluding _outputs/, data/, outputs/
python scripts/ckpt_change.py        # template for in-place ckpt['cfg'] surgery (e.g. renaming _target_)
```

There are **no `pytest` tests, no linter, no formatter** wired up in this repo. There is no test runner, so don't claim tests have been added without setting one up.

## Architecture: how the pieces compose

### Hydra config tree

Every policy at the repo root (`config/<policy>.yaml`) shares the same skeleton:
- `defaults: [_self_, task: default_task]` — task is overridable from CLI (`task=<name>`).
- A top-level `policy:` block whose `_target_` points at `source.policy.<x>.<Class>`. `optimizer_cfg`, `scheduler_cfg`, and `shape_meta` are passed via OmegaConf interpolation.
- `task.dataset._target_` points at one of `source.dataset.dataset2d_real.Dataset2D` / `dataset2dc_real.Dataset2D` (these are real-robot HDF5 datasets — note both files are named `Dataset2D` despite different module names).
- `hydra.run.dir: _outputs/${policy_name}/${task_name}/${now:%Y.%m.%d.%H.%M.%S}` — every run has its own dir; checkpoints land in `<run_dir>/checkpoints/`, the embedded `cfg` is restored at inference time.

Custom OmegaConf resolver: `OmegaConf.register_new_resolver("eval", eval, ...)` is registered at top of `train.py`, so configs can use `${eval:'expr'}` (e.g. `pad_after: ${eval:'${n_action_steps}-1'}`).

The full task config is interpolated into `shape_meta`, which every policy reads to figure out observation/action dimensions — adding a new modality means editing the task yaml, not the policy code.

### `train.py` flow

1. Resolves cfg, sets seed, instantiates `cfg.policy` (a `LightningModule`).
2. **DDP2 special case**: if `cfg.policy_name == "DDP2"`, derives `sample_horizon = (H-1)*internal + H + 1` from the coarse policy's horizon, builds a `torch.linspace`-derived index `idx`, calls `model.set_idx(idx)`, then *mutates* `cfg.task.dataset.horizon` and `cfg.task.dataset.pad_after`. Any DDP-style policy that needs hierarchical sample windows must follow the same pattern. (DDP3 has the same structure but is currently not gated by this branch — re-check before training DDP3.)
3. Instantiates dataset, splits 95/5 into train/val, builds `DataLoader`s from `cfg.dataloader.{train,val}`.
4. Calls `model.set_normalizer(dataset.get_normalizer())` — datasets compute statistics over `action`/`qpos` and return identity-or-`limits` normalizers; image keys (those whose name starts with `cam`) get a manual `1/255` normalizer in `dataset2d_real.py`.
5. Builds five callbacks: `LearningRateMonitor`, `ModelCheckpoint` (configured by `cfg.checkpoint`), `ModelAveragingCallback` (EMA, see below), `SaveConfigCallback` (writes resolved cfg into every ckpt), `SampleCallback` (runs `predict_action` on a val batch each epoch and logs `val_action_mse_error`), plus a `TQDMProgressBar`.
6. Logs to WandB (`logging.project = DP_real` by default).

### Policy interface

`source/policy/base_policy.py` defines two classes:
- `BaseImagePolicy` (legacy, `ModuleAttrMixin`): defines just `predict_action`/`reset`/`set_normalizer`.
- `BasePolicy(LightningModule)`: implements `training_step`/`validation_step`/`configure_optimizers` (AdamW + `get_scheduler`). Subclasses provide `compute_loss(batch)` and `predict_action(obs_dict)`. The validation step **calls `reset()` then `predict_action`** — stateful policies (CDP*, hierarchical DDP*) must implement `reset()` correctly or validation will leak state across batches.

Hierarchical DDP2/DDP3 (`source/policy/ddp{2,3}.py`) do *not* extend `BasePolicy`; they are `LightningModule`s that compose `Coarse_DP{2,3}` and `Fine_DP{2,3}` directly and own their own `training_step`/`validation_step`/`configure_optimizers`. A few things to know if editing them:
- `coarse_cache` / `coarse_cache_idx` are runtime state cleared by `reset()`.
- `predict_action` has a `predict_type ∈ {fine_dp, linear, cubic_spline, minimum_snap, only_coarse}` switch (default `cubic_spline`) that picks how to upsample coarse waypoints — when debugging weird trajectories this is the first place to look.
- The optimizer collects two parameter groups per sub-policy (transformer-ish weights vs. obs encoder) with different `weight_decay` from `optimizer.transformer_weight_decay`/`optimizer.obs_encoder_weight_decay`.

### Dataset layer

Both `source/dataset/dataset2d_real.py` and `dataset2dc_real.py` define a class **named** `Dataset2D` (different files, both used). They:
- Read an HDF5 file expected to contain top-level keys: `episode_ends`, `qpos`, `action`, plus image keys whose names start with `cam` (e.g. `cam_high`).
- Build sample indices via `source/common/sampler.py:create_indices`, padding before/after by `pad_before`/`pad_after`.
- Support `obs_only_n_steps=True` to load only the first `obs_n_steps` frames of each `cam*` window — important for memory when `horizon` is much larger than `n_obs_steps` (e.g. ACT with `chunk_size=100, n_obs_steps=1`).
- `use_mem=True` slurps the entire HDF5 into RAM up front; `False` keeps `h5py.File` open lazily.
- Differ in normalizer behavior for `cam` keys: `dataset2d_real.py` returns a manual `1/255` image normalizer; `dataset2dc_real.py` uses identity. Choose the one that matches whatever the policy's `obs_encoder` expects.

### Checkpoints & resuming

`SaveConfigCallback` (`source/common/callbacks.py`) embeds the resolved Hydra cfg as `checkpoint['cfg']`. Inference scripts rely on this — if you rename a policy's `_target_` path, **old checkpoints won't load** until you patch the cfg. `scripts/ckpt_change.py` is a small one-off template for that surgery; copy it and adjust paths/keys.

`ModelAveragingCallback` extends Lightning's `EMAWeightAveraging` but defers `AveragedModel` construction until `on_fit_start` (so the average lives on the GPU from the start). EMA weights are swapped in-place during validation.

### Inference plumbing (`inference.py`)

Two helper classes:
- `RosOperator` — manages bounded `deque`s for left/right/front RGB (and optional depth), left/right `JointState`, and `Odometry`; `get_frame()` returns a time-synced tuple by taking `min(latest stamps)` and discarding older messages. The continuous-publish thread is gated by `puppet_arm_publish_lock` (acquired-by-default to mean "no publisher running"; release to ask the running publisher to stop).
- `EnvRunner` — keeps a `deque(maxlen=n_obs_steps+1)` of obs dicts, stacks the last `n_obs_steps` for the policy, and builds the `obs_dict` Tensor the policy expects.

`get_model_input` is a small bridge — it currently hard-codes the input modalities (`cam_high`, `qpos` only; left/right cameras commented out). If your task config adds modalities, update this function or the policy can't see them.

## Things to be careful about

- **Hydra checkpoint coupling**: changing a `_target_` path in `source/policy/` or `source/model/` invalidates all prior checkpoints. Either rename via a deprecation alias or run `scripts/ckpt_change.py` over old ckpts.
- **Dataset class name collision**: both `dataset2d_real.py` and `dataset2dc_real.py` export `Dataset2D`. Always import via the full module path.
- **DDP2 horizon mutation**: `train.py` rewrites `cfg.task.dataset.horizon` and `pad_after` *only when `policy_name == "DDP2"`*. If you fork a new hierarchical policy, replicate that branch or `set_idx` will not match the dataset windows.
- **Stateful policies + validation**: any policy that caches internal state (CDP2/CDP3/DDP2/DDP3 cache coarse output, KV) must implement `reset()` correctly — the default `validation_step` calls `reset()` before each `predict_action`.
- **Output paths**: `_outputs/`, `data/`, `outputs/`, `temp/`, `Temp/` are all gitignored. The training shell wrapper writes logs to `Temp/` if `info` is empty.
- **Real-robot inference is destructive**: `inference.py` and `replay_dataset.py` publish JointState/Twist commands to live `puppet_arm_*_cmd` and `cmd_vel` topics. Don't run them without checking the robot is in a safe state.
