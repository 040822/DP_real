# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.9/PyTorch project for training and deploying real-robot diffusion policies. Entry points live at the repository root: `train.py` uses Hydra for training, `inference.py` runs ROS inference, and `replay_dataset.py` replays recorded trajectories. Put policy wrappers in `source/policy/`, architectures in `source/model/<POLICY>/`, dataset adapters in `source/dataset/`, and reusable utilities in `source/common/`. Hydra policy and task configuration belongs in `config/`; keep the checked-in task template at `config/task/default_task.yaml`. Data conversion, robot control, and batch-training tools live under `scripts/`.

## Build, Test, and Development Commands

Create the documented environment with `conda create -n dp_real python=3.9`, then activate it and install the dependencies described in `README.md`. The current `requirements.txt` ends with an incomplete `open3d==` pin; choose the project-compatible Open3D version before using `pip install -r requirements.txt`.

- `python -u train.py --config-name=dp3 task=<task>` starts one Hydra training run.
- `bash scripts/train.sh ddp2 collecting_objects 0 debug` launches training on GPU 0 and records output under `_outputs/`.
- `python -m py_compile train.py inference.py replay_dataset.py` performs a lightweight syntax check.
- `python inference.py --help` or `python replay_dataset.py --help` inspects deployment options; full execution requires ROS and robot-specific packages.

## Coding Style & Naming Conventions

Use four-space indentation and conventional Python style. Name modules, functions, variables, and YAML files with `snake_case`; use `PascalCase` for classes and uppercase algorithm directory names where the existing layout does (`source/model/DDP3/`). Keep Hydra `_target_` paths synchronized when moving classes. No formatter or linter is configured, so keep imports organized and follow the surrounding file.

## Testing Guidelines

No dedicated automated test suite or coverage threshold is currently configured. Run syntax checks plus a targeted smoke run for the policy and task you changed. New tests should use `pytest`, live in `tests/`, and follow `test_<behavior>.py`; document any fixtures or hardware requirements.

## Commit & Pull Request Guidelines

Recent history uses short, imperative Chinese summaries such as `调整训练代码`; Conventional Commits are not required. Keep each commit focused and describe the affected policy or workflow. Pull requests should explain the change, list Hydra overrides and validation commands, link related issues, and include relevant logs or before/after metrics. Never commit datasets, checkpoints, W&B artifacts, ROS credentials, or generated `Temp/`, `outputs/`, and `_outputs/` content.
