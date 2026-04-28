#!/usr/bin/env python3
"""Batch training scheduler with GPU-memory-aware dispatch and tmux orchestration.

Core behavior:
- Read tasks from a YAML file.
- Expand list-valued task fields into multiple concrete tasks.
- Dispatch tasks only when target GPU has enough free memory.
- Rule: every mem_per_task_gb free memory corresponds to one runnable slot.
- Launch one tmux window per task in a single session.
- Load .bashrc and direnv environment before running training command.
"""

from __future__ import annotations

import itertools
import re
import shlex
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


try:
	import yaml
except ImportError as exc:  # pragma: no cover
	raise SystemExit("PyYAML 未安装，请先安装: pip install pyyaml") from exc


REQUIRED_FIELDS = ("policy_name", "task_name", "addition_info", "seed")
DEFAULT_CONFIG_BASENAME = "train_many_tasks.yaml"

 # ========= Tool  ============
@dataclass(frozen=True)
class TrainTask:
	policy_name: str
	task_name: str
	addition_info: str
	seed: int
	extra_args: tuple[str, ...]
	index: int


@dataclass
class RunningTask:
	task: TrainTask
	gpu_id: int
	window_name: str
	start_time: float


def now_str() -> str:
	return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
	print(f"[{now_str()}] {msg}", flush=True)


def run_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
	return subprocess.run(cmd, check=check, text=True, capture_output=True)


def check_binary_exists(name: str) -> None:
	result = subprocess.run(["bash", "-lc", f"command -v {shlex.quote(name)}"], text=True)
	if result.returncode != 0:
		raise SystemExit(f"缺少命令: {name}")


def parse_gpu_ids(text: str) -> list[int]:
	raw = [x.strip() for x in text.split(",") if x.strip()]
	if not raw:
		raise SystemExit("--gpus 不能为空")
	try:
		gpu_ids = [int(x) for x in raw]
	except ValueError as exc:
		raise SystemExit(f"--gpus 格式错误: {text}") from exc
	if len(set(gpu_ids)) != len(gpu_ids):
		raise SystemExit("--gpus 不能包含重复GPU")
	return gpu_ids


def parse_gpu_ids_from_yaml(value: Any) -> list[int]:
	if isinstance(value, str):
		return parse_gpu_ids(value)
	if isinstance(value, list) and value:
		try:
			gpu_ids = [int(x) for x in value]
		except ValueError as exc:
			raise SystemExit("scheduler.gpus 列表必须是整数") from exc
		if len(set(gpu_ids)) != len(gpu_ids):
			raise SystemExit("scheduler.gpus 不能包含重复GPU")
		return gpu_ids
	raise SystemExit("scheduler.gpus 必须是非空字符串(如 '0,1')或整数列表")


def load_yaml(path: Path) -> dict[str, Any]:
	if not path.exists():
		raise SystemExit(f"配置文件不存在: {path}")
	data = yaml.safe_load(path.read_text(encoding="utf-8"))
	if not isinstance(data, dict):
		raise SystemExit("YAML 顶层必须是对象，且包含 tasks 字段")
	if "tasks" not in data:
		raise SystemExit("YAML 缺少 tasks 字段")
	if not isinstance(data["tasks"], list):
		raise SystemExit("YAML 中 tasks 必须是列表")
	return data


def to_bool(value: Any, field_name: str) -> bool:
	if isinstance(value, bool):
		return value
	if isinstance(value, str):
		lower = value.strip().lower()
		if lower in {"1", "true", "yes", "y", "on"}:
			return True
		if lower in {"0", "false", "no", "n", "off"}:
			return False
	raise SystemExit(f"{field_name} 必须是布尔值")


def parse_scheduler_config(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
	"""
	解析调度器相关配置，返回一个包含调度参数的字典。
	TODO: 如果需要支持更多调度参数，可以在这里添加解析逻辑，并在 YAML 配置文件中进行相应的定义和说明。
	"""
	scheduler = config.get("scheduler", {})
	if scheduler is None:
		scheduler = {}
	if not isinstance(scheduler, dict):
		raise SystemExit("scheduler 必须是对象")

	if "gpus" not in scheduler:
		raise SystemExit("YAML 缺少 scheduler.gpus")
	gpu_ids = parse_gpu_ids_from_yaml(scheduler["gpus"])

	mem_per_task_gb = float(scheduler.get("mem_per_task_gb", 10.0))
	if mem_per_task_gb <= 0:
		raise SystemExit("scheduler.mem_per_task_gb 必须大于0")

	poll_interval = int(scheduler.get("poll_interval", 20))
	if poll_interval <= 0:
		raise SystemExit("scheduler.poll_interval 必须大于0")

	task_start_interval = int(scheduler.get("task_start_interval", 5))
	if task_start_interval <= 0:
		raise SystemExit("scheduler.task_start_interval 必须大于0")

	session_name = str(scheduler.get("session", "")).strip()
	if not session_name:
		session_name = f"train_many_{datetime.now().strftime('%m%d_%H%M%S')}"

	workdir = Path(str(scheduler.get("workdir", ".")))
	if not workdir.is_absolute():
		workdir = (config_path.parent / workdir).resolve()
	else:
		workdir = workdir.resolve()

	train_script = Path(str(scheduler.get("train_script", "scripts/train.sh")))
	if not train_script.is_absolute():
		train_script = (config_path.parent / train_script).resolve()
	else:
		train_script = train_script.resolve()

	allow_existing_session = to_bool(
		scheduler.get("allow_existing_session", False),
		"scheduler.allow_existing_session",
	)
	dry_run = to_bool(scheduler.get("dry_run", False), "scheduler.dry_run")
	extra_args_expand = to_bool(
		scheduler.get("extra_args_expand", False),
		"scheduler.extra_args_expand",
	)
	addition_info_template = to_bool(
		scheduler.get("addition_info_template", False),
		"scheduler.addition_info_template",
	)

	return {
		"gpu_ids": gpu_ids,
		"mem_per_task_gb": mem_per_task_gb,
		"mem_per_task_mib": int(mem_per_task_gb * 1024),
		"poll_interval": poll_interval,
		"session_name": session_name,
		"workdir": workdir,
		"train_script": train_script,
		"allow_existing_session": allow_existing_session,
		"dry_run": dry_run,
		"task_start_interval": task_start_interval,
		"extra_args_expand": extra_args_expand,
		"addition_info_template": addition_info_template,
	}


def ensure_list(value: Any) -> list[Any]:
	if isinstance(value, list):
		if not value:
			raise ValueError("列表字段不能为空")
		return value
	return [value]


def normalize_extra_args(value: Any) -> tuple[str, ...]:
	if value is None:
		return ()
	if isinstance(value, list):
		return tuple(str(x) for x in value)
	if isinstance(value, str):
		return tuple(shlex.split(value))
	return (str(value),)


def format_hydra_value(value: Any) -> str:
	if isinstance(value, bool):
		return "true" if value else "false"
	if value is None:
		return "null"
	if isinstance(value, (int, float)):
		return str(value)
	if isinstance(value, str):
		return value
	# 对复杂类型使用 YAML flow 风格，便于作为 Hydra 覆盖参数传递。
	return yaml.safe_dump(value, default_flow_style=True, sort_keys=False).strip()


def expand_extra_args_variants_with_context(
	value: Any,
) -> list[tuple[tuple[str, ...], dict[str, Any]]]:
	if value is None:
		return [((), {})]

	# 展开模式：
	# extra_args:
	#   coarse_horizon: [1, 2]
	#   internal: [3, 4]
	# -> 4 组组合
	if isinstance(value, dict):
		items = list(value.items())
		if not items:
			return [((), {})]
		option_sets: list[list[tuple[str, Any]]] = []
		for key, raw_v in items:
			try:
				vals = ensure_list(raw_v)
			except ValueError as exc:
				raise SystemExit(f"extra_args.{key} 字段错误: {exc}") from exc
			option_sets.append([(str(key), v) for v in vals])

		variants: list[tuple[tuple[str, ...], dict[str, Any]]] = []
		for combo in itertools.product(*option_sets):
			ctx = {k: v for k, v in combo}
			variants.append(
				(
					tuple(f"{k}={format_hydra_value(v)}" for k, v in combo),
					ctx,
				)
			)
		return variants

	# 非字典时保持原有透传语义（兼容旧配置）
	return [(normalize_extra_args(value), {})]


def render_addition_info(template: str, context: dict[str, Any], field_name: str) -> str:
	pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

	def repl(match: re.Match[str]) -> str:
		key = match.group(1)
		if key not in context:
			raise SystemExit(
				f"{field_name} 引用了未定义变量: {key}。"
				"请在 extra_args 中提供同名键，或关闭 scheduler.addition_info_template"
			)
		return format_hydra_value(context[key])

	return pattern.sub(repl, template)


def expand_tasks(
	config: dict[str, Any],
	extra_args_expand: bool = False,
	addition_info_template: bool = False,
) -> list[TrainTask]:
	defaults = config.get("defaults", {})
	if defaults is None:
		defaults = {}
	if not isinstance(defaults, dict):
		raise SystemExit("defaults 必须是对象")

	expanded: list[TrainTask] = []
	next_index = 1
	for i, item in enumerate(config["tasks"], start=1):
		if not isinstance(item, dict):
			raise SystemExit(f"tasks[{i}] 必须是对象")

		merged = dict(defaults)
		merged.update(item)

		missing = [f for f in REQUIRED_FIELDS if f not in merged]
		if missing:
			raise SystemExit(f"tasks[{i}] 缺少字段: {missing}")

		try:
			field_values = [ensure_list(merged[name]) for name in REQUIRED_FIELDS]
		except ValueError as exc:
			raise SystemExit(f"tasks[{i}] 字段错误: {exc}") from exc

		if extra_args_expand:
			extra_args_variants = expand_extra_args_variants_with_context(
				merged.get("extra_args")
			)
		else:
			extra_args_variants = [(normalize_extra_args(merged.get("extra_args")), {})]

		for policy_name, task_name, addition_info, seed in itertools.product(*field_values):
			try:
				seed_int = int(seed)
			except ValueError as exc:
				raise SystemExit(f"tasks[{i}] 的 seed 无法转为整数: {seed}") from exc
			for extra_args, extra_ctx in extra_args_variants:
				addition_info_str = str(addition_info)
				if addition_info_template:
					addition_info_str = render_addition_info(
						addition_info_str,
						extra_ctx,
						f"tasks[{i}].addition_info",
					)
				expanded.append(
					TrainTask(
						policy_name=str(policy_name),
						task_name=str(task_name),
						addition_info=addition_info_str,
						seed=seed_int,
						extra_args=extra_args,
						index=next_index,
					)
				)
				next_index += 1

	if not expanded:
		raise SystemExit("配置未展开出任何任务")
	return expanded


def ensure_train_script(train_script: Path) -> None:
	if not train_script.exists():
		raise SystemExit(f"训练脚本不存在: {train_script}")


def tmux_session_exists(session_name: str) -> bool:
	result = subprocess.run(
		["tmux", "has-session", "-t", session_name],
		text=True,
		capture_output=True,
	)
	return result.returncode == 0


def ensure_tmux_session(session_name: str, allow_existing: bool, dry_run: bool) -> None:
	exists = tmux_session_exists(session_name)
	if exists and not allow_existing:
		raise SystemExit(
			f"tmux session 已存在: {session_name}。请更换 --session 或传 --allow-existing-session"
		)
	if exists:
		log(f"复用已有 tmux session: {session_name}")
		return
	if dry_run:
		log(f"[dry-run] 将创建 tmux session: {session_name}")
		return
	run_cmd(["tmux", "new-session", "-d", "-s", session_name, "-n", "dispatcher"])
	log(f"已创建 tmux session: {session_name}")


def list_windows(session_name: str) -> set[str]:
	result = subprocess.run(
		["tmux", "list-windows", "-t", session_name, "-F", "#{window_name}"],
		text=True,
		capture_output=True,
	)
	if result.returncode != 0:
		return set()
	lines = [x.strip() for x in result.stdout.splitlines() if x.strip()]
	return set(lines)


def get_free_memories_mib(gpu_ids: list[int]) -> dict[int, int]:
	result = run_cmd(
		[
			"nvidia-smi",
			"--query-gpu=index,memory.free",
			"--format=csv,noheader,nounits",
		]
	)
	free_map: dict[int, int] = {}
	for line in result.stdout.splitlines():
		parts = [x.strip() for x in line.split(",")]
		if len(parts) != 2:
			continue
		try:
			gid = int(parts[0])
			mem = int(parts[1])
		except ValueError:
			continue
		free_map[gid] = mem

	missing = [gid for gid in gpu_ids if gid not in free_map]
	if missing:
		raise SystemExit(f"以下GPU不存在或不可见: {missing}")

	return {gid: free_map[gid] for gid in gpu_ids}


def sanitize_window_name(task: TrainTask) -> str:
	base = f"job{task.index}_{task.task_name}_{task.policy_name}_s{task.seed}"
	keep = []
	for ch in base:
		if ch.isalnum() or ch in {"_", "-", "."}:
			keep.append(ch)
		else:
			keep.append("_")
	return "".join(keep)[:120]


def build_train_cmd(train_script: Path, task: TrainTask, gpu_id: int) -> str:
	parts = [
		"bash",
		str(train_script),
		task.policy_name,
		task.task_name,
		str(gpu_id),
		task.addition_info,
		f"training.seed={task.seed}",
	]
	parts.extend(task.extra_args)
	return " ".join(shlex.quote(x) for x in parts)


def build_tmux_command(workdir: Path, train_cmd: str) -> str:
	quoted_train = shlex.quote(train_cmd)
	inner = (
		"source ~/.bashrc >/dev/null 2>&1 || true; "
		"if command -v direnv >/dev/null 2>&1; then "
		"direnv allow . >/dev/null 2>&1 || true; "
		f"direnv exec . bash -lc {quoted_train}; "
		"else "
		f"bash -lc {quoted_train}; "
		"fi"
	)
	return f"cd {shlex.quote(str(workdir))} && bash -ic {shlex.quote(inner)}"


def launch_task(
	session_name: str,
	workdir: Path,
	train_script: Path,
	task: TrainTask,
	gpu_id: int,
	dry_run: bool,
	task_start_interval: int,
) -> str:
	window_name = sanitize_window_name(task)
	train_cmd = build_train_cmd(train_script, task, gpu_id)
	full_cmd = build_tmux_command(workdir, train_cmd)
	if dry_run:
		log(f"[dry-run] window={window_name} gpu={gpu_id} cmd={full_cmd}")
		return window_name

	time.sleep(task_start_interval)  # 避免过快创建多个窗口导致tmux命令冲突

	run_cmd(
		[
			"tmux",
			"new-window",
			"-d",
			"-t",
			session_name,
			"-n",
			window_name,
			full_cmd,
		]
	)
	log(
		"已启动任务: "
		f"idx={task.index} policy={task.policy_name} task={task.task_name} "
		f"seed={task.seed} gpu={gpu_id} window={window_name}"
	)
	return window_name


def pick_gpu_for_task(
	free_mib: dict[int, int],
	mem_per_task_mib: int,
) -> int | None:
	candidates: list[tuple[int, int]] = []
	for gpu_id, free in free_mib.items():
		if free >= mem_per_task_mib:
			candidates.append((free, gpu_id))
	if not candidates:
		return None
	candidates.sort(reverse=True)
	return candidates[0][1]


def log_task_plan(tasks: list[TrainTask], train_script: Path) -> None:
	log(f"任务总数: {len(tasks)}")
	for task in tasks:
		parts = [
			"bash",
			str(train_script),
			task.policy_name,
			task.task_name,
			"<gpu_id>",
			task.addition_info,
			# f"training.seed={task.seed}",
		]
		# 透传到 train.sh，第5个参数及之后会继续传给 train.py。
		parts.extend(task.extra_args)
		preview_cmd = " ".join(shlex.quote(x) for x in parts)
		log(
			f"任务计划[{task.index}]: policy={task.policy_name} task={task.task_name} "
			f"addition_info={task.addition_info} seed={task.seed} extra_args={list(task.extra_args)} "
			f"cmd={preview_cmd}"
		)


def main() -> None:
	if len(sys.argv) > 1:
		raise SystemExit(
			"train_many_tasks.py 不再接受命令行参数，请把所有调度参数写入 YAML 配置文件"
		)

	# 检查命令是否存在
	check_binary_exists("tmux")
	check_binary_exists("nvidia-smi")

	config_path = Path(__file__).with_name(DEFAULT_CONFIG_BASENAME).resolve()
	config = load_yaml(config_path) # 加载配置文件
	scheduler = parse_scheduler_config(config, config_path)

	gpu_ids = scheduler["gpu_ids"]
	workdir = scheduler["workdir"]
	train_script = scheduler["train_script"]
	mem_per_task_mib = scheduler["mem_per_task_mib"]
	poll_interval = scheduler["poll_interval"]
	session_name = scheduler["session_name"]
	allow_existing_session = scheduler["allow_existing_session"]
	dry_run = scheduler["dry_run"]
	task_start_interval = scheduler["task_start_interval"]
	extra_args_expand = scheduler["extra_args_expand"]
	addition_info_template = scheduler["addition_info_template"]

	if not workdir.exists():
		raise SystemExit(f"scheduler.workdir 不存在: {workdir}")
	ensure_train_script(train_script)

	tasks = expand_tasks(
		config,
		extra_args_expand=extra_args_expand,
		addition_info_template=addition_info_template,
	)
	log_task_plan(tasks, train_script)
	queue: deque[TrainTask] = deque(tasks)
	running: dict[str, RunningTask] = {}
	done_count = 0
	ensure_tmux_session(session_name, allow_existing_session, dry_run)

	log(
		f"调度开始: session={session_name} tasks={len(tasks)} gpus={gpu_ids} "
		f"mem_per_task={scheduler['mem_per_task_gb']:.2f}GiB poll={poll_interval}s"
	)

	while queue or running:
		live_windows = list_windows(session_name) if not dry_run else set(running.keys())
		finished = [w for w in list(running.keys()) if w not in live_windows]
		for window in finished:
			rt = running.pop(window)
			elapsed = int(time.time() - rt.start_time)
			done_count += 1
			log(
				"任务结束: "
				f"idx={rt.task.index} task={rt.task.task_name} policy={rt.task.policy_name} "
				f"seed={rt.task.seed} gpu={rt.gpu_id} elapsed={elapsed}s"
			)

		free_mib = get_free_memories_mib(gpu_ids)
		running_count = {gid: 0 for gid in gpu_ids}
		for rt in running.values():
			running_count[rt.gpu_id] = running_count.get(rt.gpu_id, 0) + 1

		launched = 0
		while queue:
			target_gpu = pick_gpu_for_task(free_mib, mem_per_task_mib)
			if target_gpu is None:
				break
			task = queue.popleft()
			window_name = launch_task(
				session_name=session_name,
				workdir=workdir,
				train_script=train_script,
				task=task,
				gpu_id=target_gpu,
				dry_run=dry_run,
				task_start_interval=task_start_interval,
			)
			if dry_run:
				done_count += 1
			else:
				running[window_name] = RunningTask(
					task=task,
					gpu_id=target_gpu,
					window_name=window_name,
					start_time=time.time(),
				)
			# 同一轮调度内做本地扣减，避免基于单次 nvidia-smi 采样在同卡超发。
			free_mib[target_gpu] = max(0, free_mib[target_gpu] - mem_per_task_mib)
			log(f"扣减后的 free: gpu{target_gpu}={free_mib[target_gpu]}MiB")
			running_count[target_gpu] = running_count.get(target_gpu, 0) + 1
			launched += 1

		if queue:
			gpu_status = " ".join(
				f"gpu{gid}:free={free_mib[gid]}MiB running={running_count.get(gid, 0)}"
				for gid in gpu_ids
			)
			log(
				f"状态: pending={len(queue)} running={len(running)} done={done_count} {gpu_status}"
			)

		if not queue:
			if running:
				log(f"队列已清空，直接结束调度器（仍有{len(running)}个任务在tmux中运行）")
			break

		if queue and launched == 0:
			log("当前显存不足，等待下一轮调度...")
		if dry_run:
			continue
		time.sleep(poll_interval)

	log("所有任务调度完成")
	if not dry_run:
		log(f"可使用以下命令查看会话: tmux attach -t {session_name}")


if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		print("\n收到中断信号，已退出调度器", file=sys.stderr)
		sys.exit(130)
