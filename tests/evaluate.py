from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.disaster_env import DisasterEnv
from src.models.manager import GraphTransformerManager, compute_manager_decode_bias
from src.models.worker import WorkerLSTM

import tests.eval_core as core
import tests.eval_viz as viz


HEADER_RE = re.compile(
    r"\[Ep (?P<episode>\d+)\] Mgr LR: (?P<mgr_lr>[-+0-9.eE]+), "
    r"Wkr LR: (?P<wkr_lr>[-+0-9.eE]+), SuccessEMA: "
    r"(?P<success_ema>[-+0-9.eE]+)%, Loss: (?P<loss>[-+0-9.eE]+)"
)
DEBUG_HEADER_RE = re.compile(r"\[DEBUG Ep (?P<episode>\d+) \| last (?P<window>\d+) eps\]")
PAREN_STD_RE = re.compile(r"^(?P<value>[-+0-9.eE]+)\s+\(std (?P<std>[-+0-9.eE]+)\)$")
TRIPLE_RE = re.compile(
    r"^mean=(?P<mean>[-+0-9.eE]+%?),\s*min=(?P<min>[-+0-9.eE]+%?),\s*max=(?P<max>[-+0-9.eE]+%?)$"
)
CORRIDOR_RE = re.compile(
    r"^mean=(?P<mean>[-+0-9.eE]+)%,\s*succ=(?P<succ>[-+0-9.eE]+)%,\s*fail=(?P<fail>[-+0-9.eE]+)%$"
)
DENSITY_RE = re.compile(
    r"^mean=(?P<mean>[-+0-9.eE]+),\s*band=(?P<band_lo>[-+0-9.eE]+)~(?P<band_hi>[-+0-9.eE]+),\s*gap=(?P<gap>[-+0-9.eE]+)$"
)
TARGET_LEN_RE = re.compile(
    r"^ref=(?P<ref>[-+0-9.eE]+),\s*band=(?P<band_lo>[-+0-9.eE]+)~(?P<band_hi>[-+0-9.eE]+)$"
)
RATE_WITH_COUNT_RE = re.compile(r"^(?P<rate>[-+0-9.eE]+)%\s+\((?P<count>\d+)\/(?P<total>\d+)\)$")
FRACTION_RATE_RE = re.compile(r"^(?P<count>\d+)\/(?P<total>\d+)\s+\((?P<rate>[-+0-9.eE]+)%\)$")
PAIR_RE = re.compile(r"^(?P<a>[-+0-9.eE]+)\s*/\s*(?P<b>[-+0-9.eE]+)$")
ERR_EXACT_RE = re.compile(r"^(?P<err>[-+0-9.eE]+)\s*/\s*(?P<exact>[-+0-9.eE]+)%$")
SG_HOPS_RE = re.compile(r"^first=(?P<first>[-+0-9.eE]+),\s*inter=(?P<inter>[-+0-9.eE]+),\s*max=(?P<max>[-+0-9.eE]+)$")
REWARD_STATS_RE = re.compile(r"^std=(?P<std>[-+0-9.eE]+),\s*min=(?P<min>[-+0-9.eE]+),\s*max=(?P<max>[-+0-9.eE]+)$")
CKPT_RE = re.compile(r"^(?P<reward>[-+0-9.eE]+)\s+\(quality (?P<quality>[-+0-9.eE]+), std (?P<std>[-+0-9.eE]+)\)$")
GRAD_RE = re.compile(
    r"^pre=(?P<pre>[-+0-9.eE]+),\s*post=(?P<post>[-+0-9.eE]+),\s*clip-hit=(?P<clip>[-+0-9.eE]+)%$"
)
HIT_AT_RE = re.compile(
    r"^(?P<h1>[-+0-9.eE]+)\/(?P<h2>[-+0-9.eE]+)\/(?P<h3>[-+0-9.eE]+)\/(?P<h4>[-+0-9.eE]+)\/(?P<h5>[-+0-9.eE]+)$"
)

DEFAULT_REWARD_CFG = {
    "PLAN_DENSITY_TARGET": 0.20,
    "PLAN_DENSITY_WEIGHT": 7.0,
    "PLAN_CORRIDOR_TARGET": 0.6,
    "PLAN_CORRIDOR_WEIGHT": 1.5,
    "TARGET_SEGMENT_HOPS": 4.5,
    "PLAN_COUNT_BAND": 1,
    "PLAN_COUNT_UNDER_PENALTY": 1.0,
    "PLAN_COUNT_OVER_PENALTY": 0.2,
    "ANCHOR_HOP_PENALTY": 0.15,
    "ANCHOR_NEAR_BONUS": 0.50,
    "FIRST_ANCHOR_PENALTY": 0.30,
    "SPACING_PENALTY_SCALE": 0.80,
    "MONOTONIC_PENALTY_SCALE": 0.50,
    "EMPTY_PLAN_PENALTY": -2.0,
    "PLAN_ADJUST_MIN": -6.0,
    "PLAN_ADJUST_MAX": 4.0,
    "CHECKPOINT_MAX_REWARD": 4.0,
    "VISIT_LOGIT_PENALTY": 0.35,
    "RECENT_NODE_PENALTY": 1.0,
    "RECENT_WINDOW": 4,
    "LOOP_LIMIT": 6,
}

SIMPLE_FIELDS = {
    "R1 PBRS": "r1_pbrs",
    "R2 Subgoal": "r2_subgoal",
    "R3 Goal": "r3_goal",
    "R4 Efficiency": "r4_efficiency",
    "R5 Milestone": "r5_milestone",
    "R6 Explore": "r6_explore",
    "R7 PlanPen.": "r7_plan_penalty",
    "P1 Time": "p1_time_penalty",
    "P2 Loop": "p2_loop_penalty",
    "P3 Fail": "p3_fail_penalty",
    "Total(Base)": "total_base",
    "Total(Final)": "total_final",
    "Goal Hit Rate": "goal_hit_rate",
    "Success EMA": "success_ema",
    "Fail Shaping": "fail_shaping",
    "Goal Share": "goal_share",
    "Loop Fail": "loop_fail_rate",
    "Stagnation Fail": "stagnation_fail_rate",
    "EOS Rate": "eos_rate",
    "Empty Plans": "empty_plan_rate",
    "Unique Ratio": "unique_ratio",
    "Far ShortPlan": "far_plan_rate",
    "Plan Utiliz.": "plan_utilization",
    "Wkr Entropy": "worker_entropy",
    "Critic V0": "critic_v0",
    "Critic MSE": "critic_mse",
    "TD |err|": "td_abs_err",
    "Expl. Var": "explained_variance",
    "Adv Mean": "adv_mean",
    "Adv Std": "adv_std",
    "Mgr Entropy": "manager_entropy",
    "Mgr NLL": "manager_nll",
    "CorridorStep": "manager_corridor_step_rate",
    "NonProgress": "manager_nonprogress_step_rate",
    "Avg Detour": "manager_avg_detour_hops",
    "EOS<=2 Rate": "manager_eos_near_goal_rate",
    "Policy Loss": "policy_loss",
    "Critic Loss": "critic_loss",
    "Ent Bonus": "entropy_bonus",
}
PAREN_STD_FIELDS = {
    "Return Mean": ("return_mean", "return_std"),
    "Norm Return": ("norm_return_mean", "norm_return_std"),
    "Value Mean": ("value_mean", "value_std"),
}
TRIPLE_FIELDS = {
    "Goal Dist": ("goal_dist_mean", "goal_dist_min", "goal_dist_max"),
    "Progress": ("progress_mean", "progress_min", "progress_max"),
    "Plan Length": ("plan_length_mean", "plan_length_min", "plan_length_max"),
    "Active SG Hop": ("active_sg_hop_mean", "active_sg_hop_min", "active_sg_hop_max"),
    "Max SG Prog.": ("max_sg_progress_mean", "max_sg_progress_min", "max_sg_progress_max"),
    "Wkr Steps": ("worker_steps_mean", "worker_steps_min", "worker_steps_max"),
}


def _clean_line(line: str) -> str:
    return re.sub(r"^[\s│├└┌┐┬┴┼─]+", "", line).strip()


def _to_float(text: str) -> float:
    return float(text.replace("%", "").replace(",", "").strip())


def _metric_value(row: pd.Series | dict[str, Any], key: str, default: float = math.nan) -> float:
    value = row.get(key, default)
    if value is None:
        return default
    return default if pd.isna(value) else float(value)


# eval_viz 모듈로 이동됨
_rolling = viz._rolling


def format_percent(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.1f}%"


def format_float(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.3f}"


def load_runtime_config(run_dir: Path) -> dict[str, Any]:
    runtime_path = run_dir / "runtime_config.json"
    if not runtime_path.exists():
        return {}
    try:
        return json.loads(runtime_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def is_apte_phase1_run(run_dir: Path, runtime_config: dict[str, Any] | None = None) -> bool:
    cfg = runtime_config if runtime_config is not None else load_runtime_config(run_dir)
    reward_cfg = cfg.get("reward_cfg", {}) if isinstance(cfg, dict) else {}
    save_dir = str(cfg.get("save_dir", "")) if isinstance(cfg, dict) else ""
    return (
        cfg.get("stage") == "phase1"
        and (
            reward_cfg.get("PHASE1_MODE") == "apte_guided_worker"
            or save_dir.endswith("logs/rl_phase1_apte")
        )
    )


def load_metrics_frame(run_dir: Path, runtime_config: dict[str, Any]) -> pd.DataFrame:
    if is_apte_phase1_run(run_dir, runtime_config):
        csv_path = run_dir / "debug_metrics.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"APTE debug metrics not found: {csv_path}")
        return pd.read_csv(csv_path)
    log_path = run_dir / "rl_debug_log.txt"
    if not log_path.exists():
        raise FileNotFoundError(f"Debug log not found: {log_path}")
    return parse_debug_log(log_path)


def build_apte_diagnosis(latest: pd.Series) -> list[str]:
    notes: list[str] = []
    if _metric_value(latest, "success_ema", 0.0) < 10.0:
        notes.append("Goal-reaching remains difficult, so the worker policy still needs stronger end-to-end execution.")
    if _metric_value(latest, "ordered_checkpoint_completion_ratio", 0.0) < 40.0:
        notes.append("Hidden training checkpoints are not being completed reliably, so exploration is still weak early in trajectories.")
    if _metric_value(latest, "stagnation_fail_rate", 0.0) >= 30.0:
        notes.append("Many rollouts still terminate for stagnation, which suggests the flat goal-conditioned policy is wandering or hesitating.")
    if _metric_value(latest, "critic_explained_variance", -1.0) < 0.05:
        notes.append("The critic is still weak, so value guidance is adding limited stability.")
    return notes


def build_apte_summary(df: pd.DataFrame) -> dict[str, Any]:
    latest = df.iloc[-1]
    best_success = df.loc[df["success_rate"].idxmax()] if "success_rate" in df.columns else latest
    best_ema = df.loc[df["success_ema"].idxmax()] if "success_ema" in df.columns else latest
    return {
        "num_records": int(len(df)),
        "episode_start": int(df["episode"].min()),
        "episode_end": int(df["episode"].max()),
        "latest": {
            "episode": int(latest["episode"]),
            "success_rate": _metric_value(latest, "success_rate"),
            "success_ema": _metric_value(latest, "success_ema"),
            "goal_dist_mean": _metric_value(latest, "goal_dist_mean"),
            "goal_hop_mean": _metric_value(latest, "goal_hop_mean"),
            "best_goal_dist_mean": _metric_value(latest, "best_goal_dist_mean"),
            "best_goal_hops_mean": _metric_value(latest, "best_goal_hops_mean"),
            "progress_mean": _metric_value(latest, "progress_mean"),
            "goal_progress_per100": _metric_value(latest, "goal_progress_per100"),
            "stagnation_fail_rate": _metric_value(latest, "stagnation_fail_rate"),
            "loop_fail_rate": _metric_value(latest, "loop_fail_rate"),
            "path_length_ratio": _metric_value(latest, "path_length_ratio"),
            "teacher_path_hops_mean": _metric_value(latest, "teacher_path_hops_mean"),
            "teacher_hidden_checkpoint_count_actual": _metric_value(latest, "teacher_hidden_checkpoint_count_actual"),
            "worker_aux_ce_loss": _metric_value(latest, "worker_aux_ce_loss"),
            "worker_entropy": _metric_value(latest, "worker_entropy"),
            "critic_explained_variance": _metric_value(latest, "critic_explained_variance"),
            "hidden_checkpoint_count_mean": _metric_value(latest, "hidden_checkpoint_count_mean"),
            "hidden_checkpoint_hit_rate": _metric_value(latest, "hidden_checkpoint_hit_rate"),
            "ordered_checkpoint_completion_ratio": _metric_value(latest, "ordered_checkpoint_completion_ratio"),
            "hidden_checkpoint_hit_at_rank_1": _metric_value(latest, "hidden_checkpoint_hit_at_rank_1"),
            "hidden_checkpoint_hit_at_rank_2": _metric_value(latest, "hidden_checkpoint_hit_at_rank_2"),
            "hidden_checkpoint_hit_at_rank_3plus": _metric_value(latest, "hidden_checkpoint_hit_at_rank_3plus"),
            "goal_neighbor_rate": _metric_value(latest, "goal_neighbor_rate"),
            "goal_neighbor_success_rate": _metric_value(latest, "goal_neighbor_success_rate"),
            "goal_hop_1_hit_rate": _metric_value(latest, "goal_hop_1_hit_rate"),
            "goal_hop_1_to_success_rate": _metric_value(latest, "goal_hop_1_to_success_rate"),
            "goal_regression_after_best4_rate": _metric_value(latest, "goal_regression_after_best4_rate"),
            "near_goal_ce_mult_applied_rate": _metric_value(latest, "near_goal_ce_mult_applied_rate"),
            "goal_threshold_hit_8_rate": _metric_value(latest, "goal_threshold_hit_8_rate"),
            "goal_threshold_hit_4_rate": _metric_value(latest, "goal_threshold_hit_4_rate"),
            "goal_threshold_hit_2_rate": _metric_value(latest, "goal_threshold_hit_2_rate"),
            "stagnation_reset_on_best_goal_improve_count": _metric_value(latest, "stagnation_reset_on_best_goal_improve_count"),
            "stagnation_reset_near_goal_count": _metric_value(latest, "stagnation_reset_near_goal_count"),
            "stagnation_reset_goal4_count": _metric_value(latest, "stagnation_reset_goal4_count"),
            "stagnation_reset_goal2_count": _metric_value(latest, "stagnation_reset_goal2_count"),
        },
        "best_success_rate": {
            "episode": int(best_success["episode"]),
            "success_rate": _metric_value(best_success, "success_rate"),
            "success_ema": _metric_value(best_success, "success_ema"),
        },
        "best_success_ema": {
            "episode": int(best_ema["episode"]),
            "success_rate": _metric_value(best_ema, "success_rate"),
            "success_ema": _metric_value(best_ema, "success_ema"),
        },
        "available_checkpoint_episodes": [],
        "diagnosis": build_apte_diagnosis(latest),
    }


def save_apte_summary_markdown(summary: dict[str, Any], path: Path) -> None:
    latest = summary["latest"]
    best_success = summary["best_success_rate"]
    best_ema = summary["best_success_ema"]
    lines = [
        "# APTE Phase1 Evaluation",
        "",
        f"- Parsed windows: `{summary['num_records']}`",
        f"- Episode range: `{summary['episode_start']} -> {summary['episode_end']}`",
        f"- Latest Success / EMA: `{format_percent(latest['success_rate'])}` / `{format_percent(latest['success_ema'])}`",
        f"- Goal Dist / Goal Hops / Best Dist / Best Hops: `{format_float(latest['goal_dist_mean'])}` / `{format_float(latest['goal_hop_mean'])}` / `{format_float(latest['best_goal_dist_mean'])}` / `{format_float(latest['best_goal_hops_mean'])}`",
        f"- Goal Progress per100 / Teacher Hops / Teacher Hidden Count: `{format_float(latest['goal_progress_per100'])}` / `{format_float(latest['teacher_path_hops_mean'])}` / `{format_float(latest['teacher_hidden_checkpoint_count_actual'])}`",
        f"- Stagnation / Loop Fail: `{format_percent(latest['stagnation_fail_rate'])}` / `{format_percent(latest['loop_fail_rate'])}`",
        f"- Path Length Ratio: `{format_float(latest['path_length_ratio'])}`",
        f"- Aux CE / Entropy / ExplVar: `{format_float(latest['worker_aux_ce_loss'])}` / `{format_float(latest['worker_entropy'])}` / `{format_float(latest['critic_explained_variance'])}`",
        f"- Hidden Ckpt Count / Hit / Completion: `{format_float(latest['hidden_checkpoint_count_mean'])}` / `{format_percent(latest['hidden_checkpoint_hit_rate'])}` / `{format_percent(latest['ordered_checkpoint_completion_ratio'])}`",
        f"- Hidden Hit@1 / Hit@2 / Hit@3+: `{format_percent(latest['hidden_checkpoint_hit_at_rank_1'])}` / `{format_percent(latest['hidden_checkpoint_hit_at_rank_2'])}` / `{format_percent(latest['hidden_checkpoint_hit_at_rank_3plus'])}`",
        f"- GoalNbr / GoalNbrSucc / Hop1 / Hop1->Succ: `{format_percent(latest['goal_neighbor_rate'])}` / `{format_percent(latest['goal_neighbor_success_rate'])}` / `{format_percent(latest['goal_hop_1_hit_rate'])}` / `{format_percent(latest['goal_hop_1_to_success_rate'])}`",
        f"- GoalRegress / NearGoalCE: `{format_percent(latest['goal_regression_after_best4_rate'])}` / `{format_percent(latest['near_goal_ce_mult_applied_rate'])}`",
        f"- Goal<=8/4/2: `{format_percent(latest['goal_threshold_hit_8_rate'])}` / `{format_percent(latest['goal_threshold_hit_4_rate'])}` / `{format_percent(latest['goal_threshold_hit_2_rate'])}`",
        f"- Stagnation resets best/near/4/2: `{format_float(latest['stagnation_reset_on_best_goal_improve_count'])}` / `{format_float(latest['stagnation_reset_near_goal_count'])}` / `{format_float(latest['stagnation_reset_goal4_count'])}` / `{format_float(latest['stagnation_reset_goal2_count'])}`",
        "",
        f"- Best Success Rate window: `Ep {best_success['episode']}` -> `{format_percent(best_success['success_rate'])}`",
        f"- Best Success EMA window: `Ep {best_ema['episode']}` -> `{format_percent(best_ema['success_ema'])}`",
        "",
        "## Diagnosis",
        "",
    ]
    for note in summary.get("diagnosis", []):
        lines.append(f"- {note}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# eval_viz 모듈로 이동됨
plot_apte_training_dashboard = viz.plot_apte_training_dashboard


def print_apte_terminal_summary(summary: dict[str, Any]) -> None:
    latest = summary["latest"]
    print("APTE phase1 summary")
    print(f"  Episodes: {summary['episode_start']} -> {summary['episode_end']} ({summary['num_records']} windows)")
    print(f"  Success / EMA: {format_percent(latest['success_rate'])} / {format_percent(latest['success_ema'])}")
    print(
        "  Goal Dist / Goal Hops / Best Dist / Best Hops: "
        f"{format_float(latest['goal_dist_mean'])} / {format_float(latest['goal_hop_mean'])} / "
        f"{format_float(latest['best_goal_dist_mean'])} / {format_float(latest['best_goal_hops_mean'])}"
    )
    print(
        "  Goal per100 / Teacher Hops / Teacher Hidden Count: "
        f"{format_float(latest['goal_progress_per100'])} / {format_float(latest['teacher_path_hops_mean'])} / "
        f"{format_float(latest['teacher_hidden_checkpoint_count_actual'])}"
    )
    print(
        "  Hidden Count / Hit / Completion: "
        f"{format_float(latest['hidden_checkpoint_count_mean'])} / "
        f"{format_percent(latest['hidden_checkpoint_hit_rate'])} / "
        f"{format_percent(latest['ordered_checkpoint_completion_ratio'])}"
    )
    print(
        "  Hidden Hit@1 / Hit@2 / Hit@3+: "
        f"{format_percent(latest['hidden_checkpoint_hit_at_rank_1'])} / "
        f"{format_percent(latest['hidden_checkpoint_hit_at_rank_2'])} / "
        f"{format_percent(latest['hidden_checkpoint_hit_at_rank_3plus'])}"
    )
    print(
        "  GoalNbr / GoalNbrSucc / Hop1 / Hop1->Succ: "
        f"{format_percent(latest['goal_neighbor_rate'])} / "
        f"{format_percent(latest['goal_neighbor_success_rate'])} / "
        f"{format_percent(latest['goal_hop_1_hit_rate'])} / "
        f"{format_percent(latest['goal_hop_1_to_success_rate'])}"
    )
    print(
        "  GoalRegress / NearGoalCE: "
        f"{format_percent(latest['goal_regression_after_best4_rate'])} / "
        f"{format_percent(latest['near_goal_ce_mult_applied_rate'])}"
    )
    print(
        "  Goal<=8/4/2: "
        f"{format_percent(latest['goal_threshold_hit_8_rate'])} / "
        f"{format_percent(latest['goal_threshold_hit_4_rate'])} / "
        f"{format_percent(latest['goal_threshold_hit_2_rate'])}"
    )
    print(
        "  Stagnation resets best/near/4/2: "
        f"{format_float(latest['stagnation_reset_on_best_goal_improve_count'])} / "
        f"{format_float(latest['stagnation_reset_near_goal_count'])} / "
        f"{format_float(latest['stagnation_reset_goal4_count'])} / "
        f"{format_float(latest['stagnation_reset_goal2_count'])}"
    )


def parse_debug_log(log_path: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for raw_line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        header = HEADER_RE.search(raw_line)
        if header:
            if current is not None:
                records.append(current)
            current = {
                "episode": int(header.group("episode")),
                "mgr_lr": float(header.group("mgr_lr")),
                "wkr_lr": float(header.group("wkr_lr")),
                "success_ema": float(header.group("success_ema")),
                "loss": float(header.group("loss")),
            }
            continue

        debug_header = DEBUG_HEADER_RE.search(raw_line)
        if debug_header:
            debug_episode = int(debug_header.group("episode"))
            if current is None or current.get("episode") != debug_episode:
                if current is not None:
                    records.append(current)
                current = {"episode": debug_episode}
            current["debug_window"] = int(debug_header.group("window"))
            continue

        if current is None:
            continue

        line = _clean_line(raw_line)
        if not line or line.endswith("───") or ":" not in line or "Diagnostics" in line:
            continue
        if line in {"Reward Alignment", "Batch Outcome", "Critic & Advantage", "Gradient Norms"}:
            continue

        label, value = [part.strip() for part in line.split(":", 1)]

        if label in SIMPLE_FIELDS:
            try:
                current[SIMPLE_FIELDS[label]] = _to_float(value)
            except ValueError:
                pass
            continue

        if label in PAREN_STD_FIELDS:
            match = PAREN_STD_RE.match(value)
            if match:
                value_key, std_key = PAREN_STD_FIELDS[label]
                current[value_key] = _to_float(match.group("value"))
                current[std_key] = _to_float(match.group("std"))
            continue

        if label in TRIPLE_FIELDS:
            match = TRIPLE_RE.match(value)
            if match:
                mean_key, min_key, max_key = TRIPLE_FIELDS[label]
                current[mean_key] = _to_float(match.group("mean"))
                current[min_key] = _to_float(match.group("min"))
                current[max_key] = _to_float(match.group("max"))
            continue

        if label == "Ckpt Reward":
            match = CKPT_RE.match(value)
            if match:
                current["checkpoint_reward"] = _to_float(match.group("reward"))
                current["checkpoint_quality"] = _to_float(match.group("quality"))
                current["checkpoint_std"] = _to_float(match.group("std"))
        elif label == "Reward Stats":
            match = REWARD_STATS_RE.match(value)
            if match:
                current["reward_std"] = _to_float(match.group("std"))
                current["reward_min"] = _to_float(match.group("min"))
                current["reward_max"] = _to_float(match.group("max"))
        elif label in {"Success/Fail", "Norm Ret S/F", "Goal/SG per100"}:
            match = PAIR_RE.match(value)
            if match:
                pair_keys = {
                    "Success/Fail": ("success_return", "fail_return"),
                    "Norm Ret S/F": ("success_norm_return", "fail_norm_return"),
                    "Goal/SG per100": ("goal_per100", "subgoal_per100"),
                }[label]
                current[pair_keys[0]] = _to_float(match.group("a"))
                current[pair_keys[1]] = _to_float(match.group("b"))
        elif label == "Success Rate":
            match = RATE_WITH_COUNT_RE.match(value)
            if match:
                current["success_rate"] = _to_float(match.group("rate"))
                current["success_count"] = int(match.group("count"))
                current["batch_size"] = int(match.group("total"))
        elif label == "Density":
            match = DENSITY_RE.match(value)
            if match:
                current["plan_density_mean"] = _to_float(match.group("mean"))
                current["plan_density_gap"] = _to_float(match.group("gap"))
                current["density_band_low"] = _to_float(match.group("band_lo"))
                current["density_band_high"] = _to_float(match.group("band_hi"))
        elif label == "Target Len":
            match = TARGET_LEN_RE.match(value)
            if match:
                current["plan_len_ref_mean"] = _to_float(match.group("ref"))
                current["plan_len_min_target_mean"] = _to_float(match.group("band_lo"))
                current["plan_len_max_target_mean"] = _to_float(match.group("band_hi"))
        elif label == "Plan Under/Over":
            match = PAIR_RE.match(value.replace("%", ""))
            if match:
                current["plan_under_rate"] = _to_float(match.group("a"))
                current["plan_over_rate"] = _to_float(match.group("b"))
        elif label == "Anchor Err/Near":
            match = ERR_EXACT_RE.match(value)
            if match:
                current["anchor_hop_err_mean"] = _to_float(match.group("err"))
                current["anchor_near_rate"] = _to_float(match.group("exact"))
        elif label == "Corridor":
            match = CORRIDOR_RE.match(value)
            if match:
                current["corridor_mean"] = _to_float(match.group("mean"))
                current["corridor_success"] = _to_float(match.group("succ"))
                current["corridor_fail"] = _to_float(match.group("fail"))
        elif label == "SG Hops":
            match = SG_HOPS_RE.match(value)
            if match:
                current["sg_hops_first"] = _to_float(match.group("first"))
                current["sg_hops_inter"] = _to_float(match.group("inter"))
                current["sg_hops_max"] = _to_float(match.group("max"))
        elif label == "Subgoal Reach":
            match = FRACTION_RATE_RE.match(value)
            if match:
                current["subgoal_reach_count"] = int(match.group("count"))
                current["subgoal_total"] = int(match.group("total"))
                current["subgoal_reach_rate"] = _to_float(match.group("rate"))
        elif label == "Hit@1/2/3/4/5+":
            match = HIT_AT_RE.match(value)
            if match:
                current["hit_at_1"] = _to_float(match.group("h1"))
                current["hit_at_2"] = _to_float(match.group("h2"))
                current["hit_at_3"] = _to_float(match.group("h3"))
                current["hit_at_4"] = _to_float(match.group("h4"))
                current["hit_at_5p"] = _to_float(match.group("h5"))
        elif label in {"Manager", "Worker"}:
            match = GRAD_RE.match(value)
            if match:
                prefix = "manager" if label == "Manager" else "worker"
                current[f"{prefix}_grad_pre"] = _to_float(match.group("pre"))
                current[f"{prefix}_grad_post"] = _to_float(match.group("post"))
                current[f"{prefix}_grad_clip_hit"] = _to_float(match.group("clip"))

    if current is not None:
        records.append(current)

    frame = (
        pd.DataFrame(records)
        .sort_values("episode")
        .drop_duplicates("episode", keep="last")
        .reset_index(drop=True)
    )
    if frame.empty:
        raise RuntimeError(f"No debug records could be parsed from {log_path}")
    return frame


def discover_checkpoint_episodes(run_dir: Path) -> list[int]:
    if is_apte_phase1_run(run_dir):
        return []
    manager_eps = {
        int(match.group(1))
        for path in run_dir.glob("manager_*.pt")
        if (match := re.search(r"manager_(\d+)\.pt$", path.name))
    }
    worker_eps = {
        int(match.group(1))
        for path in run_dir.glob("worker_*.pt")
        if (match := re.search(r"worker_(\d+)\.pt$", path.name))
    }
    return sorted(manager_eps & worker_eps)


def build_diagnosis(latest: pd.Series) -> list[str]:
    notes: list[str] = []
    if _metric_value(latest, "loop_fail_rate", 100.0) <= 5.0 and _metric_value(latest, "plan_utilization", 0.0) >= 35.0:
        notes.append("Worker execution is no longer the primary bottleneck; subgoal tracking is relatively strong.")
    else:
        notes.append("Worker execution is still contributing to failures; loop control or subgoal following remains weak.")
    if _metric_value(latest, "stagnation_fail_rate", 0.0) >= 5.0:
        notes.append("Some trajectories are being terminated for stagnation, which suggests wandering is still present even when loop failures stay low.")
    if _metric_value(latest, "plan_under_rate", 0.0) >= 40.0 or _metric_value(latest, "far_plan_rate", 0.0) >= 40.0:
        notes.append("Manager is still under-planning on many long problems, so short-plan collapse remains the biggest gap.")
    if _metric_value(latest, "corridor_mean", 0.0) < 45.0:
        notes.append("Plans are not consistently corridor-aligned toward the goal, so goal-directed checkpoint quality is still limited.")
    if _metric_value(latest, "anchor_hop_err_mean", 0.0) >= 2.0:
        notes.append("Subgoal placement is still far from the shortest-path anchors, so manager spacing quality remains inconsistent.")
    if _metric_value(latest, "manager_nonprogress_step_rate", 100.0) > 25.0:
        notes.append("Manager is still choosing too many non-progress subgoals, so checkpoint quality remains inconsistent even when worker execution is stable.")
    if _metric_value(latest, "manager_grad_clip_hit", 0.0) >= 95.0:
        notes.append("Manager gradients are saturated almost every update, which points to optimizer or scheduler bottlenecks late in training.")
    if _metric_value(latest, "explained_variance", -1.0) < 0.05:
        notes.append("The critic is usable but still weak; it is probably not the main blocker, but it is not adding much headroom either.")
    if _metric_value(latest, "goal_per100", -999.0) < 0.0:
        notes.append("Subgoals are being followed, but final goal progress is still inconsistent, so plan quality is not yet reliably goal-aligned.")
    return notes


def build_summary(df: pd.DataFrame, checkpoint_episodes: list[int]) -> dict[str, Any]:
    latest = df.iloc[-1]
    best_success = df.loc[df["success_rate"].idxmax()] if "success_rate" in df.columns else latest
    best_ema = df.loc[df["success_ema"].idxmax()] if "success_ema" in df.columns else latest
    return {
        "num_records": int(len(df)),
        "episode_start": int(df["episode"].min()),
        "episode_end": int(df["episode"].max()),
        "latest": {
            "episode": int(latest["episode"]),
            "success_rate": _metric_value(latest, "success_rate"),
            "success_ema": _metric_value(latest, "success_ema"),
            "plan_density_mean": _metric_value(latest, "plan_density_mean"),
            "far_plan_rate": _metric_value(latest, "far_plan_rate"),
            "plan_under_rate": _metric_value(latest, "plan_under_rate"),
            "plan_over_rate": _metric_value(latest, "plan_over_rate"),
            "plan_len_ref_mean": _metric_value(latest, "plan_len_ref_mean"),
            "anchor_hop_err_mean": _metric_value(latest, "anchor_hop_err_mean"),
            "anchor_near_rate": _metric_value(latest, "anchor_near_rate"),
            "corridor_mean": _metric_value(latest, "corridor_mean"),
            "manager_corridor_step_rate": _metric_value(latest, "manager_corridor_step_rate"),
            "manager_nonprogress_step_rate": _metric_value(latest, "manager_nonprogress_step_rate"),
            "manager_avg_detour_hops": _metric_value(latest, "manager_avg_detour_hops"),
            "manager_eos_near_goal_rate": _metric_value(latest, "manager_eos_near_goal_rate"),
            "plan_utilization": _metric_value(latest, "plan_utilization"),
            "loop_fail_rate": _metric_value(latest, "loop_fail_rate"),
            "stagnation_fail_rate": _metric_value(latest, "stagnation_fail_rate"),
            "goal_per100": _metric_value(latest, "goal_per100"),
            "subgoal_per100": _metric_value(latest, "subgoal_per100"),
            "explained_variance": _metric_value(latest, "explained_variance"),
            "manager_grad_clip_hit": _metric_value(latest, "manager_grad_clip_hit"),
            "worker_grad_clip_hit": _metric_value(latest, "worker_grad_clip_hit"),
        },
        "best_success_rate": {
            "episode": int(best_success["episode"]),
            "success_rate": _metric_value(best_success, "success_rate"),
            "success_ema": _metric_value(best_success, "success_ema"),
        },
        "best_success_ema": {
            "episode": int(best_ema["episode"]),
            "success_rate": _metric_value(best_ema, "success_rate"),
            "success_ema": _metric_value(best_ema, "success_ema"),
        },
        "available_checkpoint_episodes": checkpoint_episodes[-20:],
        "diagnosis": build_diagnosis(latest),
    }


def save_summary_markdown(summary: dict[str, Any], path: Path) -> None:
    latest = summary["latest"]
    best_success = summary["best_success_rate"]
    best_ema = summary["best_success_ema"]
    lines = [
        "# RL Intermediate Evaluation",
        "",
        f"- Parsed windows: `{summary['num_records']}`",
        f"- Episode range: `{summary['episode_start']} -> {summary['episode_end']}`",
        f"- Latest episode: `{latest['episode']}`",
        "",
        "## Latest Snapshot",
        "",
        f"- Success Rate: `{format_percent(latest['success_rate'])}`",
        f"- Success EMA: `{format_percent(latest['success_ema'])}`",
        f"- Plan Density: `{format_float(latest['plan_density_mean'])}`",
        f"- Far ShortPlan: `{format_percent(latest['far_plan_rate'])}`",
        f"- Plan Under/Over: `{format_percent(latest['plan_under_rate'])}` / `{format_percent(latest['plan_over_rate'])}`",
        f"- Target Plan Len: `{format_float(latest['plan_len_ref_mean'])}`",
        f"- Anchor Hop Err/Near: `{format_float(latest['anchor_hop_err_mean'])}` / `{format_percent(latest['anchor_near_rate'])}`",
        f"- Corridor: `{format_percent(latest['corridor_mean'])}`",
        f"- Manager CorridorStep: `{format_percent(latest['manager_corridor_step_rate'])}`",
        f"- Manager NonProgress: `{format_percent(latest['manager_nonprogress_step_rate'])}`",
        f"- Manager Avg Detour: `{format_float(latest['manager_avg_detour_hops'])}`",
        f"- Manager EOS<=2: `{format_percent(latest['manager_eos_near_goal_rate'])}`",
        f"- Plan Utilization: `{format_percent(latest['plan_utilization'])}`",
        f"- Loop Fail: `{format_percent(latest['loop_fail_rate'])}`",
        f"- Stagnation Fail: `{format_percent(latest['stagnation_fail_rate'])}`",
        f"- Goal/SG per100: `{format_float(latest['goal_per100'])}` / `{format_float(latest['subgoal_per100'])}`",
        f"- Explained Variance: `{format_float(latest['explained_variance'])}`",
        "",
        "## Best Windows",
        "",
        f"- Best Success Rate: `Ep {best_success['episode']}` with `{format_percent(best_success['success_rate'])}`",
        f"- Best Success EMA: `Ep {best_ema['episode']}` with `{format_percent(best_ema['success_ema'])}`",
        "",
        "## Diagnosis",
        "",
    ]
    lines.extend(f"- {note}" for note in summary["diagnosis"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_terminal_summary(summary: dict[str, Any]) -> None:
    latest = summary["latest"]
    best_success = summary["best_success_rate"]
    best_ema = summary["best_success_ema"]
    print("\n=== RL Intermediate Evaluation ===")
    print(f"Range: Ep {summary['episode_start']} -> {summary['episode_end']} ({summary['num_records']} debug windows)")
    print(
        "Latest: "
        f"SR={format_percent(latest['success_rate'])}, "
        f"EMA={format_percent(latest['success_ema'])}, "
        f"Density={format_float(latest['plan_density_mean'])}, "
        f"FarShort={format_percent(latest['far_plan_rate'])}, "
        f"AnchorErr={format_float(latest['anchor_hop_err_mean'])}, "
        f"Corridor={format_percent(latest['corridor_mean'])}, "
        f"PlanUtil={format_percent(latest['plan_utilization'])}"
    )
    print(
        "Best: "
        f"SR Ep {best_success['episode']} ({format_percent(best_success['success_rate'])}), "
        f"EMA Ep {best_ema['episode']} ({format_percent(best_ema['success_ema'])})"
    )
    print("Diagnosis:")
    for note in summary["diagnosis"]:
        print(f"- {note}")


# eval_viz 모듈로 이동됨
_plot_line = viz._plot_line
_plot_single_chart = viz._plot_single_chart
plot_training_dashboard = viz.plot_training_dashboard
plot_relationships = viz.plot_relationships


def plot_snapshot_comparison(df: pd.DataFrame, output_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    latest = df.iloc[-1]
    best_success = df.loc[df["success_rate"].idxmax()] if "success_rate" in df.columns else latest
    best_ema = df.loc[df["success_ema"].idxmax()] if "success_ema" in df.columns else latest
    snapshots = {
        f"Latest\nEp {int(latest['episode'])}": latest,
        f"Best SR\nEp {int(best_success['episode'])}": best_success,
        f"Best EMA\nEp {int(best_ema['episode'])}": best_ema,
    }
    metrics = [
        ("success_rate", False),
        ("success_ema", False),
        ("corridor_mean", False),
        ("plan_utilization", False),
        ("plan_density_mean", True),
        ("loop_fail_rate", True),
        ("stagnation_fail_rate", True),
    ]
    labels = [key.replace("_", " ").title() for key, _ in metrics]
    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    fig.suptitle("Snapshot Comparison", fontsize=16, fontweight="bold")
    for idx, (name, row) in enumerate(snapshots.items()):
        values = []
        for key, invert in metrics:
            value = _metric_value(row, key, math.nan)
            values.append(np.nan if math.isnan(value) else (-value if invert else value))
        ax.bar(x + (idx - 1) * width, values, width=width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Higher is better (inverted for Density and Loop Fail)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def maybe_load_episode_samples(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
        if isinstance(payload, list):
            return payload
    except json.JSONDecodeError:
        pass

    samples = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            samples.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return samples


def infer_disaster(run_dir: Path) -> bool:
    return "phase2" in run_dir.name.lower()


# eval_core로 통합됨
set_seed = core.set_seed


def nearest_checkpoint(episodes: list[int], target: int) -> int:
    if not episodes:
        raise RuntimeError("No paired manager/worker checkpoints were found in the run directory.")
    return min(episodes, key=lambda ep: (abs(ep - target), ep))


def resolve_checkpoint_spec(
    run_dir: Path,
    checkpoint_spec: str,
    summary: dict[str, Any],
    checkpoint_episodes: list[int],
) -> dict[str, Any]:
    if is_apte_phase1_run(run_dir):
        spec = checkpoint_spec.strip().lower()
        if spec == "sl":
            sl_path = Path("logs") / "sl_pretrain" / "model_sl_final.pt"
            if not sl_path.exists():
                raise FileNotFoundError(f"SL checkpoint not found: {sl_path}")
            return {
                "label": "sl",
                "display": "SL warm start",
                "source": "sl",
                "episode": None,
                "sl_path": sl_path,
            }

        if spec in {"latest", "final"}:
            worker_path = run_dir / "final.pt"
            display = "APTE final checkpoint"
            label = "final"
        elif spec in {"best_sr", "best_ema", "best"}:
            worker_path = run_dir / "final.pt"
            display = "APTE final checkpoint (best alias)"
            label = "final"
        else:
            raise ValueError(
                "APTE phase1 checkpoints support only final/latest, best aliases mapped to final, or sl."
            )
        if not worker_path.exists():
            raise FileNotFoundError(f"APTE checkpoint not found: {worker_path}")
        return {
            "label": label,
            "display": display,
            "source": "apte_phase1",
            "episode": summary["latest"]["episode"],
            "worker_path": worker_path,
        }

    spec = checkpoint_spec.strip().lower()

    # Worker-only 평가: logs/rl_worker_stage/best.pt에서 Worker만 로드
    if spec in {"worker_best", "worker"}:
        worker_path = Path("logs") / "rl_worker_stage" / "best.pt"
        if not worker_path.exists():
            raise FileNotFoundError(f"Worker best checkpoint not found: {worker_path}")
        return {
            "label": "worker_best",
            "display": "Worker RL best (goal-conditioned)",
            "source": "worker_only",
            "episode": None,
            "worker_path": worker_path,
        }

    if spec == "sl":
        sl_path = Path("logs") / "sl_pretrain" / "model_sl_final.pt"
        if not sl_path.exists():
            raise FileNotFoundError(f"SL checkpoint not found: {sl_path}")
        return {
            "label": "sl",
            "display": "SL warm start",
            "source": "sl",
            "episode": None,
            "sl_path": sl_path,
        }

    # Joint best.pt 지원: Manager+Worker가 하나의 파일에 저장된 Joint RL 체크포인트
    if spec in {"joint_best", "joint"}:
        joint_path = Path("logs") / "rl_joint_stage" / "best.pt"
        if not joint_path.exists():
            raise FileNotFoundError(f"Joint best checkpoint not found: {joint_path}")
        return {
            "label": "joint_best",
            "display": "Joint RL best checkpoint",
            "source": "joint",
            "episode": None,
            "joint_path": joint_path,
        }

    if spec == "best_sr":
        requested = int(summary["best_success_rate"]["episode"])
    elif spec == "best_ema":
        requested = int(summary["best_success_ema"]["episode"])
    elif spec == "latest":
        requested = int(summary["latest"]["episode"])
    else:
        try:
            requested = int(spec)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported checkpoint spec '{checkpoint_spec}'. Use latest, best_sr, best_ema, sl, or an episode number."
            ) from exc

    actual = nearest_checkpoint(checkpoint_episodes, requested)
    manager_path = run_dir / f"manager_{actual}.pt"
    worker_path = run_dir / f"worker_{actual}.pt"
    if not manager_path.exists() or not worker_path.exists():
        raise FileNotFoundError(f"Checkpoint pair for episode {actual} is incomplete in {run_dir}")
    return {
        "label": f"ep{actual}",
        "display": f"RL checkpoint Ep {actual}",
        "source": "rl",
        "episode": actual,
        "requested_episode": requested,
        "manager_path": manager_path,
        "worker_path": worker_path,
    }


# eval_core로 통합됨
select_edge_attr = core.select_edge_attr


def load_state_dict_flexible(path: Path, device: torch.device) -> dict[str, Any]:
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    return payload


# eval_core로 통합됨
extract_worker_state = core.extract_worker_state


def load_worker_state_compat(worker: WorkerLSTM, state_dict: dict[str, Any]) -> None:
    """eval_core의 호환 로드 래퍼 (verbose=True 기본)."""
    core.load_worker_state_compat(worker, state_dict, verbose=True)


def load_models_for_rollout(
    run_dir: Path,
    checkpoint_info: dict[str, Any],
    map_name: str,
    disaster: bool,
    hidden_dim: int,
    device: torch.device,
) -> tuple[DisasterEnv, GraphTransformerManager, WorkerLSTM]:
    env = DisasterEnv(
        f"data/{map_name}_node.tntp",
        f"data/{map_name}_net.tntp",
        device=str(device),
        verbose=False,
        enable_disaster=disaster,
    )

    worker = WorkerLSTM(node_dim=9, hidden_dim=hidden_dim, edge_dim=1).to(device)

    if checkpoint_info["source"] == "apte_phase1":
        manager = None
        payload = torch.load(checkpoint_info["worker_path"], map_location=device)
        load_worker_state_compat(worker, extract_worker_state(payload))
    elif checkpoint_info["source"] == "worker_only":
        # Worker-only: best.pt에서 worker_state만 추출 (Manager 없이 평가)
        manager = None
        payload = torch.load(checkpoint_info["worker_path"], map_location=device)
        if isinstance(payload, dict) and "worker_state" in payload:
            load_worker_state_compat(worker, payload["worker_state"])
        else:
            load_worker_state_compat(worker, extract_worker_state(payload))
    elif checkpoint_info["source"] == "sl":
        manager = GraphTransformerManager(node_dim=4, hidden_dim=hidden_dim, dropout=0.2, edge_dim=1).to(device)
        payload = torch.load(checkpoint_info["sl_path"], map_location=device)
        manager.load_state_dict(payload["manager_state"])
        load_worker_state_compat(worker, payload["worker_state"])
    elif checkpoint_info["source"] == "joint":
        # Joint best.pt: manager_state + worker_state가 하나의 파일에 저장
        manager = GraphTransformerManager(node_dim=4, hidden_dim=hidden_dim, dropout=0.2, edge_dim=1).to(device)
        payload = torch.load(checkpoint_info["joint_path"], map_location=device)
        if isinstance(payload, dict) and "manager_state" in payload:
            manager.load_state_dict(payload["manager_state"])
            load_worker_state_compat(worker, payload["worker_state"])
        else:
            # Fallback: 전체가 state_dict인 경우
            manager.load_state_dict(payload)
            print("⚠️ Joint checkpoint에서 worker_state를 찾을 수 없어 Manager만 로드했습니다.")
    else:
        manager = GraphTransformerManager(node_dim=4, hidden_dim=hidden_dim, dropout=0.2, edge_dim=1).to(device)
        manager.load_state_dict(load_state_dict_flexible(checkpoint_info["manager_path"], device))
        load_worker_state_compat(worker, load_state_dict_flexible(checkpoint_info["worker_path"], device))

    if manager is not None:
        manager.eval()
    worker.eval()
    return env, manager, worker


# eval_core로 통합됨 (이름 매핑)
ensure_env_node_feature_layout = core.ensure_env_layout
refresh_env_node_features = core.refresh_env_features


# eval_core로 통합됨
configure_single_problem = core.configure_single_problem


def truncate_plan(sequence: list[int], eos_index: int, num_nodes: int) -> list[int]:
    plan: list[int] = []
    for token in sequence:
        if token == eos_index:
            break
        if 0 <= token < num_nodes:
            plan.append(int(token))
    return plan


# eval_core로 통합됨
safe_softmax = core.safe_softmax


def categorical_entropy(probs: torch.Tensor) -> float:
    probs = torch.clamp(probs, min=1e-12)
    return float((-(probs * probs.log()).sum()).item())


def select_action(scores: torch.Tensor, temperature: float) -> tuple[int, torch.Tensor]:
    probs = safe_softmax(scores, temperature=temperature if temperature > 1e-5 else 1.0)
    finite = torch.isfinite(scores)
    scores_safe = scores.clone()
    scores_safe[~finite] = -1e9
    if temperature <= 1e-5:
        action = int(torch.argmax(scores_safe).item())
    else:
        action = int(torch.multinomial(probs, 1).item())
    return action, probs


def _extract_topk_entries(
    scores: torch.Tensor,
    probs: torch.Tensor,
    topk: int,
    eos_index: int | None = None,
    penalties: torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    finite = torch.isfinite(scores)
    if not finite.any():
        return []
    prob_scores = probs.clone()
    prob_scores[~finite] = -1.0
    k = min(topk, int(finite.sum().item()))
    values, indices = torch.topk(prob_scores, k=k)
    entries = []
    for value, index in zip(values.tolist(), indices.tolist()):
        entry = {
            "node_index": int(index),
            "node_label": "EOS" if eos_index is not None and index == eos_index else str(index),
            "probability": float(value),
            "score": float(scores[index].item()),
        }
        if penalties is not None:
            entry["penalty"] = float(penalties[index].item())
        entries.append(entry)
    return entries


def _compute_expected_stats(
    probs: torch.Tensor,
    positions: torch.Tensor,
    goal_dists: torch.Tensor,
    eos_index: int | None = None,
) -> dict[str, float]:
    node_probs = probs[:-1] if eos_index is not None else probs
    node_probs = node_probs / node_probs.sum().clamp(min=1e-12)
    expected_x = float((node_probs * positions[:, 0]).sum().item())
    expected_y = float((node_probs * positions[:, 1]).sum().item())
    expected_goal_distance = float((node_probs * goal_dists).sum().item())
    stats = {
        "expected_x": expected_x,
        "expected_y": expected_y,
        "expected_goal_distance": expected_goal_distance,
    }
    if eos_index is not None:
        stats["eos_probability"] = float(probs[eos_index].item())
    return stats


def _safe_hop_value(env: DisasterEnv, src: int, dst: int) -> float:
    value = float(env.hop_matrix[src, dst].item())
    return value if math.isfinite(value) else float("inf")


def _manager_candidate_metrics(
    env: DisasterEnv,
    start_idx: int,
    goal_idx: int,
    current_node: int,
    candidate_idx: int,
    eos_index: int,
    shortest_hops: float,
    bias_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    current_goal_hops = _safe_hop_value(env, current_node, goal_idx)
    bias_idx = eos_index if candidate_idx == eos_index else candidate_idx
    bias_total = 0.0
    corridor_bias = 0.0
    progress_bias = 0.0
    detour_bias = 0.0
    nonprogress_bias = 0.0
    eos_bias = 0.0
    if bias_payload is not None:
        bias_total = float(bias_payload["total_bias"][bias_idx].item())
        corridor_bias = float(bias_payload["corridor_bonus"][bias_idx].item())
        progress_bias = float(bias_payload["progress_bonus"][bias_idx].item())
        detour_bias = float(bias_payload["detour_penalty"][bias_idx].item())
        nonprogress_bias = float(bias_payload["nonprogress_penalty"][bias_idx].item())
        eos_bias = float(bias_payload["eos_bonus_or_penalty"][bias_idx].item())

    if candidate_idx == eos_index:
        eos_reasonable = current_goal_hops <= 2.0
        return {
            "candidate_type": "eos",
            "hops_from_current": 0.0,
            "hops_from_start": float(current_node != start_idx),
            "hops_to_goal": current_goal_hops,
            "goal_hops": current_goal_hops,
            "goal_progress_hops": 0.0,
            "progress_ratio": 0.0,
            "total_via_hops": current_goal_hops,
            "detour_hops": current_goal_hops - shortest_hops,
            "corridor_ok": eos_reasonable,
            "corridor_slack": 2.0 - current_goal_hops,
            "eos_reasonable": eos_reasonable,
            "bias_total": bias_total,
            "bias_corridor": corridor_bias,
            "bias_progress": progress_bias,
            "bias_detour": detour_bias,
            "bias_nonprogress": nonprogress_bias,
            "bias_eos": eos_bias,
        }

    hops_from_current = _safe_hop_value(env, current_node, candidate_idx)
    hops_from_start = _safe_hop_value(env, start_idx, candidate_idx)
    hops_to_goal = _safe_hop_value(env, candidate_idx, goal_idx)
    total_via_hops = hops_from_start + hops_to_goal if math.isfinite(hops_from_start + hops_to_goal) else float("inf")
    goal_progress_hops = current_goal_hops - hops_to_goal
    progress_ratio = goal_progress_hops / max(current_goal_hops, 1.0) if math.isfinite(current_goal_hops) else 0.0
    corridor_slack = (shortest_hops + 2.0) - total_via_hops
    return {
        "candidate_type": "node",
        "hops_from_current": hops_from_current,
        "hops_from_start": hops_from_start,
        "hops_to_goal": hops_to_goal,
        "goal_hops": hops_to_goal,
        "goal_progress_hops": goal_progress_hops,
        "progress_ratio": progress_ratio,
        "total_via_hops": total_via_hops,
        "detour_hops": total_via_hops - shortest_hops,
        "corridor_ok": corridor_slack >= 0.0,
        "corridor_slack": corridor_slack,
        "eos_reasonable": False,
        "bias_total": bias_total,
        "bias_corridor": corridor_bias,
        "bias_progress": progress_bias,
        "bias_detour": detour_bias,
        "bias_nonprogress": nonprogress_bias,
        "bias_eos": eos_bias,
    }


def _annotate_manager_topk(
    entries: list[dict[str, Any]],
    env: DisasterEnv,
    start_idx: int,
    goal_idx: int,
    current_node: int,
    eos_index: int,
    shortest_hops: float,
    bias_payload: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    annotated = []
    for entry in entries:
        metrics = _manager_candidate_metrics(
            env=env,
            start_idx=start_idx,
            goal_idx=goal_idx,
            current_node=current_node,
            candidate_idx=int(entry["node_index"]),
            eos_index=eos_index,
            shortest_hops=shortest_hops,
            bias_payload=bias_payload,
        )
        annotated.append({**entry, **metrics})
    return annotated


def diagnose_manager_steps(
    steps: list[dict[str, Any]],
    shortest_hops: float,
) -> dict[str, Any]:
    flagged_steps: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()

    for step in steps:
        selected_label = step.get("selected_label", str(step.get("selected_node", "?")))
        selected_prob = float(step.get("selected_probability", 0.0))
        prob_gap = float(step.get("topk_prob_gap", 1.0))
        current_goal_hops = float(step.get("current_goal_hops", float("inf")))
        score = 0
        reasons: list[str] = []

        if selected_label == "EOS":
            if step.get("step", 0) == 0:
                score += 5
                reasons.append("Immediate EOS was selected at the first manager step")
            elif current_goal_hops > 2.0:
                score += 3
                reasons.append(f"EOS was chosen while the goal was still {current_goal_hops:.1f} hops away")
            if selected_prob < 0.5:
                score += 1
                reasons.append("EOS selection confidence was low")
        else:
            selected_goal_progress = float(step.get("selected_goal_progress_hops", 0.0))
            selected_detour = float(step.get("selected_detour_hops", 0.0))
            if not bool(step.get("selected_corridor_ok", True)):
                score += 3
                reasons.append("Selected subgoal lies outside the start-goal corridor")
            if selected_goal_progress < 0.0:
                score += 3
                reasons.append("Selected subgoal increases remaining goal hops")
            elif selected_goal_progress < 1.0:
                score += 1
                reasons.append("Selected subgoal barely improves remaining goal hops")
            if selected_detour > 2.0:
                score += 1
                reasons.append(f"Selected subgoal adds a {selected_detour:.1f}-hop detour")
            if selected_prob < 0.35:
                score += 1
                reasons.append("Selection probability is low")
            if prob_gap < 0.10:
                score += 1
                reasons.append("Top-1 margin over the runner-up candidate is narrow")

            topk_entries = step.get("topk", [])
            best_corridor = next((entry for entry in topk_entries if entry.get("corridor_ok")), None)
            if best_corridor is not None and not bool(step.get("selected_corridor_ok", True)):
                score += 1
                reasons.append(f"Top-k already contained corridor-friendly candidate {best_corridor['node_label']}")

            if topk_entries:
                best_goal = max(topk_entries, key=lambda entry: float(entry.get("goal_progress_hops", -math.inf)))
                if (
                    best_goal.get("node_label") != selected_label
                    and float(best_goal.get("goal_progress_hops", -math.inf)) > selected_goal_progress + 1.0
                ):
                    score += 1
                    reasons.append(
                        f"Candidate {best_goal['node_label']} reduced goal hops more "
                        f"({best_goal.get('goal_progress_hops', math.nan):.1f} vs {selected_goal_progress:.1f})"
                    )

        level = "high" if score >= 5 else "medium" if score >= 3 else "ok"
        step["warning_score"] = score
        step["warning_level"] = level
        step["warning_reasons"] = reasons
        step["warning_summary"] = "; ".join(reasons[:3])

        for reason in reasons:
            reason_counts[reason] += 1
        if level != "ok":
            flagged_steps.append(
                {
                    "step": int(step["step"]),
                    "selected_node": selected_label,
                    "warning_level": level,
                    "warning_score": score,
                    "warning_reasons": reasons,
                }
            )

    flagged_steps.sort(key=lambda item: (-item["warning_score"], item["step"]))
    return {
        "flagged_count": len(flagged_steps),
        "flagged_steps": flagged_steps,
        "reason_counts": dict(reason_counts),
        "max_warning_score": max((item["warning_score"] for item in flagged_steps), default=0),
        "shortest_hops": shortest_hops,
    }


def _build_revisit_penalty(
    visit_counts: torch.Tensor,
    recent_nodes: torch.Tensor | None,
    reward_cfg: dict[str, float],
) -> torch.Tensor:
    penalty = visit_counts * reward_cfg["VISIT_LOGIT_PENALTY"]
    if recent_nodes is not None and recent_nodes.numel() > 0:
        candidate_nodes = torch.arange(visit_counts.size(1), device=visit_counts.device).unsqueeze(0)
        recent_hit = (candidate_nodes.unsqueeze(-1) == recent_nodes.unsqueeze(1)).any(dim=-1).float()
        penalty = penalty + recent_hit * reward_cfg["RECENT_NODE_PENALTY"]
    return penalty


def _compute_checkpoint_quality_single(
    env: DisasterEnv,
    start_idx: int,
    goal_idx: int,
    plan: list[int],
    max_reward: float,
) -> float:
    if not plan:
        return 0.0
    optimal_dist = float(env.apsp_matrix[start_idx, goal_idx].item())
    if optimal_dist < 1.0 or not math.isfinite(optimal_dist):
        return 0.0
    alpha = 0.6
    beta = 0.4
    waypoints = [start_idx] + plan + [goal_idx]
    total_segment_dist = 0.0
    segment_lengths = []
    for src, dst in zip(waypoints[:-1], waypoints[1:]):
        seg_dist = float(env.apsp_matrix[src, dst].item())
        total_segment_dist += seg_dist
        segment_lengths.append(seg_dist)
    efficiency = min(optimal_dist / max(total_segment_dist, 1.0), 1.0)
    if len(segment_lengths) > 1:
        mean_len = sum(segment_lengths) / len(segment_lengths)
        variance = sum((seg - mean_len) ** 2 for seg in segment_lengths) / len(segment_lengths)
        balance = 1.0 / (1.0 + math.sqrt(variance) / max(mean_len, 1.0))
    else:
        balance = 0.5
    quality = alpha * efficiency + beta * balance
    return float(quality * max_reward)


def compute_plan_diag_single(
    env: DisasterEnv,
    start_idx: int,
    goal_idx: int,
    plan: list[int],
    reward_cfg: dict[str, float],
) -> dict[str, Any]:
    shortest_hops_raw = float(env.hop_matrix[start_idx, goal_idx].item())
    shortest_hops = max(shortest_hops_raw, 1.0) if math.isfinite(shortest_hops_raw) else 1.0
    plan_length = len(plan)
    plan_density = plan_length / shortest_hops
    corridor_ok = []
    for subgoal in plan:
        s_to_c = float(env.hop_matrix[start_idx, subgoal].item())
        c_to_g = float(env.hop_matrix[subgoal, goal_idx].item())
        corridor_ok.append(bool((s_to_c + c_to_g) <= (shortest_hops + 2.0)))
    corridor_ratio = sum(corridor_ok) / max(plan_length, 1)
    density_excess = max(plan_density - reward_cfg["PLAN_DENSITY_TARGET"], 0.0)
    corridor_deficit = max(reward_cfg["PLAN_CORRIDOR_TARGET"] - corridor_ratio, 0.0)
    plan_len_ref = max(1.0, float(math.ceil(shortest_hops / max(reward_cfg["TARGET_SEGMENT_HOPS"], 1.0))))
    plan_len_min = 1.0 if shortest_hops <= 4.0 else plan_len_ref
    plan_len_max = plan_len_ref + float(reward_cfg["PLAN_COUNT_BAND"])
    empty_plan_penalty = reward_cfg["EMPTY_PLAN_PENALTY"] if plan_length == 0 else 0.0
    under_plan = max(plan_len_min - plan_length, 0.0)
    over_plan = max(plan_length - plan_len_max, 0.0)
    far_plan_penalty = reward_cfg["PLAN_COUNT_UNDER_PENALTY"] * under_plan
    over_plan_penalty = reward_cfg["PLAN_COUNT_OVER_PENALTY"] * over_plan
    optimal_path = env.reconstruct_weighted_shortest_path_indices(start_idx, goal_idx)
    if not optimal_path:
        optimal_path = [start_idx, goal_idx] if start_idx != goal_idx else [start_idx]
    path_hops = max(len(optimal_path) - 1, 1)
    max_path_index = max(len(optimal_path) - 1, 0)
    if max_path_index <= 0:
        min_anchor_pos = 0
        max_anchor_pos = 0
    else:
        min_anchor_pos = 1
        max_anchor_pos = max(1, max_path_index - 1)
    ref_token_count = max(1, int(plan_len_ref))
    reference_anchor_positions: list[int] = []
    for step_idx in range(1, ref_token_count + 1):
        pos = int(round(step_idx * path_hops / (ref_token_count + 1)))
        pos = max(min_anchor_pos, min(max_anchor_pos, pos))
        if reference_anchor_positions and pos < reference_anchor_positions[-1]:
            pos = reference_anchor_positions[-1]
        reference_anchor_positions.append(pos)
    reference_anchor_nodes = [optimal_path[pos] for pos in reference_anchor_positions]
    compare_len = min(plan_length, ref_token_count)
    anchor_targets = reference_anchor_nodes[:compare_len]
    if compare_len > 0:
        anchor_hop_errors = [
            min(float(env.hop_matrix[subgoal, anchor].item()), 4.0)
            for subgoal, anchor in zip(plan[:compare_len], anchor_targets)
        ]
        anchor_hop_err_mean = float(sum(anchor_hop_errors) / max(len(anchor_hop_errors), 1))
        anchor_near_rate = float(sum(err <= 1.0 for err in anchor_hop_errors) / max(len(anchor_hop_errors), 1))
    else:
        anchor_hop_errors = []
        anchor_hop_err_mean = 0.0
        anchor_near_rate = 0.0
    anchor_reward = (
        -reward_cfg["ANCHOR_HOP_PENALTY"] * anchor_hop_err_mean
        + reward_cfg["ANCHOR_NEAR_BONUS"] * anchor_near_rate
    )
    first_anchor_penalty = 0.0
    if plan_length > 0 and plan_len_ref >= 2.0 and reference_anchor_nodes:
        first_anchor_err = min(float(env.hop_matrix[plan[0], reference_anchor_nodes[0]].item()), 4.0)
        first_anchor_penalty = -reward_cfg["FIRST_ANCHOR_PENALTY"] * first_anchor_err
    plan_penalty = (
        - reward_cfg["PLAN_CORRIDOR_WEIGHT"] * corridor_deficit
        - far_plan_penalty
        - over_plan_penalty
        + anchor_reward
        + first_anchor_penalty
        + empty_plan_penalty
    )
    checkpoint_quality = _compute_checkpoint_quality_single(
        env,
        start_idx,
        goal_idx,
        plan,
        max_reward=reward_cfg["CHECKPOINT_MAX_REWARD"],
    )
    plan_adjustment = float(
        max(
            reward_cfg["PLAN_ADJUST_MIN"],
            min(reward_cfg["PLAN_ADJUST_MAX"], checkpoint_quality + plan_penalty),
        )
    )
    return {
        "shortest_hops": shortest_hops,
        "plan_length": plan_length,
        "plan_len_ref": plan_len_ref,
        "plan_len_min": plan_len_min,
        "plan_len_max": plan_len_max,
        "plan_density": plan_density,
        "density_excess": density_excess,
        "corridor_ratio": corridor_ratio,
        "corridor_ok_by_rank": corridor_ok,
        "corridor_deficit": corridor_deficit,
        "far_plan_penalty": far_plan_penalty,
        "first_subgoal_hops": float(env.hop_matrix[start_idx, plan[0]].item()) if plan else 0.0,
        "anchor_targets": anchor_targets,
        "anchor_hop_errors": anchor_hop_errors,
        "anchor_hop_err_mean": anchor_hop_err_mean,
        "anchor_near_rate": anchor_near_rate,
        "anchor_reward": anchor_reward,
        "first_anchor_penalty": first_anchor_penalty,
        "plan_penalty": plan_penalty,
        "checkpoint_quality": checkpoint_quality,
        "plan_adjustment": plan_adjustment,
    }


def manager_debug_rollout(
    manager: GraphTransformerManager,
    env: DisasterEnv,
    start_idx: int,
    goal_idx: int,
    max_len: int,
    temperature: float,
    topk: int,
) -> dict[str, Any]:
    device = env.device
    batch = env.pyg_data.batch
    x = env.pyg_data.x[:, :4]
    edge_attr = select_edge_attr(env.pyg_data.edge_attr)

    with torch.no_grad():
        memory, memory_mask = manager.encode_graph(x, env.pyg_data.edge_index, batch, edge_attr=edge_attr)
        batch_size = memory.size(0)
        eos_node = manager.eos_token_emb.expand(batch_size, -1, -1)
        memory_extended = torch.cat([memory, eos_node], dim=1)
        eos_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=device)
        memory_mask_extended = torch.cat([memory_mask, eos_mask], dim=1)
        curr_emb = manager.sos_emb.expand(batch_size, -1, -1)
        visited_mask = torch.zeros_like(memory_mask_extended, dtype=torch.bool)
        finished_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        eos_index = memory.size(1)

        steps: list[dict[str, Any]] = []
        full_sequence: list[int] = []
        node_positions = env.pos_tensor
        goal_dists = env.apsp_matrix[:, goal_idx].float()
        shortest_hops = max(_safe_hop_value(env, start_idx, goal_idx), 1.0)
        next_node_indices: torch.Tensor | None = None

        for step_idx in range(max_len):
            curr_input = curr_emb + manager.pos_emb[:, step_idx : step_idx + 1, :]
            padding_mask_float = torch.zeros(memory_mask_extended.size(), device=device, dtype=torch.float32)
            padding_mask_float.masked_fill_(~memory_mask_extended, float("-inf"))
            out = manager.decoder(tgt=curr_input, memory=memory_extended, memory_key_padding_mask=padding_mask_float)
            query = manager.pointer_query(out).unsqueeze(2)
            key = manager.pointer_key(memory_extended).unsqueeze(1)
            scores = torch.tanh(query + key)
            scores = torch.matmul(scores, manager.pointer_v).squeeze(-1).squeeze(1)
            scores.masked_fill_(~memory_mask_extended, float("-inf"))
            scores.masked_fill_(visited_mask, float("-inf"))

            current_node = int(next_node_indices[0].item()) if next_node_indices is not None else start_idx
            if current_node != eos_index:
                hops_from_curr = env.hop_matrix[current_node]
                radius_mask = (hops_from_curr < 3) | (hops_from_curr > 10)
                radius_mask[goal_idx] = False
                scores[0, :-1].masked_fill_(radius_mask.to(device), float("-inf"))

                curr_pos = node_positions[current_node]
                goal_pos = node_positions[goal_idx]
                target_vec = goal_pos - curr_pos
                target_norm = torch.norm(target_vec)
                if target_norm > 1e-5:
                    all_vecs = node_positions - curr_pos
                    all_norms = torch.norm(all_vecs, dim=1).clamp(min=1e-8)
                    cos_sim = (
                        all_vecs[:, 0] * target_vec[0] + all_vecs[:, 1] * target_vec[1]
                    ) / (all_norms * (target_norm + 1e-8))
                    angle_mask = cos_sim < -0.707
                    scores[0, :-1].masked_fill_(angle_mask.to(device), float("-inf"))

                bias_payload = compute_manager_decode_bias(
                    apsp_matrix=env.hop_matrix,
                    start_idx=start_idx,
                    current_idx=current_node,
                    goal_idx=goal_idx,
                    eos_index=eos_index,
                    cfg=getattr(manager, "decode_bias_cfg", None),
                    generated_tokens_so_far=step_idx,
                )
                scores[0] = scores[0] + bias_payload["total_bias"].to(device)
            else:
                bias_payload = compute_manager_decode_bias(
                    apsp_matrix=env.hop_matrix,
                    start_idx=start_idx,
                    current_idx=start_idx,
                    goal_idx=goal_idx,
                    eos_index=eos_index,
                    cfg=getattr(manager, "decode_bias_cfg", None),
                    generated_tokens_so_far=step_idx,
                )

            if finished_mask.any():
                scores[finished_mask] = float("-inf")
                scores[finished_mask, eos_index] = 0.0
            if step_idx == 0:
                scores[:, eos_index] = float("-inf")

            row_scores = scores[0]
            selected, probs = select_action(row_scores, temperature)
            full_sequence.append(selected)
            next_node_indices = torch.tensor([selected], dtype=torch.long, device=device)
            current_goal_hops = _safe_hop_value(env, current_node, goal_idx)
            topk_entries = _annotate_manager_topk(
                _extract_topk_entries(row_scores, probs, topk, eos_index=eos_index),
                env=env,
                start_idx=start_idx,
                goal_idx=goal_idx,
                current_node=current_node,
                eos_index=eos_index,
                shortest_hops=shortest_hops,
                bias_payload=bias_payload,
            )
            selected_metrics = _manager_candidate_metrics(
                env=env,
                start_idx=start_idx,
                goal_idx=goal_idx,
                current_node=current_node,
                candidate_idx=selected,
                eos_index=eos_index,
                shortest_hops=shortest_hops,
                bias_payload=bias_payload,
            )
            best_alt_prob = max(
                (float(entry["probability"]) for entry in topk_entries if int(entry["node_index"]) != selected),
                default=0.0,
            )
            selected_rank = next(
                (idx for idx, entry in enumerate(topk_entries, start=1) if int(entry["node_index"]) == selected),
                None,
            )

            steps.append(
                {
                    "step": step_idx,
                    "current_node": current_node,
                    "selected_node": selected,
                    "selected_label": "EOS" if selected == eos_index else str(selected),
                    "selected_probability": float(probs[selected].item()),
                    "selected_score": float(row_scores[selected].item()),
                    "current_goal_hops": current_goal_hops,
                    "selected_hops_from_current": float(selected_metrics["hops_from_current"]),
                    "selected_hops_from_start": float(selected_metrics["hops_from_start"]),
                    "selected_hops_to_goal": float(selected_metrics["hops_to_goal"]),
                    "selected_goal_hops": float(selected_metrics["goal_hops"]),
                    "selected_goal_progress_hops": float(selected_metrics["goal_progress_hops"]),
                    "selected_progress_ratio": float(selected_metrics["progress_ratio"]),
                    "selected_total_via_hops": float(selected_metrics["total_via_hops"]),
                    "selected_detour_hops": float(selected_metrics["detour_hops"]),
                    "selected_corridor_ok": bool(selected_metrics["corridor_ok"]),
                    "selected_corridor_slack": float(selected_metrics["corridor_slack"]),
                    "selected_eos_reasonable": bool(selected_metrics["eos_reasonable"]),
                    "selected_bias_total": float(selected_metrics["bias_total"]),
                    "selected_bias_corridor": float(selected_metrics["bias_corridor"]),
                    "selected_bias_progress": float(selected_metrics["bias_progress"]),
                    "selected_bias_detour": float(selected_metrics["bias_detour"]),
                    "selected_bias_nonprogress": float(selected_metrics["bias_nonprogress"]),
                    "selected_bias_eos": float(selected_metrics["bias_eos"]),
                    "selected_rank_in_topk": selected_rank,
                    "runnerup_probability": best_alt_prob,
                    "topk_prob_gap": float(probs[selected].item()) - best_alt_prob,
                    "entropy": categorical_entropy(probs),
                    "topk": topk_entries,
                    **_compute_expected_stats(probs, node_positions, goal_dists, eos_index=eos_index),
                }
            )

            if selected != eos_index:
                visited_mask[0, selected] = True
            finished_mask = finished_mask | (next_node_indices == eos_index)
            curr_emb = memory_extended[torch.arange(batch_size, device=device), next_node_indices].unsqueeze(1)
            if finished_mask.all():
                break

    plan = truncate_plan(full_sequence, eos_index=eos_index, num_nodes=env.num_nodes)
    diagnosis = diagnose_manager_steps(steps, shortest_hops=shortest_hops)
    return {
        "raw_sequence": full_sequence,
        "eos_index": eos_index,
        "plan": plan,
        "steps": steps,
        "diagnosis": diagnosis,
        "plan_diag": compute_plan_diag_single(env, start_idx, goal_idx, plan, DEFAULT_REWARD_CFG),
    }


def worker_debug_rollout(
    worker: WorkerLSTM,
    env: DisasterEnv,
    goal_idx: int,
    manager_debug: dict[str, Any],
    max_total_steps: int,
    temperature: float,
    topk: int,
) -> dict[str, Any]:
    device = env.device
    ensure_env_node_feature_layout(env)
    raw_sequence = manager_debug["raw_sequence"]
    if raw_sequence:
        sequence_tensor = torch.tensor(raw_sequence, dtype=torch.long, device=device).unsqueeze(0)
    else:
        sequence_tensor = torch.tensor([[env.num_nodes]], dtype=torch.long, device=device)
    valid_mask = (sequence_tensor >= 0) & (sequence_tensor < env.num_nodes)
    safe_sequences = torch.where(valid_mask, sequence_tensor, torch.full_like(sequence_tensor, goal_idx))
    generated_plan_length = int(valid_mask.sum().item())

    h = torch.zeros(1, worker.lstm.hidden_size, device=device)
    c = torch.zeros(1, worker.lstm.hidden_size, device=device)
    visit_counts = torch.zeros(1, env.num_nodes, device=device)
    path_history = [env.current_node.clone()]
    path_nodes = [int(env.current_node.item())]
    worker_steps: list[dict[str, Any]] = []
    subgoal_ptr = 0
    reached_subgoals = 0
    failed_early = False

    goal_dist_start = float(env.apsp_matrix[path_nodes[0], goal_idx].item())
    max_subgoal_progress = 0.0
    min_hop_to_active_subgoal = float("inf")

    initial_target = int(safe_sequences[0, 0].item()) if sequence_tensor.size(1) > 0 else goal_idx
    env.update_target_features(torch.tensor([initial_target], dtype=torch.long, device=device))
    current_subgoal_start_hops = max(float(env.hop_matrix[path_nodes[0], initial_target].item()), 1.0)

    with torch.no_grad():
        for step_idx in range(max_total_steps):
            current_node = int(env.current_node.item())
            if current_node == goal_idx:
                break

            curr_nodes = env.current_node.unsqueeze(1)
            visit_counts.scatter_add_(1, curr_nodes, torch.ones((1, 1), device=device))
            curr_visits = int(torch.gather(visit_counts, 1, curr_nodes).item())
            if curr_visits > int(DEFAULT_REWARD_CFG["LOOP_LIMIT"]):
                failed_early = True
                break

            current_target = int(env.target_node.item())
            valid_active_subgoal = subgoal_ptr < generated_plan_length
            if valid_active_subgoal:
                current_subgoal = int(safe_sequences[0, subgoal_ptr].item())
                active_subgoal_hops = float(env.hop_matrix[current_node, current_subgoal].item())
                min_hop_to_active_subgoal = min(min_hop_to_active_subgoal, active_subgoal_hops)
                subgoal_progress = 1.0 - active_subgoal_hops / max(current_subgoal_start_hops, 1.0)
                max_subgoal_progress = max(max_subgoal_progress, min(max(subgoal_progress, 0.0), 1.0))
                if current_node == current_subgoal:
                    reached_subgoals += 1
                    subgoal_ptr = min(subgoal_ptr + 1, max(sequence_tensor.size(1) - 1, 0))
                    next_target = int(safe_sequences[0, subgoal_ptr].item()) if subgoal_ptr < sequence_tensor.size(1) else goal_idx
                    env.update_target_features(torch.tensor([next_target], dtype=torch.long, device=device))
                    current_target = next_target
                    current_subgoal_start_hops = max(float(env.hop_matrix[current_node, current_target].item()), 1.0)
            else:
                env.update_target_features(torch.tensor([goal_idx], dtype=torch.long, device=device))
                current_target = goal_idx

            env_x = env.pyg_data.x
            worker_input = torch.cat([env_x[:, :4], env_x[:, 5:]], dim=1)
            edge_attr = select_edge_attr(env.pyg_data.edge_attr)
            scores, h_next, c_next, value_pred = worker.predict_next_hop(
                worker_input,
                env.pyg_data.edge_index,
                h,
                c,
                env.pyg_data.batch,
                detach_spatial=False,
                edge_attr=edge_attr,
            )
            h, c = h_next, c_next
            row_scores = scores.view(1, -1)[0]
            mask = env.get_mask()[0]
            recent_nodes = torch.stack(path_history[-int(DEFAULT_REWARD_CFG["RECENT_WINDOW"]) :], dim=1)
            penalties = _build_revisit_penalty(visit_counts, recent_nodes, DEFAULT_REWARD_CFG)[0]
            adjusted_scores = row_scores - penalties
            adjusted_scores = adjusted_scores.masked_fill(~mask.bool(), float("-inf"))
            selected, probs = select_action(adjusted_scores, temperature)

            goal_dists = env.apsp_matrix[:, goal_idx].float()
            goal_distance_before = float(env.apsp_matrix[current_node, goal_idx].item())
            env.step(torch.tensor([selected], dtype=torch.long, device=device))
            next_node = int(env.current_node.item())
            goal_distance_after = float(env.apsp_matrix[next_node, goal_idx].item())

            worker_steps.append(
                {
                    "step": step_idx,
                    "current_node": current_node,
                    "target_node": current_target,
                    "selected_node": selected,
                    "selected_probability": float(probs[selected].item()),
                    "selected_score": float(adjusted_scores[selected].item()),
                    "raw_score": float(row_scores[selected].item()),
                    "revisit_penalty": float(penalties[selected].item()),
                    "entropy": categorical_entropy(probs),
                    "critic_value": float(value_pred.squeeze().item()),
                    "goal_distance_before": goal_distance_before,
                    "goal_distance_after": goal_distance_after,
                    "topk": _extract_topk_entries(adjusted_scores, probs, topk, penalties=penalties),
                    **_compute_expected_stats(probs, env.pos_tensor, goal_dists, eos_index=None),
                }
            )

            path_nodes.append(next_node)
            path_history.append(env.current_node.clone())

    final_node = int(env.current_node.item())
    final_goal_distance = float(env.apsp_matrix[final_node, goal_idx].item())
    success = final_node == goal_idx
    progress = 0.0
    goal_per100 = 0.0
    if math.isfinite(goal_dist_start) and goal_dist_start > 0.0:
        progress = 1.0 - final_goal_distance / goal_dist_start
        goal_per100 = ((goal_dist_start - final_goal_distance) / max(len(worker_steps), 1)) * 100.0 / goal_dist_start
    subgoal_per100 = (reached_subgoals / max(len(worker_steps), 1)) * 100.0

    return {
        "success": success,
        "failed_early": failed_early,
        "path_nodes": path_nodes,
        "final_node": final_node,
        "steps": len(worker_steps),
        "worker_steps": worker_steps,
        "generated_plan_length": generated_plan_length,
        "reached_subgoals": reached_subgoals,
        "plan_utilization": reached_subgoals / max(generated_plan_length, 1),
        "final_goal_distance": final_goal_distance,
        "progress": progress,
        "goal_per100": goal_per100,
        "subgoal_per100": subgoal_per100,
        "max_subgoal_progress": max_subgoal_progress,
        "min_hop_to_active_subgoal": 0.0 if min_hop_to_active_subgoal == float("inf") else min_hop_to_active_subgoal,
    }


def worker_goal_debug_rollout(
    worker: WorkerLSTM,
    env: DisasterEnv,
    goal_idx: int,
    max_total_steps: int,
    temperature: float,
    topk: int,
) -> dict[str, Any]:
    device = env.device
    ensure_env_node_feature_layout(env)
    env.update_target_features(
        torch.tensor([goal_idx], dtype=torch.long, device=device),
        torch.ones(1, device=device),
    )

    h = torch.zeros(1, worker.lstm.hidden_size, device=device)
    c = torch.zeros(1, worker.lstm.hidden_size, device=device)
    visit_counts = torch.zeros(1, env.num_nodes, device=device)
    path_history = [env.current_node.clone()]
    path_nodes = [int(env.current_node.item())]
    worker_steps: list[dict[str, Any]] = []
    failed_early = False

    goal_dist_start = float(env.apsp_matrix[path_nodes[0], goal_idx].item())

    with torch.no_grad():
        for step_idx in range(max_total_steps):
            current_node = int(env.current_node.item())
            if current_node == goal_idx:
                break

            curr_nodes = env.current_node.unsqueeze(1)
            visit_counts.scatter_add_(1, curr_nodes, torch.ones((1, 1), device=device))
            curr_visits = int(torch.gather(visit_counts, 1, curr_nodes).item())
            if curr_visits > int(DEFAULT_REWARD_CFG["LOOP_LIMIT"]):
                failed_early = True
                break

            env.update_target_features(
                torch.tensor([goal_idx], dtype=torch.long, device=device),
                torch.ones(1, device=device),
            )
            env_x = env.pyg_data.x
            # Worker 입력: visit(채널4) 제외, 나머지 8채널 사용 (pomo_trainer.py와 동일)
            worker_input = torch.cat([env_x[:, :4], env_x[:, 5:]], dim=1)
            edge_attr = select_edge_attr(env.pyg_data.edge_attr)
            scores, h_next, c_next, value_pred = worker.predict_next_hop(
                worker_input,
                env.pyg_data.edge_index,
                h,
                c,
                env.pyg_data.batch,
                detach_spatial=False,
                edge_attr=edge_attr,
            )
            h, c = h_next, c_next
            row_scores = scores.view(1, -1)[0]
            mask = env.get_mask()[0]
            recent_nodes = torch.stack(path_history[-int(DEFAULT_REWARD_CFG["RECENT_WINDOW"]) :], dim=1)
            penalties = _build_revisit_penalty(visit_counts, recent_nodes, DEFAULT_REWARD_CFG)[0]
            adjusted_scores = row_scores - penalties
            adjusted_scores = adjusted_scores.masked_fill(~mask.bool(), float("-inf"))
            selected, probs = select_action(adjusted_scores, temperature)

            goal_dists = env.apsp_matrix[:, goal_idx].float()
            goal_distance_before = float(env.apsp_matrix[current_node, goal_idx].item())
            env.step(torch.tensor([selected], dtype=torch.long, device=device))
            env.update_target_features(
                torch.tensor([goal_idx], dtype=torch.long, device=device),
                torch.ones(1, device=device),
            )
            next_node = int(env.current_node.item())
            goal_distance_after = float(env.apsp_matrix[next_node, goal_idx].item())

            worker_steps.append(
                {
                    "step": step_idx,
                    "current_node": current_node,
                    "target_node": goal_idx,
                    "selected_node": selected,
                    "selected_probability": float(probs[selected].item()),
                    "selected_score": float(adjusted_scores[selected].item()),
                    "raw_score": float(row_scores[selected].item()),
                    "revisit_penalty": float(penalties[selected].item()),
                    "entropy": categorical_entropy(probs),
                    "critic_value": float(value_pred.squeeze().item()),
                    "goal_distance_before": goal_distance_before,
                    "goal_distance_after": goal_distance_after,
                    "topk": _extract_topk_entries(adjusted_scores, probs, topk, penalties=penalties),
                    **_compute_expected_stats(probs, env.pos_tensor, goal_dists, eos_index=None),
                }
            )

            path_nodes.append(next_node)
            path_history.append(env.current_node.clone())

    final_node = int(env.current_node.item())
    final_goal_distance = float(env.apsp_matrix[final_node, goal_idx].item())
    success = final_node == goal_idx
    progress = 0.0
    goal_per100 = 0.0
    if math.isfinite(goal_dist_start) and goal_dist_start > 0.0:
        progress = 1.0 - final_goal_distance / goal_dist_start
        goal_per100 = (
            (goal_dist_start - final_goal_distance) / max(len(worker_steps), 1)
        ) * 100.0 / goal_dist_start

    return {
        "success": success,
        "failed_early": failed_early,
        "path_nodes": path_nodes,
        "final_node": final_node,
        "steps": len(worker_steps),
        "worker_steps": worker_steps,
        "final_goal_distance": final_goal_distance,
        "progress": progress,
        "goal_per100": goal_per100,
    }


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def flatten_step_records(steps: list[dict[str, Any]], topk_slots: int = 5) -> pd.DataFrame:
    rows = []
    for step in steps:
        row = {k: v for k, v in step.items() if k != "topk"}
        for idx, entry in enumerate(step.get("topk", [])[:topk_slots], start=1):
            row[f"top{idx}_node"] = entry["node_label"]
            row[f"top{idx}_prob"] = entry["probability"]
            row[f"top{idx}_score"] = entry["score"]
            if "corridor_ok" in entry:
                row[f"top{idx}_corridor_ok"] = entry["corridor_ok"]
            if "goal_progress_hops" in entry:
                row[f"top{idx}_goal_progress_hops"] = entry["goal_progress_hops"]
            if "detour_hops" in entry:
                row[f"top{idx}_detour_hops"] = entry["detour_hops"]
            if "bias_total" in entry:
                row[f"top{idx}_bias_total"] = entry["bias_total"]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_rollout_map(
    output_path: Path,
    env: DisasterEnv,
    start_idx: int,
    goal_idx: int,
    manager_plan: list[int],
    worker_path: list[int],
    checkpoint_label: str,
    success: bool,
    manager_steps: list[dict[str, Any]] | None = None,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=True)
    positions = env.pos_tensor.detach().cpu().numpy()

    for src, dst in env.map_core.graph.edges():
        u = env.node_mapping[src]
        v = env.node_mapping[dst]
        ax.plot([positions[u, 0], positions[v, 0]], [positions[u, 1], positions[v, 1]], color="#b9c2cf", linewidth=0.6, alpha=0.35, zorder=1)

    if manager_plan:
        planned_nodes = [start_idx] + manager_plan + [goal_idx]
        ax.plot(positions[planned_nodes, 0], positions[planned_nodes, 1], linestyle="--", color="#ff7f0e", linewidth=2.0, alpha=0.95, label="Manager plan", zorder=3)
        ax.scatter(positions[manager_plan, 0], positions[manager_plan, 1], marker="*", s=250, facecolors="#ffbf00", edgecolors="black", linewidths=1.0, zorder=5)
        for rank, node in enumerate(manager_plan, start=1):
            ax.annotate(f"SG{rank}\n(N={node})", (positions[node, 0], positions[node, 1]),
                        textcoords="offset points", xytext=(8, 8),
                        fontsize=9, fontweight="bold", ha="left", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff8e8", edgecolor="#ff7f0e", alpha=0.9), zorder=6)
    if manager_steps:
        suspicious = [
            step for step in manager_steps if step.get("warning_level") != "ok" and step.get("selected_label") != "EOS"
        ]
        if suspicious:
            suspicious_nodes = [int(step["selected_node"]) for step in suspicious]
            ax.scatter(
                positions[suspicious_nodes, 0],
                positions[suspicious_nodes, 1],
                s=240,
                facecolors="none",
                edgecolors="#d62728",
                linewidths=2.0,
                label="Flagged manager step",
                zorder=8,
            )
            for step in suspicious:
                node = int(step["selected_node"])
                ax.annotate(
                    f"!{step['step']}",
                    xy=(positions[node, 0], positions[node, 1]),
                    xytext=(0, 10),
                    textcoords="offset points",
                    color="#d62728",
                    fontsize=9,
                    fontweight="bold",
                    ha="center",
                    va="bottom",
                    zorder=9,
                )

    if len(worker_path) > 1:
        cmap = plt.get_cmap("viridis")
        denom = max(len(worker_path) - 1, 1)
        for idx in range(len(worker_path) - 1):
            src = worker_path[idx]
            dst = worker_path[idx + 1]
            ax.plot([positions[src, 0], positions[dst, 0]], [positions[src, 1], positions[dst, 1]], color=cmap(idx / denom), linewidth=2.6, alpha=0.95, zorder=6)
        scatter = ax.scatter(positions[worker_path, 0], positions[worker_path, 1], c=np.arange(len(worker_path)), cmap="viridis", s=26, zorder=7, label="Worker path")
        # Colorbar를 하단 수평으로 배치하여 지도 비율 유지
        cbar = fig.colorbar(scatter, ax=ax, label="Worker step", orientation="horizontal", pad=0.06, shrink=0.6, aspect=30)
        cbar.ax.tick_params(labelsize=9)

    ax.scatter(positions[start_idx, 0], positions[start_idx, 1], marker="o", s=150, color="#2ca02c", label="Start", zorder=8)
    ax.scatter(positions[goal_idx, 0], positions[goal_idx, 1], marker="X", s=170, color="#d62728", label="Goal", zorder=8)
    final_node = worker_path[-1]
    if final_node not in {start_idx, goal_idx}:
        ax.scatter(positions[final_node, 0], positions[final_node, 1], marker="D", s=90, color="#1f77b4" if success else "#9467bd", label="Final node", zorder=8)

    status = "SUCCESS" if success else "FAIL"
    ax.set_title(f"Rollout Map - {checkpoint_label} ({status})", fontsize=16, fontweight="bold")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    # Legend를 지도 바깥 좌측 상단에 배치하여 지도와 겹치지 않게 함
    ax.legend(frameon=True, framealpha=0.85, edgecolor="#cccccc", loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    focus_nodes = list(dict.fromkeys([start_idx, goal_idx] + manager_plan + worker_path))
    if focus_nodes:
        focus_xy = positions[focus_nodes]
        x_min, y_min = focus_xy.min(axis=0)
        x_max, y_max = focus_xy.max(axis=0)
        x_span = max(float(x_max - x_min), 1e-3)
        y_span = max(float(y_max - y_min), 1e-3)
        x_margin = max(x_span * 0.12, 0.0025)
        y_margin = max(y_span * 0.12, 0.0025)
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_step_grid(
    output_path: Path,
    steps: list[dict[str, Any]],
    title: str,
    panel_prefix: str,
    max_panels: int,
) -> None:
    steps = steps[:max_panels]
    if not steps:
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        ax.text(0.5, 0.5, "No step data", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    cols = 2
    rows = math.ceil(len(steps) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.2 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(rows, cols)
    fig.suptitle(title, fontsize=16, fontweight="bold")

    for ax, step in zip(axes.flat, steps):
        entries = step.get("topk", [])
        labels = [entry["node_label"] for entry in entries]
        probs = [entry["probability"] * 100.0 for entry in entries]
        selected_label = step.get("selected_label", str(step.get("selected_node")))
        warning_level = step.get("warning_level", "ok")
        colors = []
        for entry in entries:
            if entry["node_label"] == selected_label:
                if warning_level == "high":
                    colors.append("#d62728")
                elif warning_level == "medium":
                    colors.append("#ff7f0e")
                else:
                    colors.append("#ff7f0e")
            elif entry.get("corridor_ok"):
                colors.append("#2ca02c")
            else:
                colors.append("#4c78a8")
        ax.bar(range(len(labels)), probs, color=colors, alpha=0.9)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("Prob (%)")
        headline = (
            f"{panel_prefix} {step['step']} | select={selected_label} "
            f"| p={step.get('selected_probability', math.nan) * 100:.1f}%"
        )
        if warning_level != "ok":
            headline += f" | {warning_level.upper()}"
        subtitle_bits = []
        if "selected_score" in step:
            subtitle_bits.append(f"score={step['selected_score']:.3f}")
        if "expected_goal_distance" in step:
            subtitle_bits.append(f"E[d_goal]={step['expected_goal_distance']:.0f}")
        if "critic_value" in step:
            subtitle_bits.append(f"value={step['critic_value']:.3f}")
        if "selected_goal_progress_hops" in step:
            subtitle_bits.append(f"goal-delta={step['selected_goal_progress_hops']:.1f}")
        if "selected_detour_hops" in step:
            subtitle_bits.append(f"detour={step['selected_detour_hops']:.1f}")
        if "selected_corridor_ok" in step:
            subtitle_bits.append("corridor=OK" if step["selected_corridor_ok"] else "corridor=OUT")
        if "selected_bias_total" in step:
            subtitle_bits.append(f"bias={step['selected_bias_total']:+.2f}")
        ax.set_title(headline + ("\n" + ", ".join(subtitle_bits) if subtitle_bits else ""), fontsize=10)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        if step.get("warning_summary"):
            ax.text(
                0.01,
                0.98,
                step["warning_summary"],
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="#7f0000" if warning_level == "high" else "#6b4f00",
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "#fff5f5" if warning_level == "high" else "#fff8e8",
                    "edgecolor": "#d62728" if warning_level == "high" else "#ffbf00",
                    "alpha": 0.95,
                },
            )
        if "selected_bias_total" in step:
            ax.text(
                0.01,
                0.02,
                (
                    f"bias total={step.get('selected_bias_total', 0.0):+.2f}, "
                    f"corr={step.get('selected_bias_corridor', 0.0):+.2f}, "
                    f"prog={step.get('selected_bias_progress', 0.0):+.2f}, "
                    f"det={step.get('selected_bias_detour', 0.0):+.2f}, "
                    f"nonprog={step.get('selected_bias_nonprogress', 0.0):+.2f}, "
                    f"eos={step.get('selected_bias_eos', 0.0):+.2f}"
                ),
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=7.5,
                color="#333333",
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "#f6f6f6",
                    "edgecolor": "#c7c7c7",
                    "alpha": 0.95,
                },
            )

    for ax in axes.flat[len(steps) :]:
        ax.axis("off")

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_worker_trace_summary(output_path: Path, worker_steps: list[dict[str, Any]]) -> None:
    if not worker_steps:
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        ax.text(0.5, 0.5, "No worker trace data", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    df = pd.DataFrame(worker_steps)

    # 개별 차트 저장 (worker_charts/ 서브디렉토리)
    wc_dir = output_path.parent / "worker_charts"
    wc_dir.mkdir(parents=True, exist_ok=True)

    # 1. Action Probability
    fig1, ax1 = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax1.plot(df["step"], df["selected_probability"] * 100.0, color="#ff7f0e", linewidth=2.0)
    ax1.set_title("Selected Action Probability", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Prob (%)")
    ax1.set_xlabel("Worker step")
    ax1.grid(alpha=0.25, linestyle="--")
    fig1.savefig(wc_dir / "01_action_probability.png", dpi=200, bbox_inches="tight")
    plt.close(fig1)

    # 2. Goal Distance
    fig2, ax2 = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax2.plot(df["step"], df["goal_distance_before"], label="Before", linewidth=2.0)
    ax2.plot(df["step"], df["goal_distance_after"], label="After", linewidth=2.0)
    ax2.set_title("Goal Distance by Step", fontsize=11, fontweight="bold")
    ax2.set_ylabel("APSP distance")
    ax2.set_xlabel("Worker step")
    ax2.legend(frameon=False)
    ax2.grid(alpha=0.25, linestyle="--")
    fig2.savefig(wc_dir / "02_goal_distance.png", dpi=200, bbox_inches="tight")
    plt.close(fig2)

    # 3. Critic Value
    fig3, ax3 = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax3.plot(df["step"], df["critic_value"], color="#2ca02c", linewidth=2.0)
    ax3.set_title("Critic Value", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Value")
    ax3.set_xlabel("Worker step")
    ax3.grid(alpha=0.25, linestyle="--")
    fig3.savefig(wc_dir / "03_critic_value.png", dpi=200, bbox_inches="tight")
    plt.close(fig3)

    # 4. Exploration vs Penalty
    fig4, ax4 = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax4.plot(df["step"], df["entropy"], label="Entropy", linewidth=2.0)
    ax4.plot(df["step"], df["revisit_penalty"], label="Revisit penalty", linewidth=2.0)
    ax4.set_title("Exploration vs Penalty", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Worker step")
    ax4.legend(frameon=False)
    ax4.grid(alpha=0.25, linestyle="--")
    fig4.savefig(wc_dir / "04_exploration_penalty.png", dpi=200, bbox_inches="tight")
    plt.close(fig4)

    # 합본도 유지
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle("Worker Trace Summary", fontsize=16, fontweight="bold")

    axes[0, 0].plot(df["step"], df["selected_probability"] * 100.0, color="#ff7f0e", linewidth=2.0)
    axes[0, 0].set_title("Selected Action Probability")
    axes[0, 0].set_ylabel("Prob (%)")
    axes[0, 0].grid(alpha=0.25, linestyle="--")

    axes[0, 1].plot(df["step"], df["goal_distance_before"], label="Before", linewidth=2.0)
    axes[0, 1].plot(df["step"], df["goal_distance_after"], label="After", linewidth=2.0)
    axes[0, 1].set_title("Goal Distance by Step")
    axes[0, 1].set_ylabel("APSP distance")
    axes[0, 1].legend(frameon=False)
    axes[0, 1].grid(alpha=0.25, linestyle="--")

    axes[1, 0].plot(df["step"], df["critic_value"], color="#2ca02c", linewidth=2.0)
    axes[1, 0].set_title("Critic Value")
    axes[1, 0].set_ylabel("Value")
    axes[1, 0].set_xlabel("Worker step")
    axes[1, 0].grid(alpha=0.25, linestyle="--")

    axes[1, 1].plot(df["step"], df["entropy"], label="Entropy", linewidth=2.0)
    axes[1, 1].plot(df["step"], df["revisit_penalty"], label="Revisit penalty", linewidth=2.0)
    axes[1, 1].set_title("Exploration vs Penalty")
    axes[1, 1].set_xlabel("Worker step")
    axes[1, 1].legend(frameon=False)
    axes[1, 1].grid(alpha=0.25, linestyle="--")

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_manager_diagnostic_summary(
    output_path: Path,
    manager_steps: list[dict[str, Any]],
    diagnosis: dict[str, Any],
) -> None:
    if not manager_steps:
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        ax.text(0.5, 0.5, "No manager trace data", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    df = pd.DataFrame(manager_steps)
    colors = df["warning_level"].map({"high": "#d62728", "medium": "#ffbf00", "ok": "#2ca02c"}).fillna("#4c78a8")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    fig.suptitle("Manager Step Diagnostics", fontsize=16, fontweight="bold")

    axes[0, 0].plot(df["step"], df["current_goal_hops"], color="#7f7f7f", linewidth=2.0, label="Current goal hops")
    axes[0, 0].plot(df["step"], df["selected_goal_hops"], color="#1f77b4", linewidth=2.0, label="Selected goal hops")
    axes[0, 0].scatter(df["step"], df["selected_goal_hops"], c=colors, s=54, zorder=3)
    axes[0, 0].set_title("Remaining Goal Hops by Manager Step")
    axes[0, 0].set_ylabel("Hop count")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].grid(alpha=0.25, linestyle="--")

    axes[0, 1].axhline(0.0, color="#999999", linewidth=1.0, linestyle="--")
    axes[0, 1].plot(df["step"], df["selected_goal_progress_hops"], color="#2ca02c", linewidth=2.0, label="Goal-hop improvement")
    axes[0, 1].plot(df["step"], df["selected_detour_hops"], color="#d62728", linewidth=2.0, label="Detour hops")
    axes[0, 1].scatter(df["step"], df["selected_goal_progress_hops"], c=colors, s=54, zorder=3)
    axes[0, 1].set_title("Step Quality Signals")
    axes[0, 1].set_ylabel("Hop delta")
    axes[0, 1].legend(frameon=False)
    axes[0, 1].grid(alpha=0.25, linestyle="--")

    axes[1, 0].plot(df["step"], df["selected_probability"] * 100.0, color="#ff7f0e", linewidth=2.0, label="Selected prob")
    axes[1, 0].plot(df["step"], df["topk_prob_gap"] * 100.0, color="#9467bd", linewidth=2.0, label="Top-1 gap")
    if "eos_probability" in df.columns:
        axes[1, 0].plot(df["step"], df["eos_probability"] * 100.0, color="#8c564b", linewidth=1.8, label="EOS prob")
    axes[1, 0].scatter(df["step"], df["selected_probability"] * 100.0, c=colors, s=54, zorder=3)
    axes[1, 0].set_title("Confidence and EOS Pressure")
    axes[1, 0].set_xlabel("Manager step")
    axes[1, 0].set_ylabel("Percent")
    axes[1, 0].legend(frameon=False)
    axes[1, 0].grid(alpha=0.25, linestyle="--")

    axes[1, 1].axis("off")
    flagged_steps = diagnosis.get("flagged_steps", [])
    reason_counts = diagnosis.get("reason_counts", {})
    summary_lines = [
        f"Flagged steps: {diagnosis.get('flagged_count', 0)}",
        f"Max warning score: {diagnosis.get('max_warning_score', 0)}",
        f"Avg selected bias: {df['selected_bias_total'].mean():+.2f}" if "selected_bias_total" in df.columns else "Avg selected bias: n/a",
        (
            f"Bias mix (corr/prog/det/nonprog/eos): "
            f"{df['selected_bias_corridor'].mean():+.2f} / "
            f"{df['selected_bias_progress'].mean():+.2f} / "
            f"{df['selected_bias_detour'].mean():+.2f} / "
            f"{df['selected_bias_nonprogress'].mean():+.2f} / "
            f"{df['selected_bias_eos'].mean():+.2f}"
        ) if {"selected_bias_corridor", "selected_bias_progress", "selected_bias_detour", "selected_bias_nonprogress", "selected_bias_eos"}.issubset(df.columns) else "Bias mix: n/a",
        "",
        "Top suspicious steps:",
    ]
    if flagged_steps:
        for item in flagged_steps[:5]:
            reasons = "; ".join(item.get("warning_reasons", [])[:2]) or "No details"
            summary_lines.append(
                f"- step {item['step']} -> {item['selected_node']} ({item['warning_level']}): {reasons}"
            )
    else:
        summary_lines.append("- none")
    summary_lines.extend(["", "Most common warning reasons:"])
    if reason_counts:
        for reason, count in sorted(reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]:
            summary_lines.append(f"- {count}x {reason}")
    else:
        summary_lines.append("- none")
    axes[1, 1].text(
        0.02,
        0.98,
        "\n".join(summary_lines),
        ha="left",
        va="top",
        fontsize=10,
        family="monospace",
        transform=axes[1, 1].transAxes,
    )

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_step_panels(
    output_dir: Path,
    manager_steps: list[dict[str, Any]],
    worker_steps: list[dict[str, Any]],
    max_worker_step_plots: int,
) -> None:
    manager_dir = output_dir / "manager_steps"
    worker_dir = output_dir / "worker_steps"
    manager_dir.mkdir(parents=True, exist_ok=True)
    worker_dir.mkdir(parents=True, exist_ok=True)

    for step in manager_steps:
        plot_step_grid(manager_dir / f"manager_step_{step['step']:02d}.png", [step], f"Manager Step {step['step']}", "Mgr", 1)
    for step in worker_steps[:max_worker_step_plots]:
        plot_step_grid(worker_dir / f"worker_step_{step['step']:03d}.png", [step], f"Worker Step {step['step']}", "Wkr", 1)


def build_rollout_markdown(
    checkpoint_info: dict[str, Any],
    start_idx: int,
    goal_idx: int,
    manager_debug: dict[str, Any],
    worker_debug: dict[str, Any],
) -> str:
    plan_diag = manager_debug["plan_diag"]
    diagnosis = manager_debug.get("diagnosis", {})
    flagged_steps = diagnosis.get("flagged_steps", [])
    lines = [
        "# Rollout Debug Summary",
        "",
        f"- Checkpoint: `{checkpoint_info['display']}`",
        f"- Start / Goal: `{start_idx} -> {goal_idx}`",
        f"- Success: `{worker_debug['success']}`",
        f"- Final node: `{worker_debug['final_node']}`",
        f"- Worker steps: `{worker_debug['steps']}`",
        f"- Manager plan: `{manager_debug['plan']}`",
        f"- Plan length: `{plan_diag['plan_length']}`",
        f"- Plan density: `{plan_diag['plan_density']:.3f}`",
        f"- Corridor ratio: `{plan_diag['corridor_ratio'] * 100.0:.1f}%`",
        f"- Plan adjustment estimate: `{plan_diag['plan_adjustment']:.3f}`",
        f"- Manager flagged steps: `{diagnosis.get('flagged_count', 0)}`",
        f"- Manager avg selected bias: `{np.mean([step.get('selected_bias_total', 0.0) for step in manager_debug['steps']]) if manager_debug['steps'] else 0.0:.3f}`",
        f"- Plan utilization: `{worker_debug['plan_utilization'] * 100.0:.1f}%`",
        f"- Reached subgoals: `{worker_debug['reached_subgoals']}/{worker_debug['generated_plan_length']}`",
        f"- Final goal distance: `{worker_debug['final_goal_distance']:.1f}`",
        f"- Progress: `{worker_debug['progress'] * 100.0:.1f}%`",
        f"- Goal / SG per100: `{worker_debug['goal_per100']:.3f}` / `{worker_debug['subgoal_per100']:.3f}`",
        "",
    ]
    if flagged_steps:
        lines.append("## Suspicious Manager Steps")
        lines.append("")
        for item in flagged_steps[:5]:
            reasons = "; ".join(item.get("warning_reasons", [])[:3]) or "No details"
            lines.append(
                f"- Step {item['step']} -> `{item['selected_node']}` ({item['warning_level']}, score={item['warning_score']}): {reasons}"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def build_apte_rollout_markdown(
    checkpoint_info: dict[str, Any],
    start_idx: int,
    goal_idx: int,
    worker_debug: dict[str, Any],
) -> str:
    lines = [
        "# APTE Phase1 Rollout Summary",
        "",
        f"- Checkpoint: `{checkpoint_info['display']}`",
        f"- Start / Goal: `{start_idx} -> {goal_idx}`",
        f"- Success: `{worker_debug['success']}`",
        f"- Final node: `{worker_debug['final_node']}`",
        f"- Worker steps: `{worker_debug['steps']}`",
        f"- Final goal distance: `{worker_debug['final_goal_distance']:.1f}`",
        f"- Progress: `{worker_debug['progress'] * 100.0:.1f}%`",
        f"- Goal per100: `{worker_debug['goal_per100']:.3f}`",
        "",
        "This APTE branch rollout uses worker-only goal-conditioned execution.",
        "No runtime manager, subgoal pointer, handoff, or skip logic is involved.",
        "",
    ]
    return "\n".join(lines)


def run_rollout_visualization(
    run_dir: Path,
    dashboard_dir: Path,
    checkpoint_info: dict[str, Any],
    args: argparse.Namespace,
) -> Path:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    env, manager, worker = load_models_for_rollout(
        run_dir=run_dir,
        checkpoint_info=checkpoint_info,
        map_name=args.map,
        disaster=args.disaster,
        hidden_dim=args.hidden_dim,
        device=device,
    )
    start_idx, goal_idx = configure_single_problem(env, args.start_node, args.goal_node)
    rollout_dir = dashboard_dir / f"rollout_{checkpoint_info['label']}"
    rollout_dir.mkdir(parents=True, exist_ok=True)

    if checkpoint_info["source"] in ("apte_phase1", "worker_only"):
        worker_debug = worker_goal_debug_rollout(
            worker=worker,
            env=env,
            goal_idx=goal_idx,
            max_total_steps=args.max_total_steps,
            temperature=args.worker_temperature,
            topk=args.topk,
        )
        rollout_summary = {
            "checkpoint": checkpoint_info,
            "map": args.map,
            "disaster": args.disaster,
            "start_node": start_idx,
            "goal_node": goal_idx,
            "worker": worker_debug,
            "mode": "apte_phase1_worker_only",
        }
        save_json(rollout_dir / "rollout_summary.json", rollout_summary)
        (rollout_dir / "rollout_summary.md").write_text(
            build_apte_rollout_markdown(checkpoint_info, start_idx, goal_idx, worker_debug),
            encoding="utf-8",
        )
        flatten_step_records(worker_debug["worker_steps"]).to_csv(rollout_dir / "worker_steps.csv", index=False)
        plot_rollout_map(
            rollout_dir / "rollout_map.png",
            env,
            start_idx,
            goal_idx,
            [],
            worker_debug["path_nodes"],
            checkpoint_info["display"],
            worker_debug["success"],
            manager_steps=None,
        )
        plot_step_grid(
            rollout_dir / "worker_policy_preview.png",
            worker_debug["worker_steps"],
            "Worker Step Policies (Preview)",
            "Wkr",
            min(args.max_worker_step_plots, 12),
        )
        plot_worker_trace_summary(rollout_dir / "worker_trace_summary.png", worker_debug["worker_steps"])
        print(f"Saved APTE rollout debug outputs to: {rollout_dir}")
        return rollout_dir

    manager_debug = manager_debug_rollout(
        manager=manager,
        env=env,
        start_idx=start_idx,
        goal_idx=goal_idx,
        max_len=args.manager_max_len,
        temperature=args.manager_temperature,
        topk=args.topk,
    )
    worker_debug = worker_debug_rollout(
        worker=worker,
        env=env,
        goal_idx=goal_idx,
        manager_debug=manager_debug,
        max_total_steps=args.max_total_steps,
        temperature=args.worker_temperature,
        topk=args.topk,
    )

    rollout_summary = {
        "checkpoint": checkpoint_info,
        "map": args.map,
        "disaster": args.disaster,
        "start_node": start_idx,
        "goal_node": goal_idx,
        "manager": {
            "plan": manager_debug["plan"],
            "raw_sequence": manager_debug["raw_sequence"],
            "plan_diag": manager_debug["plan_diag"],
            "diagnosis": manager_debug.get("diagnosis", {}),
            "steps": manager_debug["steps"],
        },
        "worker": worker_debug,
    }
    save_json(rollout_dir / "rollout_summary.json", rollout_summary)
    save_json(rollout_dir / "manager_step_flags.json", manager_debug.get("diagnosis", {}))
    (rollout_dir / "rollout_summary.md").write_text(
        build_rollout_markdown(checkpoint_info, start_idx, goal_idx, manager_debug, worker_debug),
        encoding="utf-8",
    )
    flatten_step_records(manager_debug["steps"]).to_csv(rollout_dir / "manager_steps.csv", index=False)
    flatten_step_records(worker_debug["worker_steps"]).to_csv(rollout_dir / "worker_steps.csv", index=False)

    plot_rollout_map(
        rollout_dir / "rollout_map.png",
        env,
        start_idx,
        goal_idx,
        manager_debug["plan"],
        worker_debug["path_nodes"],
        checkpoint_info["display"],
        worker_debug["success"],
        manager_steps=manager_debug["steps"],
    )
    plot_step_grid(rollout_dir / "manager_policy.png", manager_debug["steps"], "Manager Step Policies", "Mgr", args.manager_max_len)
    plot_manager_diagnostic_summary(
        rollout_dir / "manager_step_diagnostics.png",
        manager_debug["steps"],
        manager_debug.get("diagnosis", {}),
    )
    plot_step_grid(rollout_dir / "worker_policy_preview.png", worker_debug["worker_steps"], "Worker Step Policies (Preview)", "Wkr", min(args.max_worker_step_plots, 12))
    plot_worker_trace_summary(rollout_dir / "worker_trace_summary.png", worker_debug["worker_steps"])

    if args.save_step_plots:
        save_step_panels(rollout_dir, manager_debug["steps"], worker_debug["worker_steps"], args.max_worker_step_plots)

    flagged_count = manager_debug.get("diagnosis", {}).get("flagged_count", 0)
    print(f"Saved rollout debug outputs to: {rollout_dir} (flagged manager steps: {flagged_count})")
    return rollout_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="통합 평가/시각화 CLI — 모든 테스트·분석을 하나의 진입점으로 실행한다.",
    )
    subparsers = parser.add_subparsers(dest="command", help="서브커맨드")

    # ── dashboard 서브커맨드 ──
    p_dash = subparsers.add_parser("dashboard", help="학습 로그 대시보드 + 롤아웃 진단")
    p_dash.add_argument("--run-dir", default="logs/rl_phase1_apte", help="학습 로그 디렉토리")
    p_dash.add_argument("--rolling-window", type=int, default=5, help="Rolling mean window")
    p_dash.add_argument("--map", default="Anaheim", help="맵 이름")
    p_dash.add_argument("--disaster", action="store_true", help="재난 모드 활성화")
    p_dash.add_argument("--device", default=None, help="Device override (cpu/cuda)")
    p_dash.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    p_dash.add_argument("--seed", type=int, default=42, help="Random seed")
    p_dash.add_argument("--rollout", action="store_true", help="롤아웃 진단 실행")
    p_dash.add_argument("--checkpoint", default="final", help="체크포인트 지정")
    p_dash.add_argument("--start-node", type=int, default=None, help="고정 시작 노드")
    p_dash.add_argument("--goal-node", type=int, default=None, help="고정 목표 노드")
    p_dash.add_argument("--manager-max-len", type=int, default=20, help="Manager 최대 디코딩 길이")
    p_dash.add_argument("--max-total-steps", type=int, default=400, help="Worker 최대 스텝")
    p_dash.add_argument("--manager-temperature", type=float, default=1e-6, help="Manager 온도")
    p_dash.add_argument("--worker-temperature", type=float, default=1e-6, help="Worker 온도")
    p_dash.add_argument("--topk", type=int, default=8, help="Top-K 후보 수")
    p_dash.add_argument("--save-step-plots", action="store_true", help="스텝별 플롯 저장")
    p_dash.add_argument("--max-worker-step-plots", type=int, default=30, help="Worker 스텝 플롯 최대 수")

    # ── worker 서브커맨드 ──
    p_wkr = subparsers.add_parser("worker", help="Worker 배치 성능 평가 (성공률 + PLR)")
    p_wkr.add_argument("--checkpoint", required=True, help="Worker 체크포인트 경로")
    p_wkr.add_argument("--trials", type=int, default=100, help="평가 문제 수")
    p_wkr.add_argument("--max-steps", type=int, default=400, help="최대 스텝")
    p_wkr.add_argument("--map", default="Anaheim", help="맵 이름")
    p_wkr.add_argument("--seed", type=int, default=42, help="시드")
    p_wkr.add_argument("--temperature", type=float, default=0.0, help="행동 선택 온도 (0=greedy)")

    # ── joint 서브커맨드 ──
    p_jnt = subparsers.add_parser("joint", help="Manager + Worker 조인트 롤아웃")
    p_jnt.add_argument("--checkpoint", required=True, help="체크포인트 경로")
    p_jnt.add_argument("--mgr-checkpoint", default=None, help="Manager 별도 체크포인트")
    p_jnt.add_argument("--start", type=int, default=1, help="시작 노드")
    p_jnt.add_argument("--goal", type=int, default=400, help="목표 노드")
    p_jnt.add_argument("--map", default="Anaheim", help="맵 이름")
    p_jnt.add_argument("--max-steps", type=int, default=400, help="최대 스텝")
    p_jnt.add_argument("--temperature", type=float, default=0.0, help="Worker 온도")
    p_jnt.add_argument("--mgr-temperature", type=float, default=0.5, help="Manager 온도")
    p_jnt.add_argument("--output", default=None, help="출력 이미지 경로")

    # ── compare 서브커맨드 ──
    p_cmp = subparsers.add_parser("compare", help="SL vs RL 모델 비교 평가")
    p_cmp.add_argument("--sl", required=True, help="SL 체크포인트 경로")
    p_cmp.add_argument("--rl", required=True, help="RL 체크포인트 경로")
    p_cmp.add_argument("--episodes", type=int, default=500, help="평가 에피소드 수")
    p_cmp.add_argument("--map", default="Anaheim", help="맵 이름")
    p_cmp.add_argument("--seed", type=int, default=42, help="시드")

    # ── memorization 서브커맨드 ──
    p_mem = subparsers.add_parser("memorization", help="Worker 암기 vs 일반화 진단")
    p_mem.add_argument("--checkpoint", required=True, help="Worker 체크포인트 경로")
    p_mem.add_argument("--trials", type=int, default=100, help="평가 문제 수")
    p_mem.add_argument("--map", default="Anaheim", help="맵 이름")
    p_mem.add_argument("--seed", type=int, default=42, help="시드")

    # ── physics 서브커맨드 ──
    p_phy = subparsers.add_parser("physics", help="재난 물리 시뮬레이션 시각화")
    p_phy.add_argument("--map", default="Anaheim", help="맵 이름")

    # ── paper 서브커맨드 ──
    p_pap = subparsers.add_parser("paper", help="논문용 Figure/Table 자동 생성 (Table 1)")
    p_pap.add_argument("--map", default="Anaheim", help="맵 이름")
    p_pap.add_argument("--eval-episodes", type=int, default=500, help="Table 1 평가 에피소드 수")
    p_pap.add_argument("--output-dir", default="tests/paper_figures", help="출력 디렉토리")
    p_pap.add_argument("--skip-eval", action="store_true", help="Table 1 평가 건너뛰기")
    p_pap.add_argument("--seed", type=int, default=42, help="시드")

    # ── paper_full 서브커맨드 ──
    p_papf = subparsers.add_parser("paper_full", help="논문용 벤치마크 평가, 궤적, 지연시간 등 전체 맵 종합 산출물 생성")
    p_papf.add_argument("--joint-ckpt", required=True, help="Joint RL 체크포인트 경로")
    p_papf.add_argument("--worker-ckpt", required=True, help="Worker-Only RL 체크포인트 경로")
    p_papf.add_argument("--episodes", type=int, default=500, help="배치 평가 에피소드 수")
    p_papf.add_argument("--num-samples", type=int, default=16, help="Multi-Sampling 번호")
    p_papf.add_argument("--seed", type=int, default=42, help="시드")

    # ── regen 서브커맨드 ──
    p_reg = subparsers.add_parser("regen", help="긴 OD 쌍에 대한 HRL vs Flat RL 최적 궤적 재시각화")
    p_reg.add_argument("--joint-ckpt", required=True, help="Joint RL 체크포인트 경로")
    p_reg.add_argument("--worker-ckpt", required=True, help="Worker-Only RL 체크포인트 경로")
    p_reg.add_argument("--map", default="Goldcoast", help="대상 맵 이름")
    p_reg.add_argument("--candidates", type=int, default=10, help="탐색할 후보의 수")
    p_reg.add_argument("--num-samples", type=int, default=16, help="HRL 롤아웃 시 샘플링 수")
    p_reg.add_argument("--seed", type=int, default=42, help="시드")

    return parser.parse_args()


# ============================================================
# dashboard 서브커맨드 (기존 visualize_result.py main)
# ============================================================

def cmd_dashboard(args: argparse.Namespace) -> None:
    """학습 로그 대시보드 + 선택적 롤아웃 진단."""
    set_seed(args.seed)

    run_dir = Path(args.run_dir)
    dashboard_dir = run_dir / "eval_dashboard"
    dashboard_dir.mkdir(parents=True, exist_ok=True)

    runtime_config = load_runtime_config(run_dir)
    apte_mode = is_apte_phase1_run(run_dir, runtime_config)
    df = load_metrics_frame(run_dir, runtime_config)
    checkpoint_episodes = discover_checkpoint_episodes(run_dir)
    summary = build_apte_summary(df) if apte_mode else build_summary(df, checkpoint_episodes)

    df.to_csv(dashboard_dir / "parsed_debug_metrics.csv", index=False)
    save_json(dashboard_dir / "summary.json", summary)
    if apte_mode:
        save_apte_summary_markdown(summary, dashboard_dir / "summary.md")
        plot_apte_training_dashboard(df, dashboard_dir / "training_dashboard.png", args.rolling_window)
    else:
        save_summary_markdown(summary, dashboard_dir / "summary.md")
        plot_training_dashboard(df, dashboard_dir / "training_dashboard.png", args.rolling_window)
        plot_relationships(df, dashboard_dir / "metric_relationships.png")
        plot_snapshot_comparison(df, dashboard_dir / "snapshot_comparison.png")

    samples_path = dashboard_dir / "episode_samples_preview.json"
    if samples_path.exists():
        maybe_load_episode_samples(samples_path)

    if apte_mode:
        print_apte_terminal_summary(summary)
    else:
        print_terminal_summary(summary)
    print(f"Saved parsed metrics to: {dashboard_dir / 'parsed_debug_metrics.csv'}")
    print(f"Saved dashboard outputs to: {dashboard_dir}")

    if args.rollout:
        checkpoint_info = resolve_checkpoint_spec(run_dir, args.checkpoint, summary, checkpoint_episodes)
        run_rollout_visualization(run_dir, dashboard_dir, checkpoint_info, args)


# ============================================================
# worker 서브커맨드 (기존 eval_worker_batch.py)
# ============================================================

def cmd_worker(args: argparse.Namespace) -> None:
    """Worker 배치 성능 평가."""
    import tests.eval_core as core

    core.set_seed(args.seed)
    device = core.get_device()
    env = core.setup_env(args.map, device)
    worker, _, _ = core.load_checkpoint(args.checkpoint, device=device)

    # 유효 OD 쌍 생성
    pairs = core.generate_od_pairs(env, args.trials, min_hops=3, seed=args.seed)
    print(f"\n{'='*60}")
    print(f"Worker 배치 평가: {len(pairs)}개 문제, max_steps={args.max_steps}")
    print(f"체크포인트: {args.checkpoint}")
    print(f"{'='*60}")

    results = []
    successes = 0

    for idx, (s, g) in enumerate(pairs):
        result = core.run_worker_rollout(worker, env, s, g, max_steps=args.max_steps, temperature=args.temperature)
        results.append(result)
        if result["success"]:
            successes += 1
        if (idx + 1) % 10 == 0:
            sr = successes / (idx + 1) * 100
            print(f"  [{idx + 1}/{len(pairs)}] 현재 성공률: {sr:.1f}%")

    # 통계
    total = len(results)
    success_rate = successes / total * 100.0
    opt_ratios = [r["path_length_ratio"] for r in results if r["success"]]

    # 난이도별 분석
    easy = [r for r in results if r["optimal_hops"] <= 5]
    medium = [r for r in results if 5 < r["optimal_hops"] <= 15]
    hard = [r for r in results if r["optimal_hops"] > 15]

    print(f"\n{'='*60}")
    print(f"[최종 결과] Worker 배치 평가 ({total}개 문제)")
    print(f"{'='*60}")
    print(f"\n📊 성공률: {success_rate:.1f}% ({successes}/{total})")
    print(f"\n📏 난이도별 성공률:")
    for label, subset in [("쉬움 ≤5홉", easy), ("보통 6~15홉", medium), ("어려움 >15홉", hard)]:
        if subset:
            sr = sum(1 for r in subset if r["success"]) / len(subset) * 100
            print(f"   {label}: {sr:.1f}% ({sum(1 for r in subset if r['success'])}/{len(subset)}건)")
    if opt_ratios:
        print(f"\n🎯 경로 품질 (성공 건만):")
        print(f"   PLR 평균: {np.mean(opt_ratios):.2f}x (1.0 = 최적)")
        print(f"   PLR 중앙값: {np.median(opt_ratios):.2f}x")


# ============================================================
# joint 서브커맨드 (기존 eval_joint_rollout.py)
# ============================================================

def cmd_joint(args: argparse.Namespace) -> None:
    """Manager + Worker 조인트 롤아웃."""
    import tests.eval_core as core

    device = core.get_device()
    env = core.setup_env(args.map, device)
    worker, manager, _ = core.load_checkpoint(
        args.checkpoint, device=device, load_manager=True,
    )
    # 별도 Manager 체크포인트 처리
    if args.mgr_checkpoint and manager is None:
        from src.models.manager import GraphTransformerManager
        manager = GraphTransformerManager(node_dim=4, hidden_dim=256, edge_dim=1).to(device)
        mgr_payload = torch.load(args.mgr_checkpoint, map_location=device, weights_only=False)
        if "manager_state" in mgr_payload:
            manager.load_state_dict(mgr_payload["manager_state"])
        else:
            manager.load_state_dict(mgr_payload)
        manager.eval()
        print(f"📦 Manager 별도 로드: {args.mgr_checkpoint}")

    if manager is None:
        print("❌ Manager를 로드할 수 없습니다. 체크포인트에 manager_state가 필요합니다.")
        return

    result = core.run_joint_rollout(
        worker, manager, env, args.start, args.goal,
        max_steps=args.max_steps,
        temperature=args.temperature,
        mgr_temperature=args.mgr_temperature,
    )

    print(f"\n{'='*60}")
    print(f"Joint 롤아웃: {args.start} → {args.goal}")
    print(f"{'='*60}")
    print(f"  성공: {'✅' if result['success'] else '❌'}")
    print(f"  경로 길이: {result['actual_steps']} (최적: {result['optimal_hops']:.0f})")
    print(f"  Plan: {result['plan_subgoals']}")
    print(f"  Subgoal 도달: {len(result['subgoal_reached_log'])}/{len(result['plan_subgoals'])}")

    # 시각화
    output_path = Path(args.output or "tests/paper_figures/joint_rollout_map.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_rollout_map(
        output_path=output_path, env=env,
        start_idx=result["start"], goal_idx=result["goal"],
        manager_plan=result["plan_subgoals"],
        worker_path=result["path"],
        checkpoint_label="Joint Model",
        success=result["success"],
    )
    print(f"💾 저장됨: {output_path}")


# ============================================================
# compare 서브커맨드 (기존 compare_models.py)
# ============================================================

def cmd_compare(args: argparse.Namespace) -> None:
    """SL vs RL 모델 비교 평가."""
    import tests.eval_core as core

    core.set_seed(args.seed)
    device = core.get_device()
    env = core.setup_env(args.map, device)

    print(f"\n{'='*60}")
    print(f"🚀 SL vs RL 모델 비교 ({args.map}, {args.episodes} 에피소드)")
    print(f"{'='*60}")

    # SL 모델
    print(f"\n📊 SL 모델 평가 중... ({args.sl})")
    worker_sl, _, _ = core.load_checkpoint(args.sl, device=device)
    sl_results = core.evaluate_worker_batch(worker_sl, env, args.episodes, label="SL Pre-train", seed=args.seed)

    # RL 모델
    print(f"\n📊 RL 모델 평가 중... ({args.rl})")
    worker_rl, _, _ = core.load_checkpoint(args.rl, device=device)
    rl_results = core.evaluate_worker_batch(worker_rl, env, args.episodes, label="RL (Ours)", seed=args.seed)

    print(f"\n{'='*60}")
    print(f"📊 비교 결과 ({args.episodes} 에피소드)")
    print(f"{'='*60}")
    print(f"                      | SL Pre-train    | RL (Ours)           |")
    print(f"----------------------|-----------------|---------------------|")
    print(f" 🎯 Success Rate      | {sl_results['success_rate']:>14.1f}% | {rl_results['success_rate']:>18.1f}% |")
    print(f" 📏 PLR               | {sl_results['path_length_ratio']:>14.3f}x | {rl_results['path_length_ratio']:>18.3f}x |")
    print(f" ⏱️ Latency           | {sl_results['inference_latency_ms']:>13.2f}ms | {rl_results['inference_latency_ms']:>17.2f}ms |")
    print(f"{'='*60}")


# ============================================================
# memorization 서브커맨드 (기존 eval_worker_memorization.py)
# ============================================================

def cmd_memorization(args: argparse.Namespace) -> None:
    """Worker 암기 vs 일반화 진단."""
    import tests.eval_core as core

    core.set_seed(args.seed)
    device = core.get_device()
    env = core.setup_env(args.map, device)
    worker, _, _ = core.load_checkpoint(args.checkpoint, device=device)

    pairs = core.generate_od_pairs(env, args.trials, min_hops=3, seed=args.seed)
    print(f"\n{'='*65}")
    print(f" Worker 암기 vs 일반화 진단 ({len(pairs)}개 문제)")
    print(f"{'='*65}")

    all_results = []
    exact_match_count = 0

    for idx, (s, g) in enumerate(pairs):
        # A* 최적 경로
        optimal_path = env.reconstruct_weighted_shortest_path_indices(s, g)
        # Worker 경로
        rollout = core.run_worker_rollout(worker, env, s, g)
        worker_path = rollout["path_nodes"]
        overlap = core.compute_path_overlap(worker_path, optimal_path)
        is_exact = (worker_path == optimal_path)
        if is_exact:
            exact_match_count += 1
        all_results.append({
            **overlap,
            "success": rollout["success"],
            "exact_match": is_exact,
            "optimal_hops": rollout["optimal_hops"],
        })
        if (idx + 1) % 25 == 0:
            print(f"  [{idx + 1}/{len(pairs)}] 진행 중...")

    # 분석
    coverages = [r["node_coverage"] for r in all_results]
    edge_overlaps = [r["edge_overlap"] for r in all_results]

    print(f"\n{'='*65}")
    print(f" [결과] 경로 일치율 분석")
    print(f"{'='*65}")
    print(f"\n📌 완전 동일 경로: {exact_match_count}/{len(pairs)} ({exact_match_count / len(pairs) * 100:.1f}%)")
    print(f"\n📊 노드 수준 일치율:")
    print(f"   Coverage: {np.mean(coverages) * 100:.1f}% ± {np.std(coverages) * 100:.1f}%")
    print(f"\n📊 간선 수준 일치율:")
    print(f"   Edge Overlap: {np.mean(edge_overlaps) * 100:.1f}% ± {np.std(edge_overlaps) * 100:.1f}%")

    # 분포
    high_eo = sum(1 for e in edge_overlaps if e >= 0.9)
    mid_eo = sum(1 for e in edge_overlaps if 0.5 <= e < 0.9)
    low_eo = sum(1 for e in edge_overlaps if e < 0.5)
    print(f"\n📊 간선 일치율 분포:")
    print(f"   ≥90% (거의 암기): {high_eo}/{len(pairs)}")
    print(f"   50~89% (부분 일치): {mid_eo}/{len(pairs)}")
    print(f"   <50% (다른 경로): {low_eo}/{len(pairs)}")

    mean_eo = np.mean(edge_overlaps)
    print(f"\n{'='*65}")
    print(f" [최종 판단]")
    print(f"{'='*65}")
    if mean_eo >= 0.85:
        print(f"   ⚠️  평균 간선 일치율 {mean_eo * 100:.1f}% → 암기 가능성 높음")
    elif mean_eo >= 0.6:
        print(f"   🔶 평균 간선 일치율 {mean_eo * 100:.1f}% → 부분 암기 + 일반화 혼합")
    else:
        print(f"   ✅ 평균 간선 일치율 {mean_eo * 100:.1f}% → 일반화된 정책")


# ============================================================
# physics 서브커맨드 (기존 visualize_physics.py)
# ============================================================

def cmd_physics(args: argparse.Namespace) -> None:
    """재난 물리 시뮬레이션 시각화."""
    import csv
    import os
    import networkx as nx

    map_name = args.map
    print(f"🧪 Starting Physics Engine Verification for {map_name}...")

    base_dir = str(PROJECT_ROOT)
    node_file = os.path.join(base_dir, "data", f"{map_name}_node.tntp")
    net_file = os.path.join(base_dir, "data", f"{map_name}_net.tntp")

    if not os.path.exists(node_file):
        print(f"❌ File not found: {node_file}")
        return

    # 출력 디렉토리
    base_output_dir = os.path.join(base_dir, "tests", "physics", map_name)
    log_dir = os.path.join(base_output_dir, "logs")
    img_dir = os.path.join(base_output_dir, "images")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    env = DisasterEnv(node_file, net_file)
    env.reset(batch_size=1)

    # 고가도로 검증 맵
    _physics_plot_elevated(env, img_dir)

    # Main Shock
    print(f"⚠️ Triggering Main Shock...")
    env.reset(batch_size=1)
    main_score = env.last_score if hasattr(env, 'last_score') else torch.zeros(1, env.num_physical_edges)
    _physics_plot_damage_map(env, main_score[0].cpu().numpy(), img_dir, "Shock_0_Intensity")
    _physics_plot_state_map(env, env.damage_states[0].cpu().numpy(), img_dir, "Shock_0_Damage_Actual")
    _physics_save_log(env, main_score, log_dir, "Shock_0_Damage_Actual", env.damage_states, "Shock_0_Actual")

    if hasattr(env, 'damage_states_theoretical'):
        theo = env.damage_states_theoretical[0].cpu().numpy()
        _physics_plot_state_map(env, theo, img_dir, "Shock_0_Damage_Theoretical")

    cumulative_pga = main_score[0].cpu().numpy().copy()

    # Aftershocks
    print("⚠️ Simulating Seismic Schedule...")
    sorted_steps = sorted([t for t in env.seismic_schedule if t > 0])
    for i, t in enumerate(sorted_steps):
        params = env.seismic_schedule[t]
        env.step_count[:] = t - 1
        env.step(torch.zeros(1, dtype=torch.long, device=env.device))
        print(f"   [Event #{i + 1} @ t={t}] {params['type'].upper()}")

        as_pga = env.last_score if hasattr(env, 'last_score') else torch.zeros(1, env.num_physical_edges)
        pga_after = as_pga[0].cpu().numpy()
        _physics_plot_damage_map(env, pga_after, img_dir, f"Shock_{i + 1}_Intensity")
        cumulative_pga = np.maximum(cumulative_pga, pga_after)
        _physics_plot_damage_map(env, cumulative_pga, img_dir, f"Shock_{i + 1}_Cumulative_Intensity_HAZUS")
        _physics_plot_state_map(env, env.damage_states[0].cpu().numpy(), img_dir, f"Shock_{i + 1}_Cumulative_State_Actual")
        _physics_save_log(env, as_pga, log_dir, f"Shock_{i + 1}_Intensity", None, f"Shock_{i + 1}")

    print(f"✅ Physics Visualization Complete.")


def _physics_plot_elevated(env: DisasterEnv, output_dir: str) -> None:
    """고가도로 검증 맵."""
    import networkx as nx
    G = env.map_core.graph
    pos = env.map_core.pos
    fig = plt.figure(figsize=(20, 16))
    nx.draw_networkx_nodes(G, pos, node_size=20, node_color='#eeeeee', edgecolors='grey', alpha=0.5)
    edge_list = list(G.edges())
    use_hw = hasattr(env, 'edge_is_highways')
    if not use_hw and not hasattr(env, 'edge_speeds'):
        print("❌ No edge attributes.")
        plt.close(fig)
        return
    values = env.edge_is_highways.cpu().numpy() if use_hw else env.edge_speeds.cpu().numpy()
    threshold = 0.5 if use_hw else 4842
    elevated = [(u, v) for i, (u, v) in enumerate(edge_list) if i < len(values) and values[i] >= threshold]
    normal = [(u, v) for i, (u, v) in enumerate(edge_list) if i < len(values) and values[i] < threshold]
    nx.draw_networkx_edges(G, pos, edgelist=normal, width=1.5, edge_color='#cccccc', alpha=0.3)
    nx.draw_networkx_edges(G, pos, edgelist=elevated, width=4.0, edge_color='cyan', alpha=1.0)
    plt.title("Elevated Road Verification", fontsize=20, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    save_path = os.path.join(output_dir, "Elevated_Road_Verification.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"📸 Saved: {save_path}")


def _physics_plot_damage_map(env: DisasterEnv, pga_data: np.ndarray, output_dir: str, title: str) -> None:
    """HAZUS 스케일 손상 강도 맵."""
    import os
    G = env.map_core.graph
    pos = env.map_core.pos
    fig = plt.figure(figsize=(20, 16))
    import networkx as nx
    nx.draw_networkx_nodes(G, pos, node_size=30, node_color='#dddddd', edgecolors='grey')
    edge_list = list(G.edges())
    categories = {k: [] for k in ["normal", "slight", "moderate", "extensive", "complete"]}
    for i, (u, v) in enumerate(edge_list):
        if i >= len(pga_data):
            break
        val = pga_data[i]
        if val < 2.0:
            categories["normal"].append((u, v))
        elif val < 3.0:
            categories["slight"].append((u, v))
        elif val < 4.0:
            categories["moderate"].append((u, v))
        elif val < 5.0:
            categories["extensive"].append((u, v))
        else:
            categories["complete"].append((u, v))
    colors = {"normal": '#cccccc', "slight": '#FFEB3B', "moderate": '#FF9800', "extensive": '#FF5722', "complete": '#500000'}
    widths = {"normal": 1.5, "slight": 2.0, "moderate": 3.0, "extensive": 4.0, "complete": 5.0}
    for cat, edges in categories.items():
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=widths[cat], edge_color=colors[cat], alpha=0.7 if cat != "normal" else 0.4)
    plt.title(f"{title} (HAZUS Scale)", fontsize=20, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{title}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"📸 Saved: {save_path}")


def _physics_plot_state_map(env: DisasterEnv, state_data: np.ndarray, output_dir: str, title: str) -> None:
    """이산 손상 상태 맵."""
    import os
    G = env.map_core.graph
    pos = env.map_core.pos
    fig = plt.figure(figsize=(20, 16))
    import networkx as nx
    nx.draw_networkx_nodes(G, pos, node_size=30, node_color='#dddddd', edgecolors='grey')
    edge_list = list(G.edges())
    state_arr = state_data.flatten()
    normal = [(u, v) for i, (u, v) in enumerate(edge_list) if i < len(state_arr) and int(state_arr[i]) == 0]
    damaged = [(u, v) for i, (u, v) in enumerate(edge_list) if i < len(state_arr) and int(state_arr[i]) != 0]
    nx.draw_networkx_edges(G, pos, edgelist=normal, width=1.5, edge_color='#cccccc', alpha=0.4)
    nx.draw_networkx_edges(G, pos, edgelist=damaged, width=4.0, edge_color='#D32F2F', alpha=1.0)
    plt.title(f"{title} (State)", fontsize=20, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{title}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"📸 Saved: {save_path}")


def _physics_save_log(env: DisasterEnv, pga_tensor, log_dir: str, event_name: str, state_tensor, event_label: str) -> None:
    """손상 로그 CSV 저장."""
    import csv, os
    log_path = os.path.join(log_dir, f"{event_name}_Log.csv")
    G = env.map_core.graph
    edge_list = list(G.edges(data=True))
    speeds = env.edge_speeds.cpu().numpy() if hasattr(env, 'edge_speeds') else [0.0] * len(edge_list)
    is_highways = env.edge_is_highways.cpu().numpy() if hasattr(env, 'edge_is_highways') else [0.0] * len(edge_list)
    pga_vals = pga_tensor[0].cpu().numpy() if isinstance(pga_tensor, torch.Tensor) and len(pga_tensor.shape) > 1 else (pga_tensor.cpu().numpy() if isinstance(pga_tensor, torch.Tensor) else pga_tensor)
    state_vals = None
    if state_tensor is not None:
        state_vals = state_tensor[0].cpu().numpy() if isinstance(state_tensor, torch.Tensor) and len(state_tensor.shape) > 1 else (state_tensor.cpu().numpy() if isinstance(state_tensor, torch.Tensor) else state_tensor)

    with open(log_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["# Event_Label", event_label])
        writer.writerow([])
        writer.writerow(["Edge_Index", "U_Node", "V_Node", "Speed", "Is_Highway", "Intensity_Score", "HAZUS_Scale", "Damage_Desc", "Damage_Code"])
        for i, (u, v, d) in enumerate(edge_list):
            if i >= len(pga_vals):
                break
            pga = pga_vals[i]
            hazus = "Normal" if pga < 2.0 else ("Slight" if pga < 3.0 else ("Moderate" if pga < 4.0 else ("Extensive" if pga < 5.0 else "Complete")))
            code = int(state_vals[i]) if state_vals is not None and i < len(state_vals) else -1
            desc = "Normal" if code == 0 else "Damaged" if code > 0 else "Unknown"
            writer.writerow([i, u, v, speeds[i] if i < len(speeds) else 0, int(is_highways[i]) if i < len(is_highways) else 0, f"{pga:.4f}", hazus, desc, code])
    print(f"📝 Saved: {log_path}")


# ============================================================
# paper 서브커맨드 (기존 generate_paper_figures.py)
# ============================================================

def cmd_paper(args: argparse.Namespace) -> None:
    """논문용 Figure/Table 자동 생성."""
    import tests.eval_core as core

    core.set_seed(args.seed)
    device = core.get_device()
    env = core.setup_env(args.map, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 체크포인트 로드: 새 파이프라인 (joint > worker > legacy phase1) 순서 탐색
    # Worker 체크포인트 탐색: joint > worker_stage > phase1_apte(레거시)
    joint_ckpt = core.find_latest_checkpoint("logs/rl_joint_stage", "final.pt")
    # [Fix] Joint 학습 미완주 시 best.pt로 폴백
    if joint_ckpt is None:
        # 특정 런 우선 사용 (EMA 25.6% 달성)
        preferred = Path("logs/rl_joint_stage/2026-04-14_0315_joint_pomo32/best.pt")
        if preferred.exists():
            joint_ckpt = preferred
        else:
            joint_ckpt = core.find_latest_checkpoint("logs/rl_joint_stage", "best.pt")
    worker_ckpt = core.find_latest_checkpoint("logs/rl_worker_stage", "final.pt")
    legacy_ckpt = core.find_latest_checkpoint("logs/rl_phase1_apte", "final.pt")
    sl_ckpt = core.find_sl_checkpoint()

    # Worker 로드: joint(둘 다 포함) > worker_stage > phase1(레거시) > SL
    hrl_ckpt = joint_ckpt or worker_ckpt or legacy_ckpt
    if not hrl_ckpt:
        print("❌ Worker 체크포인트를 찾을 수 없습니다.")
        return

    worker_hrl, manager_hrl, _ = core.load_checkpoint(str(hrl_ckpt), device=device, load_manager=True)
    print(f"📦 Worker loaded from: {hrl_ckpt}")

    # Manager 로드: 체크포인트에 없으면 manager_stage > SL pretrain fallback
    if manager_hrl is None:
        mgr_ckpt = core.find_latest_checkpoint("logs/rl_manager_stage", "final.pt")
        from src.models.manager import GraphTransformerManager
        manager_hrl = GraphTransformerManager(node_dim=4, hidden_dim=256, edge_dim=1).to(device)

        mgr_loaded = False
        # 1순위: manager_stage 체크포인트
        if mgr_ckpt:
            try:
                import torch
                mgr_payload = torch.load(str(mgr_ckpt), map_location=device, weights_only=False)
                if "manager_state" in mgr_payload:
                    manager_hrl.load_state_dict(mgr_payload["manager_state"])
                    mgr_loaded = True
                    print(f"📦 Manager loaded from: {mgr_ckpt}")
            except Exception as e:
                print(f"⚠️ Manager stage 로드 실패: {e}")

        # 2순위: SL pretrain 체크포인트
        if not mgr_loaded and sl_ckpt:
            try:
                import torch
                sl_payload = torch.load(str(sl_ckpt), map_location=device, weights_only=False)
                if "manager_state" in sl_payload:
                    manager_hrl.load_state_dict(sl_payload["manager_state"])
                    mgr_loaded = True
                    print(f"📦 Manager loaded from SL: {sl_ckpt}")
            except Exception as e:
                print(f"⚠️ SL Manager 로드 실패: {e}")

        if not mgr_loaded:
            print("⚠️ Manager 체크포인트를 찾을 수 없어 초기화 상태를 사용합니다.")
        manager_hrl.eval()

    worker_sl = None
    if sl_ckpt:
        worker_sl, _, _ = core.load_checkpoint(str(sl_ckpt), device=device)
        print(f"📦 SL: {sl_ckpt}")

    print(f"\n{'='*60}")
    print("📊 논문용 Figure/Table 생성 시작")
    print(f"{'='*60}")

    # Figure 1: Concept Diagram — eval_core의 rollout 사용
    print("📊 Figure 1: Concept Diagram 생성 중...")
    _paper_fig1_concept(env, worker_hrl, output_dir / "fig1_concept_diagram.png")

    # Figure 4 + Table 1: 멀티맵 Scalability 비교
    # 사용 가능한 맵을 노드 수 오름차순으로 정렬하여 순회
    multi_map_names = ["SiouxFalls", "Anaheim", "ChicagoSketch", "Goldcoast"]
    maps_config = []  # Figure 4용
    multi_table_rows = []  # Table 1용 (멀티맵)

    for mname in multi_map_names:
        try:
            m_env = core.setup_env(mname, device)
        except (FileNotFoundError, Exception) as e:
            print(f"  ⚠️ {mname} 맵 로드 실패, 건너뜀: {e}")
            continue
        print(f"\n📊 [{mname}] ({m_env.num_nodes} nodes) 평가 중...")

        # A* 벤치마크
        m_astar = core.benchmark_astar(m_env, num_queries=100)

        # RL (Worker Only) 평가
        m_worker_eval = core.evaluate_worker_batch(
            worker_hrl, m_env, args.eval_episodes,
            label=f"RL-{mname}", seed=args.seed,
        )

        # Proposed HRL (Worker + Manager) 평가 - Greedy
        m_hrl_eval_greedy = core.evaluate_joint_batch(
            worker_hrl, manager_hrl, m_env, args.eval_episodes,
            label=f"HRL(Greedy)-{mname}", seed=args.seed,
            num_manager_samples=1,
        )

        # Proposed HRL (Worker + Manager) 평가 - POMO
        m_hrl_eval_pomo = core.evaluate_joint_batch(
            worker_hrl, manager_hrl, m_env, args.eval_episodes,
            label=f"HRL(POMO)-{mname}", seed=args.seed,
            num_manager_samples=32,
        )

        # Figure 4 데이터 수집
        maps_config.append({
            "name": mname, "nodes": m_env.num_nodes,
            "astar_ms": m_astar["mean_ms"],
            "hrl_ms": m_hrl_eval_greedy["inference_latency_ms"],
        })

        # Table 1 데이터 수집
        multi_table_rows.append({
            "map": mname, "nodes": m_env.num_nodes,
            "astar_lat": m_astar["mean_ms"],
            "worker_sr": m_worker_eval["success_rate"],
            "worker_plr": m_worker_eval["path_length_ratio"],
            "worker_lat": m_worker_eval["inference_latency_ms"],
            "worker_per_step": m_worker_eval["per_step_latency_ms"],
            "hrl_greedy_sr": m_hrl_eval_greedy["success_rate"],
            "hrl_greedy_plr": m_hrl_eval_greedy["path_length_ratio"],
            "hrl_greedy_lat": m_hrl_eval_greedy["inference_latency_ms"],
            "hrl_greedy_per_step": m_hrl_eval_greedy["per_step_latency_ms"],
            "hrl_pomo_sr": m_hrl_eval_pomo["success_rate"],
            "hrl_pomo_plr": m_hrl_eval_pomo["path_length_ratio"],
            "hrl_pomo_lat": m_hrl_eval_pomo["inference_latency_ms"],
            "hrl_pomo_per_step": m_hrl_eval_pomo["per_step_latency_ms"],
        })

    # Figure 4 생성
    _paper_fig4_latency(maps_config, output_dir / "fig4_inference_latency.png")

    # Table 1: 멀티맵 비교표 생성
    if not args.skip_eval and multi_table_rows:
        _paper_table1_multimap(multi_table_rows, output_dir / "table1_performance.tex")

    # Figure 5a: Learning Curves
    print("📊 Figure 5a: Learning Curves 생성 중...")
    _paper_fig5a_curves(output_dir / "fig5a_learning_curves.png")

    # Figure 5b: Path Visualization
    print("📊 Figure 5b: Path Visualization 생성 중...")
    _paper_fig5b_path_viz(env, worker_hrl, output_dir / "fig5b_path_visualization.png")

    print(f"\n{'='*60}")
    print(f"✅ 논문 자료 생성 완료! → {output_dir}")
    print(f"{'='*60}")


def _paper_fig1_concept(env: DisasterEnv, worker, output_path: Path) -> None:
    """Figure 1: Macro-Micro 개념도."""
    import tests.eval_core as core
    core.set_seed(123)
    positions = env.pos_tensor.detach().cpu().numpy()
    best_pair, best_hops = None, 0
    for _ in range(50):
        s = random.randint(0, env.num_nodes - 1)
        g = random.randint(0, env.num_nodes - 1)
        h = float(env.hop_matrix[s, g].item())
        if math.isfinite(h) and 10 <= h <= 25 and h > best_hops:
            best_hops = h
            best_pair = (s, g)
    if best_pair is None:
        best_pair = (0, min(env.num_nodes - 1, 100))
    start_idx, goal_idx = best_pair
    astar_path = env.reconstruct_hop_shortest_path_indices(start_idx, goal_idx)
    if not astar_path or len(astar_path) < 3:
        astar_path = [start_idx, goal_idx]
    subgoals = [astar_path[int(len(astar_path) * p)] for p in [0.25, 0.5, 0.75]]
    rollout = core.run_worker_rollout(worker, env, start_idx, goal_idx)
    worker_path = rollout["path_nodes"]
    
    metrics_text = f"OD: {start_idx} -> {goal_idx} | A* Hops: {len(astar_path)-1} | Worker Steps: {rollout.get('actual_steps', len(worker_path)-1)} | PLR: {rollout.get('path_length_ratio', 1.0):.2f} | {'SUCCESS' if rollout.get('success', True) else 'FAILED'}"
    with open(output_path.with_suffix('.txt'), "w", encoding="utf-8") as f:
        f.write(metrics_text)
    print(f"  📝 Saved Metrics: {output_path.with_suffix('.txt')}")

    fig, ax = plt.subplots(figsize=(14, 11), constrained_layout=True)
    for src, dst in env.map_core.graph.edges():
        u, v = env.node_mapping[src], env.node_mapping[dst]
        ax.plot([positions[u, 0], positions[v, 0]], [positions[u, 1], positions[v, 1]], color="#6b7280", linewidth=1.0, alpha=0.8)
    plan_arr = np.array([start_idx] + subgoals + [goal_idx])
    ax.plot(positions[plan_arr, 0], positions[plan_arr, 1], "--", color="#f97316", linewidth=2.5, alpha=0.9, label="Manager Sparse Plan")
    if len(worker_path) > 1:
        import matplotlib.colors as mcolors
        import matplotlib.cm as cm
        cmap = plt.get_cmap("cool")
        norm = mcolors.Normalize(vmin=0, vmax=max(len(worker_path) - 2, 1))
        for i in range(len(worker_path) - 1):
            ax.plot([positions[worker_path[i], 0], positions[worker_path[i + 1], 0]], [positions[worker_path[i], 1], positions[worker_path[i + 1], 1]], color=cmap(norm(i)), linewidth=3.0, alpha=0.9)
        
        # Colorbar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.046, pad=0.05, aspect=40)
        cbar.set_ticks([0, max(len(worker_path) - 2, 1)])
        cbar.set_ticklabels(['Start', 'Goal'], fontsize=14)
        cbar.set_label('Worker Progress (Local Path Step)', fontsize=16)
    sg_arr = np.array(subgoals)
    ax.scatter(positions[sg_arr, 0], positions[sg_arr, 1], marker="*", s=500, facecolors="#fbbf24", edgecolors="#92400e", linewidths=1.5, label="Subgoals")
    for idx, (x, y) in enumerate(zip(positions[sg_arr, 0], positions[sg_arr, 1])):
        ax.text(x + 0.0015, y + 0.0015, f"$S_{idx+1}$", fontsize=13, fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='#fbbf24', boxstyle='round,pad=0.3', alpha=0.9))
    
    ax.scatter(positions[start_idx, 0], positions[start_idx, 1], marker="o", s=250, color="#22c55e", edgecolors="black", linewidths=1.5, label="Start")
    ax.scatter(positions[goal_idx, 0], positions[goal_idx, 1], marker="X", s=280, color="#ef4444", edgecolors="black", linewidths=1.5, label="Goal")
    # Title removed per user request
    ax.legend(fontsize=14)
    ax.tick_params(axis='both', labelsize=13)
    ax.set_aspect("equal")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved: {output_path}")


def _paper_fig4_latency(maps_config, output_path: Path) -> None:
    """Figure 4: Inference Latency."""
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    names = [m["name"] for m in maps_config]
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, [m["astar_ms"] for m in maps_config], width, label="A* (Dijkstra)", color="#ef4444", alpha=0.85)
    ax.bar(x + width / 2, [m["hrl_ms"] for m in maps_config], width, label="Proposed HRL", color="#3b82f6", alpha=0.85)
    ax.set_xlabel("Road Network", fontsize=16, fontweight="bold")
    ax.set_ylabel("Latency (ms)", fontsize=16, fontweight="bold")
    # Title removed per user request
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n({m['nodes']})" for n, m in zip(names, maps_config)], fontsize=14)
    ax.legend(fontsize=14)
    ax.tick_params(axis='y', labelsize=13)
    ax.set_ylim(bottom=0)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved: {output_path}")


def _paper_table1(models, output_path: Path, map_name: str, num_nodes: int) -> None:
    """Table 1: LaTeX 성능 비교표."""
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        f"  \\caption{{Performance Comparison on {map_name} ({num_nodes} nodes)}}",
        r"  \label{tab:performance}",
        r"  \begin{tabular}{lccc}",
        r"    \toprule",
        r"    \textbf{Method} & \textbf{SR (\%)} & \textbf{PLR} & \textbf{Latency (ms)} \\",
        r"    \midrule",
    ]
    for m in models:
        plr_str = "1.000 (Optimal)" if m["plr"] == 1.0 else f"{m['plr']:.3f}"
        lines.append(f"    {m['label']} & {m['sr']:.1f} & {plr_str} & {m['lat']:.2f} \\\\")
    lines.extend([r"    \bottomrule", r"  \end{tabular}", r"\end{table}"])
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✅ Saved: {output_path}")
    print("\n--- Table 1 Preview ---")
    print("\n".join(lines))


def _paper_table1_multimap(rows: list, output_path: Path) -> None:
    """Table 1: 멀티맵 Scalability 비교 LaTeX 표 생성.

    각 맵에 대해 A* / RL(Worker Only) / HRL(Worker+Manager) 3가지 방법을 비교한다.
    Per-Decision Latency(재경로 1회 비용)와 Total Path Latency(전체 경로 완성 비용) 두 가지를 모두 기재한다.
    """
    lines = [
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \caption{Performance Comparison Across Multiple Road Networks}",
        r"  \label{tab:scalability}",
        r"  \begin{tabular}{l r l c c r r}",
        r"    \toprule",
        r"    \textbf{Network} & \textbf{Nodes} & \textbf{Method} & \textbf{SR (\%)} & \textbf{PLR} & \textbf{Per-Decision (ms)} & \textbf{Total Path (ms)} \\",
        r"    \midrule",
    ]
    for i, row in enumerate(rows):
        map_label = f"{row['map']}"
        nodes_str = f"{row['nodes']}"
        # A* Expert: per-decision = total (단일 호출로 전체 경로 산출)
        lines.append(f"    \\multirow{{4}}{{*}}{{{map_label}}} & \\multirow{{4}}{{*}}{{{nodes_str}}} & A* Expert & 100.0 & 1.000 & {row['astar_lat']:.2f} & {row['astar_lat']:.2f} \\\\")
        # RL (Worker Only): per-decision = 1스텝 평균, total = 전체 경로 합계
        w_plr = f"{row['worker_plr']:.3f}" if math.isfinite(row['worker_plr']) else "N/A"
        lines.append(f"    & & RL (Worker Only) & {row['worker_sr']:.1f} & {w_plr} & {row['worker_per_step']:.2f} & {row['worker_lat']:.2f} \\\\")
        # HRL (Greedy):
        hg_plr = f"{row['hrl_greedy_plr']:.3f}" if math.isfinite(row['hrl_greedy_plr']) else "N/A"
        lines.append(f"    & & \\textbf{{Proposed HRL (Greedy)}} & {row['hrl_greedy_sr']:.1f} & {hg_plr} & {row['hrl_greedy_per_step']:.2f} & {row['hrl_greedy_lat']:.2f} \\\\")
        # HRL (POMO):
        hp_plr = f"{row['hrl_pomo_plr']:.3f}" if math.isfinite(row['hrl_pomo_plr']) else "N/A"
        lines.append(f"    & & \\textbf{{Proposed HRL (POMO)}} & {row['hrl_pomo_sr']:.1f} & {hp_plr} & {row['hrl_pomo_per_step']:.2f} & {row['hrl_pomo_lat']:.2f} \\\\")
        if i < len(rows) - 1:
            lines.append(r"    \midrule")
    lines.extend([r"    \bottomrule", r"  \end{tabular}", r"\end{table*}"])
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  ✅ Saved: {output_path}")
    print("\n--- Table 1 (Multi-Map) Preview ---")
    print("\n".join(lines))

def _paper_fig5a_curves(output_path: Path) -> None:
    """Figure 5a: Learning Curves (SR, Loss, PLR) comparing stages."""
    import pandas as pd
    
    stages = {
        "Worker Only": ("logs/rl_worker_stage", "#3b82f6"),    # Blue
        "Manager Only": ("logs/rl_manager_stage", "#ef4444"),  # Red
        "Proposed HRL": ("logs/rl_joint_stage", "#10b981")     # Green
    }
    
    dfs = {}
    for label, (log_dir, color) in stages.items():
        candidate_files = list(Path(log_dir).glob("**/debug_metrics.csv"))
        if candidate_files:
            metrics_file = max(candidate_files, key=lambda p: p.stat().st_mtime)
            df = pd.read_csv(metrics_file)
            if not df.empty:
                dfs[label] = (df, color)
    
    if not dfs:
        print("  ⚠️ No debug_metrics.csv found in any stage to plot curves.")
        return

    def plot_metric(metric_keys, y_label, suffix):
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        has_data = False
        
        # metric_keys can be a list of fallbacks
        if isinstance(metric_keys, str):
            metric_keys = [metric_keys]
            
        for label, (df, color) in dfs.items():
            # Find the first available metric
            valid_col = None
            for key in metric_keys:
                if key in df.columns:
                    valid_col = key
                    break
                    
            if valid_col:
                has_data = True
                smooth_win = max(len(df) // 20, 1)
                df_smooth = df[valid_col].rolling(smooth_win, min_periods=1).mean()
                
                # Plot faint raw and bold smoothed
                ax.plot(df["ep"] if "ep" in df.columns else df.index, df[valid_col], color=color, alpha=0.15)
                ax.plot(df["ep"] if "ep" in df.columns else df.index, df_smooth, color=color, linewidth=2.5, label=label)
        
        if has_data:
            ax.set_ylabel(y_label, fontsize=20, fontweight="bold")
            ax.set_xlabel("Episode", fontsize=20)
            ax.tick_params(axis='both', labelsize=17)
            ax.legend(fontsize=14, loc="best")
            
            p = output_path.with_name(f"fig5a_{suffix}.png")
            fig.savefig(p, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"  ✅ Saved: {p}")
        else:
            plt.close(fig)

    # 1. Success Rate
    plot_metric(["success_rate", "success_ema"], "Success Rate (%)", "1_success_rate")
    
    # 2. Reward / Loss
    plot_metric(["total_reward", "reward_mean", "loss_mean"], "Reward", "2_reward")

    # 3. Progress / PLR
    plot_metric(["progress_mean", "path_length_ratio"], "Goal Progress / PLR", "3_progress")


def _paper_fig5b_path_viz(env, worker, output_path: Path) -> None:
    """Figure 5b: Representative Path Visualization without titles."""
    import tests.eval_core as core
    import random
    core.set_seed(42)  # Different seed from fig1
    
    positions = env.pos_tensor.detach().cpu().numpy()
    
    # Try to find a long, interesting path
    best_pair, best_hops = None, 0
    for _ in range(50):
        s = random.randint(0, env.num_nodes - 1)
        g = random.randint(0, env.num_nodes - 1)
        if s == g: continue
        h = float(env.hop_matrix[s, g].item())
        if math.isfinite(h) and 15 <= h <= 30 and h > best_hops:
            best_hops = h
            best_pair = (s, g)
            
    if best_pair is None:
        best_pair = (10, min(env.num_nodes - 1, 90))
        
    start_idx, goal_idx = best_pair
    rollout = core.run_worker_rollout(worker, env, start_idx, goal_idx)
    worker_path = rollout["path_nodes"]
    
    fig, ax = plt.subplots(figsize=(12, 9), constrained_layout=True)
    
    # Map edges (darker like in fig1, but maybe slightly softer)
    for src, dst in env.map_core.graph.edges():
        u, v = env.node_mapping[src], env.node_mapping[dst]
        ax.plot([positions[u, 0], positions[v, 0]], [positions[u, 1], positions[v, 1]], color="#6b7280", linewidth=0.8, alpha=0.6)
        
    # Plot Worker Path
    if len(worker_path) > 1:
        cmap = plt.get_cmap("spring") # Different colormap from fig1 to distinguish
        for i in range(len(worker_path) - 1):
            ax.plot([positions[worker_path[i], 0], positions[worker_path[i + 1], 0]], 
                    [positions[worker_path[i], 1], positions[worker_path[i + 1], 1]], 
                    color=cmap(i / max(len(worker_path) - 1, 1)), linewidth=4.0, alpha=0.9)
            
    ax.scatter(positions[start_idx, 0], positions[start_idx, 1], marker="o", s=300, color="#22c55e", edgecolors="black", linewidths=1.5, zorder=5, label="Start")
    ax.scatter(positions[goal_idx, 0], positions[goal_idx, 1], marker="X", s=350, color="#ef4444", edgecolors="black", linewidths=1.5, zorder=5, label="Goal")
    
    ax.legend(fontsize=12, loc="best")
    ax.set_aspect("equal")
    
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved: {output_path}")



def _paper_fig_trajectory_comparison(
    env,
    worker,
    manager,
    output_path: Path,
    map_name: str = "Goldcoast",
    seed: int = 77,
) -> None:
    """Flat RL (Worker Only) vs HRL (Worker+Manager) 궤적 비교 시각화.

    동일한 (start, goal) 쌍에 대해 두 모델의 경로를 나란히 보여준다.
    """
    import tests.eval_core as core
    import random
    import matplotlib.colors as mcolors

    core.set_seed(seed)
    positions = env.pos_tensor.detach().cpu().numpy()

    # ── 1. Worker Only가 성공하지만 PLR이 높은 쌍 탐색 ──
    print(f"  🔍 [{map_name}] 비교용 (start, goal) 쌍 탐색 중...")
    candidates = []
    for _ in range(200):
        s = random.randint(0, env.num_nodes - 1)
        g = random.randint(0, env.num_nodes - 1)
        if s == g:
            continue
        h = float(env.hop_matrix[s, g].item())
        if not math.isfinite(h) or h < 10:
            continue
        w_result = core.run_worker_rollout(worker, env, s, g, temperature=0.0)
        if w_result["success"] and w_result["path_length_ratio"] > 1.3:
            candidates.append((s, g, w_result))
            if len(candidates) >= 5:
                break

    if not candidates:
        # 폴백
        for _ in range(100):
            s = random.randint(0, env.num_nodes - 1)
            g = random.randint(0, env.num_nodes - 1)
            if s == g:
                continue
            h = float(env.hop_matrix[s, g].item())
            if math.isfinite(h) and h >= 8:
                w_result = core.run_worker_rollout(worker, env, s, g, temperature=0.0)
                if w_result["success"]:
                    candidates.append((s, g, w_result))
                    break

    if not candidates:
        print("  ❌ 적절한 (start, goal) 쌍을 찾지 못했습니다.")
        return

    best = max(candidates, key=lambda x: x[2]["path_length_ratio"] if math.isfinite(x[2]["path_length_ratio"]) else 0)
    start_idx, goal_idx, worker_result = best
    worker_path = worker_result["path_nodes"]
    worker_plr = worker_result["path_length_ratio"]
    optimal_hops = worker_result["optimal_hops"]

    print(f"  📍 선택된 쌍: start={start_idx}, goal={goal_idx}, "
          f"optimal_hops={optimal_hops:.0f}, Worker PLR={worker_plr:.3f}")

    # ── 2. HRL 롤아웃 ──
    hrl_result = core.run_joint_rollout(
        worker, manager, env, start_idx, goal_idx,
        temperature=0.0, mgr_temperature=0.0,
    )
    hrl_path = hrl_result["path"]
    hrl_success = hrl_result["success"]
    hrl_plr = hrl_result["path_length_ratio"] if hrl_result["success"] else float("inf")
    hrl_subgoals = hrl_result.get("plan_subgoals", [])

    print(f"  📍 HRL: success={hrl_success}, PLR={hrl_plr:.3f}, subgoals={len(hrl_subgoals)}")

    # ── 3. A* 최적 경로 복원 ──
    astar_path = None
    try:
        # BFS/Dijkstra 기반 최적 경로 복원
        from collections import deque
        hop_mat = env.hop_matrix
        queue = deque([(goal_idx, [goal_idx])])
        visited = {goal_idx}
        while queue:
            node, path_so_far = queue.popleft()
            if node == start_idx:
                astar_path = list(reversed(path_so_far))
                break
            neighbors = env.map_core.graph.neighbors(
                [k for k, v in env.node_mapping.items() if v == node][0]
            )
            for nb_raw in neighbors:
                nb = env.node_mapping[nb_raw]
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, path_so_far + [nb]))
    except Exception:
        pass

    # ── 4. Side-by-side 시각화 ──
    fig, (ax_flat, ax_hrl) = plt.subplots(1, 2, figsize=(24, 11), constrained_layout=True)

    for ax in (ax_flat, ax_hrl):
        for src, dst in env.map_core.graph.edges():
            u, v = env.node_mapping[src], env.node_mapping[dst]
            ax.plot(
                [positions[u, 0], positions[v, 0]],
                [positions[u, 1], positions[v, 1]],
                color="#d1d5db", linewidth=0.4, alpha=0.5,
            )
        ax.set_aspect("equal")
        ax.tick_params(axis='both', labelsize=13)

    # ── (a) Flat RL ──
    cmap_flat = plt.get_cmap("autumn")
    if len(worker_path) > 1:
        norm_f = mcolors.Normalize(vmin=0, vmax=max(len(worker_path) - 2, 1))
        for i in range(len(worker_path) - 1):
            ax_flat.plot(
                [positions[worker_path[i], 0], positions[worker_path[i + 1], 0]],
                [positions[worker_path[i], 1], positions[worker_path[i + 1], 1]],
                color=cmap_flat(norm_f(i)), linewidth=2.5, alpha=0.85,
            )
    if astar_path and len(astar_path) > 1:
        a_arr = np.array(astar_path)
        ax_flat.plot(positions[a_arr, 0], positions[a_arr, 1],
                     "--", color="#3b82f6", linewidth=2.0, alpha=0.7, label="A* Optimal")
    ax_flat.scatter(positions[start_idx, 0], positions[start_idx, 1],
                    marker="o", s=350, color="#22c55e", edgecolors="black",
                    linewidths=2.0, zorder=10, label="Start")
    ax_flat.scatter(positions[goal_idx, 0], positions[goal_idx, 1],
                    marker="X", s=400, color="#ef4444", edgecolors="black",
                    linewidths=2.0, zorder=10, label="Goal")
    sr_w = "✓ Success" if worker_result["success"] else "✗ Fail"
    ax_flat.set_title(
        f"(a) Flat RL (Worker Only)\n"
        f"Steps: {len(worker_path)-1} | PLR: {worker_plr:.2f} | {sr_w}",
        fontsize=20, fontweight="bold", pad=14,
    )
    ax_flat.legend(fontsize=15, loc="upper left")

    # ── (b) HRL ──
    cmap_hrl = plt.get_cmap("cool")
    if len(hrl_path) > 1:
        norm_h = mcolors.Normalize(vmin=0, vmax=max(len(hrl_path) - 2, 1))
        for i in range(len(hrl_path) - 1):
            ax_hrl.plot(
                [positions[hrl_path[i], 0], positions[hrl_path[i + 1], 0]],
                [positions[hrl_path[i], 1], positions[hrl_path[i + 1], 1]],
                color=cmap_hrl(norm_h(i)), linewidth=2.5, alpha=0.85,
            )
    if hrl_subgoals:
        sg_valid = [sg for sg in hrl_subgoals if sg < env.num_nodes]
        if sg_valid:
            sg_arr = np.array(sg_valid)
            ax_hrl.scatter(positions[sg_arr, 0], positions[sg_arr, 1],
                           marker="*", s=500, facecolors="#fbbf24", edgecolors="#92400e",
                           linewidths=1.5, zorder=8, label="Subgoals")
            for idx, sg in enumerate(sg_arr):
                ax_hrl.text(positions[sg, 0] + 0.001, positions[sg, 1] + 0.001,
                            f"$S_{{{idx+1}}}$", fontsize=14, fontweight="bold",
                            bbox=dict(facecolor="white", edgecolor="#fbbf24",
                                      boxstyle="round,pad=0.2", alpha=0.9))
    if astar_path and len(astar_path) > 1:
        a_arr = np.array(astar_path)
        ax_hrl.plot(positions[a_arr, 0], positions[a_arr, 1],
                    "--", color="#3b82f6", linewidth=2.0, alpha=0.7, label="A* Optimal")
    ax_hrl.scatter(positions[start_idx, 0], positions[start_idx, 1],
                   marker="o", s=350, color="#22c55e", edgecolors="black",
                   linewidths=2.0, zorder=10, label="Start")
    ax_hrl.scatter(positions[goal_idx, 0], positions[goal_idx, 1],
                   marker="X", s=400, color="#ef4444", edgecolors="black",
                   linewidths=2.0, zorder=10, label="Goal")
    sr_h = "✓ Success" if hrl_success else "✗ Fail"
    hrl_plr_s = f"{hrl_plr:.2f}" if math.isfinite(hrl_plr) else "N/A"
    ax_hrl.set_title(
        f"(b) Proposed HRL (Worker + Manager)\n"
        f"Steps: {len(hrl_path)-1} | PLR: {hrl_plr_s} | {sr_h}",
        fontsize=20, fontweight="bold", pad=14,
    )
    ax_hrl.legend(fontsize=15, loc="upper left")

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved: {output_path}")


# ============================================================
# 논문용 배치 평가 및 궤적 시각화 (paper_full, regen) 추가 로직
# ============================================================

def generate_trajectory_figures(
    env: DisasterEnv,
    flat_worker: WorkerLSTM,
    joint_worker: WorkerLSTM,
    manager: GraphTransformerManager,
    start_idx: int,
    goal_idx: int,
    output_dir: Path,
    num_samples: int = core.DEFAULT_NUM_SAMPLES,
) -> dict[str, Any]:
    positions = env.pos_tensor.detach().cpu().numpy()
    astar_path = core.reconstruct_astar_path(env, start_idx, goal_idx)
    optimal_dist = float(env.apsp_matrix[start_idx, goal_idx].item())
    optimal_hops = float(env.hop_matrix[start_idx, goal_idx].item())

    flat_result = core.run_worker_rollout(flat_worker, env, start_idx, goal_idx, temperature=0.0)
    hrl_result = core.run_joint_rollout_multisampling(
        joint_worker, manager, env, start_idx, goal_idx, num_samples=num_samples, temperature=core.DEFAULT_SAMPLE_TEMP
    )
    hrl_path = hrl_result["path"]
    hrl_subgoals = hrl_result.get("plan_subgoals", [])

    print(f"\n📍 OD: start={start_idx}, goal={goal_idx}")
    print(f"   Optimal: {optimal_hops:.0f} hops, {optimal_dist:.1f} km")
    print(f"   Flat RL: {'✅' if flat_result['success'] else '❌'}, steps={flat_result['actual_steps']}, PLR={flat_result['path_length_ratio']:.3f}")
    print(f"   HRL(N={num_samples}): {'✅' if hrl_result['success'] else '❌'}, steps={hrl_result['actual_steps']}, PLR={hrl_result['path_length_ratio']:.3f}, sr={hrl_result.get('sampling_success_rate', 0):.1f}%")

    # (a) Flat RL
    fig_a, ax_a = plt.subplots(figsize=(14, 11), constrained_layout=True)
    viz.draw_map_background(ax_a, env, positions)
    viz.draw_astar_path(ax_a, positions, astar_path)
    sm = viz.draw_path_gradient(ax_a, positions, flat_result["path_nodes"], cmap_name="autumn")
    if sm:
        cbar = fig_a.colorbar(sm, ax=ax_a, orientation='horizontal', fraction=0.046, pad=0.06)
        cbar.set_label('Trajectory Progress (Steps)', fontsize=16, fontweight='bold')
        cbar.ax.tick_params(labelsize=14)
    viz.draw_endpoints(ax_a, positions, start_idx, goal_idx)
    ax_a.legend(fontsize=17, loc="upper right")
    fig_a.savefig(output_dir / "flat_rl_trajectory.png", dpi=300, bbox_inches="tight")
    plt.close(fig_a)

    # (b) HRL + Subgoals
    fig_b, ax_b = plt.subplots(figsize=(14, 11), constrained_layout=True)
    viz.draw_map_background(ax_b, env, positions)
    viz.draw_astar_path(ax_b, positions, astar_path)
    sm_h = viz.draw_path_gradient(ax_b, positions, hrl_path, cmap_name="cool")
    if hrl_subgoals:
        sg_valid = [sg for sg in hrl_subgoals if sg < env.num_nodes]
        if sg_valid:
            sg_arr = np.array(sg_valid)
            ax_b.scatter(positions[sg_arr, 0], positions[sg_arr, 1], marker="*", s=600, facecolors="#fbbf24", edgecolors="#92400e", linewidths=1.5, zorder=8, label="Subgoals")
            for idx, sg in enumerate(sg_arr):
                ax_b.text(positions[sg, 0] + 0.001, positions[sg, 1] + 0.001, f"$S_{{{idx+1}}}$", fontsize=16, fontweight="bold", bbox=dict(facecolor="white", edgecolor="#fbbf24", boxstyle="round,pad=0.2", alpha=0.9))
    if sm_h:
        cbar_h = fig_b.colorbar(sm_h, ax=ax_b, orientation='horizontal', fraction=0.046, pad=0.06)
        cbar_h.set_label('Trajectory Progress (Steps)', fontsize=16, fontweight='bold')
        cbar_h.ax.tick_params(labelsize=14)
    viz.draw_endpoints(ax_b, positions, start_idx, goal_idx)
    ax_b.legend(fontsize=17, loc="upper right")
    fig_b.savefig(output_dir / "hrl_trajectory_with_subgoals.png", dpi=300, bbox_inches="tight")
    plt.close(fig_b)

    # (c) HRL Clean
    fig_c, ax_c = plt.subplots(figsize=(14, 11), constrained_layout=True)
    viz.draw_map_background(ax_c, env, positions)
    viz.draw_astar_path(ax_c, positions, astar_path)
    sm_c = viz.draw_path_gradient(ax_c, positions, hrl_path, cmap_name="cool")
    if sm_c:
        cbar_c = fig_c.colorbar(sm_c, ax=ax_c, orientation='horizontal', fraction=0.046, pad=0.06)
        cbar_c.set_label('Trajectory Progress (Steps)', fontsize=16, fontweight='bold')
        cbar_c.ax.tick_params(labelsize=14)
    viz.draw_endpoints(ax_c, positions, start_idx, goal_idx)
    ax_c.legend(fontsize=17, loc="upper right")
    fig_c.savefig(output_dir / "hrl_trajectory_clean.png", dpi=300, bbox_inches="tight")
    plt.close(fig_c)

    astar_dist = core.compute_path_distance(env, astar_path) if astar_path else optimal_dist

    return {
        "start": start_idx,
        "goal": goal_idx,
        "optimal_hops": optimal_hops,
        "optimal_dist": optimal_dist,
        "astar_path": astar_path,
        "astar_steps": len(astar_path) - 1 if astar_path else optimal_hops,
        "astar_dist": astar_dist,
        "flat_result": flat_result,
        "hrl_result": hrl_result,
    }


def generate_trajectory_metrics_txt(traj_info: dict[str, Any], output_dir: Path, env: DisasterEnv) -> None:
    flat = traj_info["flat_result"]
    hrl = traj_info["hrl_result"]
    opt_dist = traj_info["optimal_dist"]
    opt_hops = traj_info["optimal_hops"]
    flat_dist = flat.get("path_distance", 0.0)
    hrl_dist = hrl.get("path_distance", 0.0)

    summary = (
        f"================================================================\n"
        f"Trajectory Comparison on {env.name} ({env.num_nodes} nodes)\n"
        f"================================================================\n\n"
        f"[Problem Setting]\n  Map: {env.name}\n  Start Node: {traj_info['start']}\n  Goal Node: {traj_info['goal']}\n"
        f"  Optimal Hops: {opt_hops:.0f}\n  Optimal Distance: {opt_dist:.1f} km\n\n[Results]\n"
        f"  {'Metric':<30s} {'A*':>12s} {'Flat RL':>12s} {'HRL(N=16)':>12s}\n  {'-'*66}\n"
        f"  {'Success':<30s} {'Yes':>12s} {'Yes' if flat['success'] else 'No':>12s} {'Yes' if hrl['success'] else 'No':>12s}\n"
        f"  {'Actual Steps':<30s} {traj_info['astar_steps']:>12.0f} {flat['actual_steps']:>12d} {hrl['actual_steps']:>12d}\n"
        f"  {'Path Distance (km)':<30s} {traj_info['astar_dist']:>12.1f} {flat_dist:>12.1f} {hrl_dist:>12.1f}\n"
        f"  {'PLR (Distance Ratio)':<30s} {'1.000':>12s} {flat['path_length_ratio']:>12.3f} {hrl['path_length_ratio']:>12.3f}\n"
        f"  {'Subgoals':<30s} {'N/A':>12s} {'N/A':>12s} {len(hrl.get('plan_subgoals', [])):>12d}\n"
        f"  {'Sampling Success Rate':<30s} {'N/A':>12s} {'N/A':>12s} {hrl.get('sampling_success_rate', 0):>11.1f}%\n\n"
        f"[Improvement (HRL vs Flat RL)]\n"
    )

    if flat["success"] and hrl["success"]:
        plr_red = (flat["path_length_ratio"] - hrl["path_length_ratio"]) / flat["path_length_ratio"] * 100
        step_red = (flat["actual_steps"] - hrl["actual_steps"]) / flat["actual_steps"] * 100
        dist_red = (flat_dist - hrl_dist) / flat_dist * 100 if flat_dist > 0 else 0
        summary += (f"  PLR Reduction: {flat['path_length_ratio']:.3f} -> {hrl['path_length_ratio']:.3f} ({plr_red:.1f}%)\n"
                    f"  Step Reduction: {flat['actual_steps']} -> {hrl['actual_steps']} ({step_red:.1f}%)\n"
                    f"  Distance Reduction: {flat_dist:.1f} -> {hrl_dist:.1f} km ({dist_red:.1f}%)\n")
    else:
        summary += "  (비교 불가: 하나 이상의 방식이 실패)\n"
    summary += f"================================================================\n"
    
    (output_dir / "trajectory_metrics.txt").write_text(summary, encoding="utf-8")


def evaluate_all_methods(flat_worker, joint_worker, manager, env, num_episodes: int, num_samples: int, seed: int) -> dict:
    pairs = core.generate_od_pairs(env, num_episodes, min_hops=2, seed=seed)
    results = {}

    print(f"\n  🔷 A* 벤치마크 평가 중...")
    astar_results, astar_times = [], []
    for s, g in pairs:
        t0 = time.perf_counter()
        try:
            raw_s = [k for k, v in env.node_mapping.items() if v == s][0]
            raw_g = [k for k, v in env.node_mapping.items() if v == g][0]
            nx_path = nx.dijkstra_path(env.map_core.graph, raw_s, raw_g, weight="length")
            mapped_path = [env.node_mapping[n] for n in nx_path]
            success = True
        except Exception:
            mapped_path = [s]
            success = False
        t1 = time.perf_counter()
        astar_times.append((t1 - t0) * 1000.0)
        dist = core.compute_path_distance(env, mapped_path)
        opt_dist = float(env.apsp_matrix[s, g].item())
        plr = dist / max(opt_dist, 1e-6) if success and opt_dist > 0 else float("inf")
        astar_results.append({"success": success, "path_length_ratio": plr})
    
    astar_sr = sum(1 for r in astar_results if r["success"]) / len(astar_results) * 100
    astar_plrs = [r["path_length_ratio"] for r in astar_results if r["success"] and math.isfinite(r["path_length_ratio"])]
    results["A*"] = {"success_rate": astar_sr, "path_length_ratio": float(np.mean(astar_plrs)) if astar_plrs else float("inf"), "inference_latency_ms": float(np.mean(astar_times))}

    print(f"\n  🔶 Flat RL 평가 중...")
    flat_eval = core.evaluate_worker_batch(flat_worker, env, len(pairs), temperature=0.0, label="Flat RL", seed=seed, min_hops=2)
    results["Flat RL"] = {"success_rate": flat_eval["success_rate"], "path_length_ratio": flat_eval["path_length_ratio"], "inference_latency_ms": flat_eval["inference_latency_ms"]}

    print(f"\n  🟢 HRL (Greedy) 평가 중...")
    hrl_greedy_eval = core.evaluate_joint_batch(joint_worker, manager, env, len(pairs), temperature=0.0, mgr_temperature=0.0, label="HRL Greedy", seed=seed, min_hops=2)
    results["HRL (Greedy)"] = {"success_rate": hrl_greedy_eval["success_rate"], "path_length_ratio": hrl_greedy_eval["path_length_ratio"], "inference_latency_ms": hrl_greedy_eval["inference_latency_ms"]}

    print(f"\n  🔴 HRL (Multi-Sampling) 평가 중...")
    ms_successes = 0
    ms_plrs, ms_times = [], []
    for ep, (s, g) in enumerate(pairs):
        ms_result = core.run_joint_rollout_multisampling(joint_worker, manager, env, s, g, num_samples=num_samples, temperature=core.DEFAULT_SAMPLE_TEMP, measure_time=True)
        if ms_result["success"]:
            ms_successes += 1
            if math.isfinite(ms_result["path_length_ratio"]):
                ms_plrs.append(ms_result["path_length_ratio"])
        ms_times.append(ms_result.get("total_sampling_time_ms", 0.0))
        if (ep + 1) % 50 == 0:
            print(f"     [{ep+1}/{len(pairs)}] SR={(ms_successes / (ep + 1) * 100):.1f}%")
    
    ms_sr = ms_successes / len(pairs) * 100
    results["HRL (Sampling)"] = {"success_rate": ms_sr, "path_length_ratio": float(np.mean(ms_plrs)) if ms_plrs else float("inf"), "inference_latency_ms": float(np.mean(ms_times))}

    return results


def save_all_maps_eval_txt(all_results: dict[str, dict[str, dict[str, Any]]], num_episodes: int, output_dir: Path) -> None:
    lines = [f"================================================================", f"Batch Evaluation Results — All Maps", f"Episodes per map: {num_episodes}", f"================================================================"]
    for map_name, results in all_results.items():
        lines.extend(["", f"  📍 {map_name}", f"  {'Method':<25s} {'SR (%)':>10s} {'PLR':>10s} {'Latency (ms)':>15s}", f"  {'-'*60}"])
        for method, data in results.items():
            sr = f"{data['success_rate']:.1f}"
            plr = f"{data['path_length_ratio']:.3f}" if math.isfinite(data['path_length_ratio']) else "inf"
            lat = f"{data['inference_latency_ms']:.2f}"
            lines.append(f"  {method:<25s} {sr:>10s} {plr:>10s} {lat:>15s}")
    lines.append(f"================================================================")
    (output_dir / "batch_eval_results.txt").write_text("\n".join(lines), encoding="utf-8")


def generate_learning_curves(output_dir: Path) -> None:
    worker_csv = Path("logs/rl_worker_stage/2026-04-13_1923_worker_pomo32/debug_metrics.csv")
    joint_csv = Path("logs/rl_joint_stage/2026-04-14_2020_joint_pomo16/debug_metrics.csv")
    if not worker_csv.exists() or not joint_csv.exists():
        print("  ⚠️ 학습 곡선 CSV를 찾을 수 없어 건너뜁니다.")
        return
    dw = pd.read_csv(worker_csv)
    dj = pd.read_csv(joint_csv)
    
    def smooth(series: pd.Series, frac: float = 0.05) -> pd.Series:
        win = max(int(len(series) * frac), 1)
        return series.rolling(win, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    j_sr = dj["success_ema"] * 100
    ax.plot(dw["episode"], dw["success_ema"], color=viz.COLORS["Worker Only"], alpha=0.12)
    ax.plot(dw["episode"], smooth(dw["success_ema"]), color=viz.COLORS["Worker Only"], linewidth=2.5, label="Worker Only")
    ax.plot(dj["ep"], j_sr, color=viz.COLORS["Proposed HRL"], alpha=0.12)
    ax.plot(dj["ep"], smooth(j_sr), color=viz.COLORS["Proposed HRL"], linewidth=2.5, label="Proposed HRL")
    ax.set_ylabel("Success Rate (EMA, %)", fontsize=18, fontweight="bold")
    ax.set_xlabel("Episode", fontsize=18)
    ax.legend(fontsize=13, framealpha=0.9)
    fig.savefig(output_dir / "success_rate.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(dw["episode"], dw["loss"], color=viz.COLORS["Worker Only"], alpha=0.12)
    ax.plot(dw["episode"], smooth(dw["loss"]), color=viz.COLORS["Worker Only"], linewidth=2.5, label="Worker Only")
    ax.plot(dj["ep"], dj["loss_mean"], color=viz.COLORS["Proposed HRL"], alpha=0.12)
    ax.plot(dj["ep"], smooth(dj["loss_mean"]), color=viz.COLORS["Proposed HRL"], linewidth=2.5, label="Proposed HRL")
    ax.set_ylabel("Policy Loss", fontsize=18, fontweight="bold")
    ax.set_xlabel("Episode", fontsize=18)
    ax.legend(fontsize=13, framealpha=0.9)
    fig.savefig(output_dir / "loss.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ✅ Saved: learning curves")


def cmd_paper_full(args: argparse.Namespace) -> None:
    device = core.get_device()
    core.set_seed(args.seed)
    output_dir = Path("tests/paper_figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n📦 모델 로드 중...")
    joint_worker, manager, _ = core.load_checkpoint(args.joint_ckpt, device=device, load_manager=True)
    flat_worker, _, _ = core.load_checkpoint(args.worker_ckpt, device=device, load_manager=False)
    
    maps_to_eval = [
        {"name": "SiouxFalls", "nodes": 24},
        {"name": "Anaheim", "nodes": 416},
        {"name": "ChicagoSketch", "nodes": 933},
        {"name": "Goldcoast", "nodes": 4807},
    ]

    all_map_results = {}
    for map_cfg in maps_to_eval:
        map_name = map_cfg["name"]
        print(f"\n{'='*60}\n🔬 배치 평가: {map_name}\n{'='*60}")
        env_map = core.setup_env(map_name, device)
        map_results = evaluate_all_methods(flat_worker, joint_worker, manager, env_map, args.episodes, args.num_samples, args.seed)
        all_map_results[map_name] = map_results

        if map_name == "Goldcoast":
            print(f"\n🎨 궤적 시각화 (Goldcoast)")
            start_idx, goal_idx = core.find_long_od_pair(env_map, min_hops=50, seed=args.seed)
            traj_info = generate_trajectory_figures(env_map, flat_worker, joint_worker, manager, start_idx, goal_idx, output_dir, args.num_samples)
            generate_trajectory_metrics_txt(traj_info, output_dir, env_map)

    save_all_maps_eval_txt(all_map_results, args.episodes, output_dir)
    
    if "Anaheim" in all_map_results:
        viz.generate_latency_chart(all_map_results["Anaheim"], output_dir)
    generate_learning_curves(output_dir)
    print(f"\n✅ 완료! 출력: {output_dir.resolve()}")


def cmd_regen(args: argparse.Namespace) -> None:
    device = core.get_device()
    core.set_seed(args.seed)
    output_dir = Path("tests/paper_figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n📦 모델 로드 중...")
    joint_worker, manager, _ = core.load_checkpoint(args.joint_ckpt, device=device, load_manager=True)
    flat_worker, _, _ = core.load_checkpoint(args.worker_ckpt, device=device, load_manager=False)
    
    print(f"\n🗺️ {args.map} 환경 로드 중...")
    env = core.setup_env(args.map, device)
    
    start_idx, goal_idx, _, _ = core.find_best_od_pair(
        env, flat_worker, joint_worker, manager, min_hops=50, num_candidates=args.candidates, num_samples=args.num_samples
    )
    print(f"\n🎨 최적 궤적 시각화 재성성 중...")
    traj_info = generate_trajectory_figures(
        env, flat_worker, joint_worker, manager, start_idx, goal_idx, output_dir, args.num_samples
    )
    generate_trajectory_metrics_txt(traj_info, output_dir, env)
    print(f"\n✅ 완료! 출력: {output_dir.resolve()}")


# ============================================================
# main 엔트리포인트
# ============================================================

def main() -> None:
    args = parse_args()

    if args.command is None:
        print("서브커맨드를 지정해주세요. 사용 가능: dashboard, worker, joint, compare, memorization, physics, paper, paper_full, regen")
        print("예시: python tests/evaluate.py dashboard --run-dir logs/rl_phase1_apte")
        return

    cmd_map = {
        "dashboard": cmd_dashboard,
        "worker": cmd_worker,
        "joint": cmd_joint,
        "compare": cmd_compare,
        "memorization": cmd_memorization,
        "physics": cmd_physics,
        "paper": cmd_paper,
        "paper_full": cmd_paper_full,
        "regen": cmd_regen,
    }
    cmd_map[args.command](args)


if __name__ == "__main__":
    main()

