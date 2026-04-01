from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
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


def _rolling(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def format_percent(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.1f}%"


def format_float(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.3f}"


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


def _plot_line(
    ax: plt.Axes,
    df: pd.DataFrame,
    columns: list[tuple[str, str]],
    title: str,
    ylabel: str,
    rolling_window: int,
    percent: bool = False,
) -> None:
    has_data = False
    for column, label in columns:
        if column not in df.columns:
            continue
        series = df[column]
        if series.notna().sum() == 0:
            continue
        has_data = True
        ax.plot(df["episode"], series, alpha=0.25, linewidth=1.0)
        ax.plot(df["episode"], _rolling(series, rolling_window), label=label, linewidth=2.2)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linestyle="--")
    if percent:
        ax.set_ylim(bottom=0.0)
    if has_data:
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)


def plot_training_dashboard(df: pd.DataFrame, output_path: Path, rolling_window: int) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(4, 3, figsize=(18, 18), constrained_layout=True)
    fig.suptitle("RL Training Dashboard", fontsize=18, fontweight="bold")
    _plot_line(
        axes[0, 0],
        df,
        [("success_rate", "Success Rate"), ("success_ema", "Success EMA"), ("goal_hit_rate", "Goal Hit Rate")],
        "Success Trends",
        "Percent",
        rolling_window,
        True,
    )
    _plot_line(
        axes[0, 1],
        df,
        [("loop_fail_rate", "Loop Fail"), ("stagnation_fail_rate", "Stagnation Fail"), ("fail_shaping", "Fail Shaping")],
        "Failure Signals",
        "Value",
        rolling_window,
    )
    _plot_line(axes[0, 2], df, [("total_final", "Total Final"), ("success_return", "Success Return"), ("fail_return", "Fail Return")], "Return Structure", "Reward", rolling_window)
    _plot_line(axes[1, 0], df, [("plan_length_mean", "Plan Length"), ("eos_rate", "EOS Rate")], "Plan Length and EOS", "Value", rolling_window)
    _plot_line(axes[1, 1], df, [("plan_density_mean", "Density"), ("plan_density_gap", "Density Gap")], "Plan Density", "Value", rolling_window)
    _plot_line(
        axes[1, 2],
        df,
        [("corridor_mean", "Corridor"), ("corridor_success", "Succ Corridor"), ("corridor_fail", "Fail Corridor")],
        "Goal Corridor Quality",
        "Percent",
        rolling_window,
        True,
    )
    _plot_line(
        axes[2, 0],
        df,
        [("plan_utilization", "Plan Utilization"), ("subgoal_reach_rate", "Subgoal Reach"), ("hit_at_3", "Hit@3")],
        "Worker Following Quality",
        "Percent",
        rolling_window,
        True,
    )
    _plot_line(axes[2, 1], df, [("goal_per100", "Goal per100"), ("subgoal_per100", "SG per100"), ("progress_mean", "Progress Mean")], "Goal Progress Signals", "Value", rolling_window)
    _plot_line(
        axes[2, 2],
        df,
        [("worker_steps_mean", "Worker Steps"), ("worker_entropy", "Worker Entropy"), ("manager_entropy", "Manager Entropy")],
        "Execution Characteristics",
        "Value",
        rolling_window,
    )
    _plot_line(axes[3, 0], df, [("explained_variance", "Explained Variance"), ("critic_mse", "Critic MSE"), ("td_abs_err", "TD |err|")], "Critic Health", "Value", rolling_window)
    _plot_line(axes[3, 1], df, [("manager_grad_clip_hit", "Manager Clip-Hit"), ("worker_grad_clip_hit", "Worker Clip-Hit")], "Gradient Saturation", "Percent", rolling_window, True)
    _plot_line(axes[3, 2], df, [("mgr_lr", "Manager LR"), ("wkr_lr", "Worker LR")], "Learning Rates", "LR", rolling_window)
    for ax in axes[-1, :]:
        ax.set_xlabel("Episode")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_relationships(df: pd.DataFrame, output_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle("Success Relationships", fontsize=16, fontweight="bold")
    specs = [
        ("plan_density_mean", "success_rate", "Plan Density vs Success Rate"),
        ("corridor_mean", "success_rate", "Corridor vs Success Rate"),
        ("plan_utilization", "success_rate", "Plan Utilization vs Success Rate"),
        ("loop_fail_rate", "success_rate", "Loop Fail vs Success Rate"),
    ]
    for ax, (x_key, y_key, title) in zip(axes.flat, specs):
        if x_key not in df.columns or y_key not in df.columns:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue
        valid = df[[x_key, y_key]].dropna()
        if valid.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue
        scatter = ax.scatter(valid[x_key], valid[y_key], c=df.loc[valid.index, "episode"], cmap="viridis", alpha=0.85, edgecolor="none")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(x_key.replace("_", " ").title())
        ax.set_ylabel(y_key.replace("_", " ").title())
        ax.grid(alpha=0.25, linestyle="--")
        fig.colorbar(scatter, ax=ax, label="Episode")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def select_edge_attr(edge_attr: torch.Tensor | None) -> torch.Tensor | None:
    if edge_attr is None:
        return None
    if edge_attr.size(1) >= 9:
        return edge_attr[:, [0, 1, 4, 6, 8]]
    if edge_attr.size(1) >= 5:
        return edge_attr[:, :5]
    pad = torch.zeros(edge_attr.size(0), 5 - edge_attr.size(1), device=edge_attr.device, dtype=edge_attr.dtype)
    return torch.cat([edge_attr, pad], dim=1)


def load_state_dict_flexible(path: Path, device: torch.device) -> dict[str, Any]:
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    return payload


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

    manager = GraphTransformerManager(node_dim=4, hidden_dim=hidden_dim, dropout=0.2).to(device)
    worker = WorkerLSTM(node_dim=7, hidden_dim=hidden_dim).to(device)

    if checkpoint_info["source"] == "sl":
        payload = torch.load(checkpoint_info["sl_path"], map_location=device)
        manager.load_state_dict(payload["manager_state"])
        worker.load_state_dict(payload["worker_state"])
    else:
        manager.load_state_dict(load_state_dict_flexible(checkpoint_info["manager_path"], device))
        worker.load_state_dict(load_state_dict_flexible(checkpoint_info["worker_path"], device))

    manager.eval()
    worker.eval()
    return env, manager, worker


def ensure_env_node_feature_layout(env: DisasterEnv) -> None:
    x = env.pyg_data.x
    if x.size(1) < 8:
        pad = torch.zeros(x.size(0), 8 - x.size(1), device=x.device, dtype=x.dtype)
        env.pyg_data.x = torch.cat([x, pad], dim=1)


def refresh_env_node_features(env: DisasterEnv) -> None:
    ensure_env_node_feature_layout(env)
    batch_indices = torch.arange(env.batch_size, device=env.device)
    flat_current = batch_indices * env.num_nodes + env.current_node
    flat_target = batch_indices * env.num_nodes + env.target_node

    env.pyg_data.x[:, 2] = 0.0
    env.pyg_data.x[:, 3] = 0.0
    env.pyg_data.x[flat_current, 2] = 1.0
    env.pyg_data.x[flat_target, 3] = 1.0

    if not hasattr(env, "visit_count") or env.visit_count is None:
        env.visit_count = torch.zeros((env.batch_size, env.num_nodes), dtype=torch.float32, device=env.device)
        env.visit_count.scatter_(1, env.current_node.unsqueeze(1), 1.0)
    env.pyg_data.x[:, 4] = env.visit_count.view(-1)

    target_dists = env.apsp_matrix[env.target_node]
    env.pyg_data.x[:, 5] = (target_dists / max(env.max_dist, 1.0)).view(-1)

    target_pos = env.pos_tensor[env.target_node].unsqueeze(1)
    node_pos = env.pos_tensor.unsqueeze(0)
    direction = target_pos - node_pos
    direction = direction / direction.norm(dim=2, keepdim=True).clamp(min=1e-8)
    env.pyg_data.x[:, 6:8] = direction.view(-1, 2)


def configure_single_problem(
    env: DisasterEnv,
    start_idx: int | None,
    goal_idx: int | None,
) -> tuple[int, int]:
    env.reset(batch_size=1, sync_problem=True)
    refresh_env_node_features(env)

    if start_idx is None and goal_idx is None:
        return int(env.current_node.item()), int(env.target_node.item())
    if (start_idx is None) != (goal_idx is None):
        raise ValueError("Please provide both --start-node and --goal-node together.")

    start = int(start_idx)
    goal = int(goal_idx)
    if start == goal:
        raise ValueError("start-node and goal-node must be different.")

    env.current_node = torch.tensor([start], dtype=torch.long, device=env.device)
    env.target_node = torch.tensor([goal], dtype=torch.long, device=env.device)
    env.visited = torch.zeros((1, env.num_nodes), dtype=torch.bool, device=env.device)
    env.visited[0, start] = True
    env.visit_count = torch.zeros((1, env.num_nodes), dtype=torch.float32, device=env.device)
    env.visit_count[0, start] = 1.0
    env.history = [env.current_node.clone()]
    env.step_count = 0

    agent_data = env.converter.networkx_to_pyg(start, goal).to(env.device)
    env.pyg_data = Batch.from_data_list([agent_data])
    env.pyg_data.num_graphs = 1
    if hasattr(env, "damage_states"):
        env._update_edge_attributes(1, env.damage_states)
    refresh_env_node_features(env)
    return start, goal


def truncate_plan(sequence: list[int], eos_index: int, num_nodes: int) -> list[int]:
    plan: list[int] = []
    for token in sequence:
        if token == eos_index:
            break
        if 0 <= token < num_nodes:
            plan.append(int(token))
    return plan


def safe_softmax(scores: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    finite = torch.isfinite(scores)
    if not finite.any():
        probs = torch.zeros_like(scores)
        probs[-1] = 1.0
        return probs
    scores_safe = scores.clone()
    scores_safe[~finite] = -1e9
    if temperature > 1e-5:
        scores_safe = scores_safe / temperature
    return F.softmax(scores_safe, dim=-1)


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
        ax.scatter(positions[manager_plan, 0], positions[manager_plan, 1], marker="s", s=48, color="#ff7f0e", zorder=4)
        for rank, node in enumerate(manager_plan, start=1):
            ax.text(positions[node, 0], positions[node, 1], str(rank), fontsize=8, ha="center", va="center", zorder=5)
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
        fig.colorbar(scatter, ax=ax, label="Worker step")

    ax.scatter(positions[start_idx, 0], positions[start_idx, 1], marker="o", s=150, color="#2ca02c", label="Start", zorder=8)
    ax.scatter(positions[goal_idx, 0], positions[goal_idx, 1], marker="X", s=170, color="#d62728", label="Goal", zorder=8)
    final_node = worker_path[-1]
    if final_node not in {start_idx, goal_idx}:
        ax.scatter(positions[final_node, 0], positions[final_node, 1], marker="D", s=90, color="#1f77b4" if success else "#9467bd", label="Final node", zorder=8)

    status = "SUCCESS" if success else "FAIL"
    ax.set_title(f"Rollout Map - {checkpoint_label} ({status})", fontsize=16, fontweight="bold")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(frameon=False, loc="upper right")
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
    parser = argparse.ArgumentParser(description="Visualize RL debug metrics and rollout traces.")
    parser.add_argument("--run-dir", default="logs/rl_finetune_phase1", help="Directory containing rl_debug_log.txt and RL checkpoints.")
    parser.add_argument("--rolling-window", type=int, default=5, help="Rolling mean window for dashboard plots.")
    parser.add_argument("--map", default="Anaheim", help="Map name used during training.")
    parser.add_argument("--disaster", action="store_true", help="Enable disaster mode when reconstructing rollout.")
    parser.add_argument("--device", default=None, help="Device override for rollout, e.g. cpu or cuda.")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Model hidden size used for checkpoint loading.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for rollout sampling.")
    parser.add_argument("--rollout", action="store_true", help="Run a single checkpoint rollout with step-level debug outputs.")
    parser.add_argument("--checkpoint", default="best_sr", help="Checkpoint to visualize: latest, best_sr, best_ema, sl, or an episode number.")
    parser.add_argument("--start-node", type=int, default=None, help="Optional fixed start node for rollout.")
    parser.add_argument("--goal-node", type=int, default=None, help="Optional fixed goal node for rollout.")
    parser.add_argument("--manager-max-len", type=int, default=20, help="Maximum manager decoding length for rollout.")
    parser.add_argument("--max-total-steps", type=int, default=400, help="Maximum worker steps for rollout.")
    parser.add_argument("--manager-temperature", type=float, default=1e-6, help="Manager sampling temperature. Use a tiny value for greedy decoding.")
    parser.add_argument("--worker-temperature", type=float, default=1e-6, help="Worker sampling temperature. Use a tiny value for greedy decoding.")
    parser.add_argument("--topk", type=int, default=8, help="Number of candidates to store and plot per step.")
    parser.add_argument("--save-step-plots", action="store_true", help="Save one image per manager and worker step.")
    parser.add_argument("--max-worker-step-plots", type=int, default=30, help="Maximum number of per-step worker plots to save.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    run_dir = Path(args.run_dir)
    log_path = run_dir / "rl_debug_log.txt"
    if not log_path.exists():
        raise FileNotFoundError(f"Debug log not found: {log_path}")

    dashboard_dir = run_dir / "eval_dashboard"
    dashboard_dir.mkdir(parents=True, exist_ok=True)

    df = parse_debug_log(log_path)
    checkpoint_episodes = discover_checkpoint_episodes(run_dir)
    summary = build_summary(df, checkpoint_episodes)

    df.to_csv(dashboard_dir / "parsed_debug_metrics.csv", index=False)
    save_json(dashboard_dir / "summary.json", summary)
    save_summary_markdown(summary, dashboard_dir / "summary.md")
    plot_training_dashboard(df, dashboard_dir / "training_dashboard.png", args.rolling_window)
    plot_relationships(df, dashboard_dir / "metric_relationships.png")
    plot_snapshot_comparison(df, dashboard_dir / "snapshot_comparison.png")

    samples_path = dashboard_dir / "episode_samples_preview.json"
    if samples_path.exists():
        maybe_load_episode_samples(samples_path)

    print_terminal_summary(summary)
    print(f"Saved parsed metrics to: {dashboard_dir / 'parsed_debug_metrics.csv'}")
    print(f"Saved dashboard outputs to: {dashboard_dir}")

    if args.rollout:
        checkpoint_info = resolve_checkpoint_spec(run_dir, args.checkpoint, summary, checkpoint_episodes)
        run_rollout_visualization(run_dir, dashboard_dir, checkpoint_info, args)


if __name__ == "__main__":
    main()
