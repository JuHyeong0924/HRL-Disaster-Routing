"""시각화 모듈 — 모든 차트/맵/궤적 렌더링 함수를 모아놓은 순수 렌더링 라이브러리.

이 모듈의 함수들은 사전 계산된 데이터를 받아 시각화만 수행한다.
롤아웃/모델 실행 로직은 포함하지 않는다.

분리 원칙:
  - matplotlib 의존: 이 모듈에 배치
  - 롤아웃/모델 의존: evaluate.py 또는 eval_core.py에 유지
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import pandas as pd


# ============================================================
# 상수
# ============================================================

# 색상 팔레트 (논문용)
COLORS: Dict[str, str] = {
    "A*": "#6366f1",
    "Flat RL": "#f59e0b",
    "HRL (Greedy)": "#10b981",
    "HRL (Sampling)": "#ef4444",
    "Worker Only": "#3b82f6",
    "Proposed HRL": "#10b981",
}


# ============================================================
# 1. 기본 헬퍼
# ============================================================

def _rolling(series: pd.Series, window: int) -> pd.Series:
    """이동 평균 계산 헬퍼."""
    return series.rolling(window=window, min_periods=1).mean()


# ============================================================
# 2. 학습 대시보드 차트
# ============================================================

def _plot_line(
    ax: plt.Axes,
    df: pd.DataFrame,
    columns: list[tuple[str, str]],
    title: str,
    ylabel: str,
    rolling_window: int,
    percent: bool = False,
) -> None:
    """개별 축에 메트릭 라인 차트를 그린다."""
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


def _plot_single_chart(
    df: pd.DataFrame,
    columns: list[tuple[str, str]],
    title: str,
    ylabel: str,
    rolling_window: int,
    output_path: Path,
    percent: bool = False,
) -> None:
    """개별 차트를 단독 figure로 저장하는 헬퍼 함수."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    _plot_line(ax, df, columns, title, ylabel, rolling_window, percent)
    ax.set_xlabel("Episode")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_training_dashboard(df: pd.DataFrame, output_path: Path, rolling_window: int) -> None:
    """RL 학습 대시보드 — 12개 지표를 개별 차트 + 합본으로 저장한다."""
    chart_specs = [
        ("01_success_trends", [("success_rate", "Success Rate"), ("success_ema", "Success EMA"), ("goal_hit_rate", "Goal Hit Rate")], "Success Trends", "Percent", True),
        ("02_failure_signals", [("loop_fail_rate", "Loop Fail"), ("stagnation_fail_rate", "Stagnation Fail"), ("fail_shaping", "Fail Shaping")], "Failure Signals", "Value", False),
        ("03_return_structure", [("total_final", "Total Final"), ("success_return", "Success Return"), ("fail_return", "Fail Return")], "Return Structure", "Reward", False),
        ("04_plan_length_eos", [("plan_length_mean", "Plan Length"), ("eos_rate", "EOS Rate")], "Plan Length and EOS", "Value", False),
        ("05_plan_density", [("plan_density_mean", "Density"), ("plan_density_gap", "Density Gap")], "Plan Density", "Value", False),
        ("06_corridor_quality", [("corridor_mean", "Corridor"), ("corridor_success", "Succ Corridor"), ("corridor_fail", "Fail Corridor")], "Goal Corridor Quality", "Percent", True),
        ("07_worker_following", [("plan_utilization", "Plan Utilization"), ("subgoal_reach_rate", "Subgoal Reach"), ("hit_at_3", "Hit@3")], "Worker Following Quality", "Percent", True),
        ("08_goal_progress", [("goal_per100", "Goal per100"), ("subgoal_per100", "SG per100"), ("progress_mean", "Progress Mean")], "Goal Progress Signals", "Value", False),
        ("09_execution_chars", [("worker_steps_mean", "Worker Steps"), ("worker_entropy", "Worker Entropy"), ("manager_entropy", "Manager Entropy")], "Execution Characteristics", "Value", False),
        ("10_critic_health", [("explained_variance", "Explained Variance"), ("critic_mse", "Critic MSE"), ("td_abs_err", "TD |err|")], "Critic Health", "Value", False),
        ("11_gradient_saturation", [("manager_grad_clip_hit", "Manager Clip-Hit"), ("worker_grad_clip_hit", "Worker Clip-Hit")], "Gradient Saturation", "Percent", True),
        ("12_learning_rates", [("mgr_lr", "Manager LR"), ("wkr_lr", "Worker LR")], "Learning Rates", "LR", False),
    ]

    # 개별 차트 저장
    charts_dir = output_path.parent / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    for filename, columns, title, ylabel, percent in chart_specs:
        _plot_single_chart(df, columns, title, ylabel, rolling_window, charts_dir / f"{filename}.png", percent)

    # 합본 대시보드
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(4, 3, figsize=(18, 18), constrained_layout=True)
    fig.suptitle("RL Training Dashboard", fontsize=18, fontweight="bold")
    for idx, (_, columns, title, ylabel, percent) in enumerate(chart_specs):
        row, col = divmod(idx, 3)
        _plot_line(axes[row, col], df, columns, title, ylabel, rolling_window, percent)
    for ax in axes[-1, :]:
        ax.set_xlabel("Episode")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_apte_training_dashboard(df: pd.DataFrame, output_path: Path, rolling_window: int) -> None:
    """APTE Phase1 전용 학습 대시보드."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle("Phase1 APTE Training Dashboard", fontsize=16, fontweight="bold")

    ep = df["ep"] if "ep" in df.columns else df.index

    # Success EMA
    if "success_ema" in df.columns:
        axes[0, 0].plot(ep, df["success_ema"], alpha=0.3)
        axes[0, 0].plot(ep, _rolling(df["success_ema"], rolling_window), linewidth=2.2)
    axes[0, 0].set_title("Success EMA (%)")
    axes[0, 0].grid(alpha=0.25, linestyle="--")

    # Stagnation / Loop
    for col, lbl in [("stagnation_fail_rate", "Stag Fail"), ("loop_fail_rate", "Loop Fail")]:
        if col in df.columns:
            axes[0, 1].plot(ep, _rolling(df[col], rolling_window), label=lbl, linewidth=2.2)
    axes[0, 1].set_title("Failure Rates")
    axes[0, 1].legend(frameon=False, fontsize=8)
    axes[0, 1].grid(alpha=0.25, linestyle="--")

    # ExplVar
    if "critic_explained_variance" in df.columns:
        axes[1, 0].plot(ep, _rolling(df["critic_explained_variance"], rolling_window), linewidth=2.2)
    axes[1, 0].set_title("Critic ExplVar")
    axes[1, 0].grid(alpha=0.25, linestyle="--")

    # PLR
    if "path_length_ratio" in df.columns:
        axes[1, 1].plot(ep, _rolling(df["path_length_ratio"], rolling_window), linewidth=2.2)
    axes[1, 1].set_title("Path Length Ratio")
    axes[1, 1].grid(alpha=0.25, linestyle="--")

    for ax in axes[-1, :]:
        ax.set_xlabel("Episode")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_relationships(df: pd.DataFrame, output_path: Path) -> None:
    """메트릭 간 관계 산점도 (2x2)."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    fig.suptitle("Metric Relationships", fontsize=14, fontweight="bold")

    pairs = [
        ("success_rate", "stagnation_fail_rate", "Success ↔ Stagnation"),
        ("success_rate", "explained_variance", "Success ↔ ExplVar"),
        ("progress_mean", "goal_per100", "Progress ↔ Goal/100"),
        ("plan_utilization", "success_rate", "Plan Util ↔ Success"),
    ]
    for idx, (x_col, y_col, title) in enumerate(pairs):
        ax = axes.flat[idx]
        if x_col in df.columns and y_col in df.columns:
            mask = df[x_col].notna() & df[y_col].notna()
            ax.scatter(df.loc[mask, x_col], df.loc[mask, y_col], alpha=0.35, s=10)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(x_col, fontsize=8)
        ax.set_ylabel(y_col, fontsize=8)
        ax.grid(alpha=0.2, linestyle="--")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 3. 롤아웃 맵 / 스텝 차트
# ============================================================

def flatten_step_records(steps: list[dict[str, Any]], topk_slots: int = 5) -> pd.DataFrame:
    """스텝 딕셔너리 리스트를 DataFrame으로 전환한다."""
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
    env: Any,
    start_idx: int,
    goal_idx: int,
    manager_plan: list[int],
    worker_path: list[int],
    checkpoint_label: str,
    success: bool,
    manager_steps: list[dict[str, Any]] | None = None,
) -> None:
    """롤아웃 지도 — 도로 네트워크 위에 Manager 계획 + Worker 경로를 시각화한다."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=True)
    positions = env.pos_tensor.detach().cpu().numpy()

    # 도로 배경
    for src, dst in env.map_core.graph.edges():
        u = env.node_mapping[src]
        v = env.node_mapping[dst]
        ax.plot([positions[u, 0], positions[v, 0]], [positions[u, 1], positions[v, 1]],
                color="#b9c2cf", linewidth=0.6, alpha=0.35, zorder=1)

    # Manager 계획
    if manager_plan:
        planned_nodes = [start_idx] + manager_plan + [goal_idx]
        ax.plot(positions[planned_nodes, 0], positions[planned_nodes, 1],
                linestyle="--", color="#ff7f0e", linewidth=2.0, alpha=0.95, label="Manager plan", zorder=3)
        ax.scatter(positions[manager_plan, 0], positions[manager_plan, 1],
                   marker="*", s=250, facecolors="#ffbf00", edgecolors="black", linewidths=1.0, zorder=5)
        for rank, node in enumerate(manager_plan, start=1):
            ax.annotate(f"SG{rank}\n(N={node})", (positions[node, 0], positions[node, 1]),
                        textcoords="offset points", xytext=(8, 8),
                        fontsize=9, fontweight="bold", ha="left", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff8e8", edgecolor="#ff7f0e", alpha=0.9), zorder=6)

    # 의심 Manager 스텝 강조
    if manager_steps:
        suspicious = [step for step in manager_steps if step.get("warning_level") != "ok" and step.get("selected_label") != "EOS"]
        if suspicious:
            suspicious_nodes = [int(step["selected_node"]) for step in suspicious]
            ax.scatter(positions[suspicious_nodes, 0], positions[suspicious_nodes, 1],
                       s=240, facecolors="none", edgecolors="#d62728", linewidths=2.0,
                       label="Flagged manager step", zorder=8)
            for step in suspicious:
                node = int(step["selected_node"])
                ax.annotate(f"!{step['step']}", xy=(positions[node, 0], positions[node, 1]),
                            xytext=(0, 10), textcoords="offset points",
                            color="#d62728", fontsize=9, fontweight="bold", ha="center", va="bottom", zorder=9)

    # Worker 경로
    if len(worker_path) > 1:
        cmap = plt.get_cmap("viridis")
        denom = max(len(worker_path) - 1, 1)
        for idx in range(len(worker_path) - 1):
            src = worker_path[idx]
            dst = worker_path[idx + 1]
            ax.plot([positions[src, 0], positions[dst, 0]], [positions[src, 1], positions[dst, 1]],
                    color=cmap(idx / denom), linewidth=2.6, alpha=0.95, zorder=6)
        scatter = ax.scatter(positions[worker_path, 0], positions[worker_path, 1],
                             c=np.arange(len(worker_path)), cmap="viridis", s=26, zorder=7, label="Worker path")
        cbar = fig.colorbar(scatter, ax=ax, label="Worker step", orientation="horizontal", pad=0.06, shrink=0.6, aspect=30)
        cbar.ax.tick_params(labelsize=9)

    # Start / Goal 마커
    ax.scatter(positions[start_idx, 0], positions[start_idx, 1],
               marker="o", s=150, color="#2ca02c", label="Start", zorder=8)
    ax.scatter(positions[goal_idx, 0], positions[goal_idx, 1],
               marker="X", s=170, color="#d62728", label="Goal", zorder=8)
    final_node = worker_path[-1]
    if final_node not in {start_idx, goal_idx}:
        ax.scatter(positions[final_node, 0], positions[final_node, 1],
                   marker="D", s=90, color="#1f77b4" if success else "#9467bd", label="Final node", zorder=8)

    status = "SUCCESS" if success else "FAIL"
    ax.set_title(f"Rollout Map - {checkpoint_label} ({status})", fontsize=16, fontweight="bold")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(frameon=True, framealpha=0.85, edgecolor="#cccccc", loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=9)
    ax.set_aspect("equal", adjustable="box")

    # 포커스 영역 계산
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
    """스텝별 Top-K 확률 분포 바 차트 그리드."""
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
                colors.append("#d62728" if warning_level == "high" else "#ff7f0e")
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

        # 경고 요약 텍스트
        if step.get("warning_summary"):
            ax.text(0.01, 0.98, step["warning_summary"], transform=ax.transAxes,
                    ha="left", va="top", fontsize=8,
                    color="#7f0000" if warning_level == "high" else "#6b4f00",
                    bbox={"boxstyle": "round,pad=0.25",
                          "facecolor": "#fff5f5" if warning_level == "high" else "#fff8e8",
                          "edgecolor": "#d62728" if warning_level == "high" else "#ffbf00",
                          "alpha": 0.95})

        # 바이어스 정보
        if "selected_bias_total" in step:
            ax.text(0.01, 0.02,
                    (f"bias total={step.get('selected_bias_total', 0.0):+.2f}, "
                     f"corr={step.get('selected_bias_corridor', 0.0):+.2f}, "
                     f"prog={step.get('selected_bias_progress', 0.0):+.2f}, "
                     f"det={step.get('selected_bias_detour', 0.0):+.2f}, "
                     f"nonprog={step.get('selected_bias_nonprogress', 0.0):+.2f}, "
                     f"eos={step.get('selected_bias_eos', 0.0):+.2f}"),
                    transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5, color="#333333",
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": "#f6f6f6",
                          "edgecolor": "#c7c7c7", "alpha": 0.95})

    for ax in axes.flat[len(steps):]:
        ax.axis("off")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_worker_trace_summary(output_path: Path, worker_steps: list[dict[str, Any]]) -> None:
    """Worker 추적 요약 — 확률/거리/Critic/탐색 4패널 + 개별 차트."""
    if not worker_steps:
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        ax.text(0.5, 0.5, "No worker trace data", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    df = pd.DataFrame(worker_steps)
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

    # 합본 대시보드
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
    """Manager 스텝별 진단 요약 — 4패널 차트."""
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

    # 진단 텍스트 패널
    axes[1, 1].axis("off")
    flagged_steps = diagnosis.get("flagged_steps", [])
    reason_counts = diagnosis.get("reason_counts", {})
    summary_lines = [
        f"Flagged steps: {diagnosis.get('flagged_count', 0)}",
        f"Max warning score: {diagnosis.get('max_warning_score', 0)}",
        f"Avg selected bias: {df['selected_bias_total'].mean():+.2f}" if "selected_bias_total" in df.columns else "Avg selected bias: n/a",
        (f"Bias mix (corr/prog/det/nonprog/eos): "
         f"{df['selected_bias_corridor'].mean():+.2f} / "
         f"{df['selected_bias_progress'].mean():+.2f} / "
         f"{df['selected_bias_detour'].mean():+.2f} / "
         f"{df['selected_bias_nonprogress'].mean():+.2f} / "
         f"{df['selected_bias_eos'].mean():+.2f}")
        if {"selected_bias_corridor", "selected_bias_progress", "selected_bias_detour", "selected_bias_nonprogress", "selected_bias_eos"}.issubset(df.columns) else "Bias mix: n/a",
        "",
        "Top suspicious steps:",
    ]
    if flagged_steps:
        for item in flagged_steps[:5]:
            reasons = "; ".join(item.get("warning_reasons", [])[:2]) or "No details"
            summary_lines.append(f"- step {item['step']} -> {item['selected_node']} ({item['warning_level']}): {reasons}")
    else:
        summary_lines.append("- none")
    summary_lines.extend(["", "Most common warning reasons:"])
    if reason_counts:
        for reason, count in sorted(reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]:
            summary_lines.append(f"- {count}x {reason}")
    else:
        summary_lines.append("- none")
    axes[1, 1].text(0.02, 0.98, "\n".join(summary_lines), ha="left", va="top",
                    fontsize=10, family="monospace", transform=axes[1, 1].transAxes)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_step_panels(
    output_dir: Path,
    manager_steps: list[dict[str, Any]],
    worker_steps: list[dict[str, Any]],
    max_worker_step_plots: int,
) -> None:
    """Manager/Worker 스텝별 개별 플롯을 각각의 서브디렉토리에 저장한다."""
    manager_dir = output_dir / "manager_steps"
    worker_dir = output_dir / "worker_steps"
    manager_dir.mkdir(parents=True, exist_ok=True)
    worker_dir.mkdir(parents=True, exist_ok=True)

    for step in manager_steps:
        plot_step_grid(manager_dir / f"manager_step_{step['step']:02d}.png", [step], f"Manager Step {step['step']}", "Mgr", 1)
    for step in worker_steps[:max_worker_step_plots]:
        plot_step_grid(worker_dir / f"worker_step_{step['step']:03d}.png", [step], f"Worker Step {step['step']}", "Wkr", 1)


# ============================================================
# 4. 궤적 시각화 프리미티브 (gen_paper_outputs.py에서 흡수)
# ============================================================

def draw_map_background(ax: plt.Axes, env: Any, positions: np.ndarray) -> None:
    """도로 네트워크 배경 렌더링."""
    for src, dst in env.map_core.graph.edges():
        u, v = env.node_mapping[src], env.node_mapping[dst]
        ax.plot([positions[u, 0], positions[v, 0]], [positions[u, 1], positions[v, 1]],
                color="#6b7280", linewidth=0.6, alpha=0.6)
    ax.set_aspect("equal")
    ax.tick_params(axis='both', labelsize=16)


def draw_endpoints(
    ax: plt.Axes,
    positions: np.ndarray,
    start_idx: int,
    goal_idx: int,
) -> None:
    """출발점/목적지 마커 렌더링."""
    ax.scatter(positions[start_idx, 0], positions[start_idx, 1],
               marker="o", s=450, color="#22c55e", edgecolors="black",
               linewidths=2.0, zorder=10, label="Start")
    ax.scatter(positions[goal_idx, 0], positions[goal_idx, 1],
               marker="X", s=500, color="#ef4444", edgecolors="black",
               linewidths=2.0, zorder=10, label="Goal")


def draw_path_gradient(
    ax: plt.Axes,
    positions: np.ndarray,
    path: list[int],
    cmap_name: str = "autumn",
    linewidth: float = 3.0,
) -> Optional[cm.ScalarMappable]:
    """경로를 그라데이션으로 렌더링하고 ScalarMappable을 반환한다."""
    if len(path) < 2:
        return None
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=0, vmax=max(len(path) - 2, 1))
    for i in range(len(path) - 1):
        ax.plot([positions[path[i], 0], positions[path[i + 1], 0]],
                [positions[path[i], 1], positions[path[i + 1], 1]],
                color=cmap(norm(i)), linewidth=linewidth, alpha=0.85)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    return sm


def draw_astar_path(
    ax: plt.Axes,
    positions: np.ndarray,
    astar_path: Optional[list[int]],
) -> None:
    """A* 최적 경로를 파선으로 렌더링."""
    if astar_path and len(astar_path) > 1:
        a_arr = np.array(astar_path)
        ax.plot(positions[a_arr, 0], positions[a_arr, 1],
                "--", color="#3b82f6", linewidth=2.5, alpha=0.7, label="A* Optimal")


def generate_latency_chart(
    results: Dict[str, Dict[str, Any]],
    output_dir: Path,
) -> None:
    """4가지 방식의 추론 지연시간 비교 바 차트를 생성한다."""
    methods = list(results.keys())
    latencies = [results[m]["inference_latency_ms"] for m in methods]
    colors = [COLORS.get(m, "#888888") for m in methods]

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    bars = ax.bar(methods, latencies, color=colors, edgecolor="black", linewidth=0.5)

    for bar, lat in zip(bars, latencies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(latencies) * 0.02,
                f"{lat:.1f}ms", ha="center", va="bottom", fontsize=13, fontweight="bold")

    ax.set_ylabel("Latency (ms)", fontsize=18, fontweight="bold")
    ax.set_xlabel("Method", fontsize=18, fontweight="bold")
    ax.tick_params(axis="both", labelsize=15)
    ax.set_ylim(bottom=0)

    fig.savefig(output_dir / "inference_latency.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved: inference_latency.png")
