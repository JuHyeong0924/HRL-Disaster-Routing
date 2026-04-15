"""논문용 통합 산출물 생성 스크립트.

A*, Flat RL, HRL(Greedy), HRL(Multi-Sampling N=16) 4가지 방식의
성능 평가, 궤적 시각화, 추론 지연시간 비교, 학습 곡선을 일괄 생성한다.

Usage:
    python tests/gen_paper_outputs.py [--episodes 500] [--checkpoint <path>]
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import torch
import networkx as nx

# 프로젝트 루트 등록
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tests.eval_core as core

# ── 전역 상수 ──────────────────────────────────────────────
DEFAULT_EPISODES: int = 500
DEFAULT_NUM_SAMPLES: int = 16
DEFAULT_SAMPLE_TEMP: float = 1.0
MIN_HOPS_VIZ: int = 50   # 궤적 시각화용 최소 hop (Goldcoast 기준)
MIN_HOPS_EVAL: int = 2    # 배치 평가용 최소 hop

# 맵 설정 (Latency 벤치마크 및 평가용)
MAPS_CONFIG: List[Dict[str, Any]] = [
    {"name": "SiouxFalls", "nodes": 24},
    {"name": "Anaheim", "nodes": 416},
    {"name": "ChicagoSketch", "nodes": 933},
    {"name": "Goldcoast", "nodes": 4807},
]

# 스타일 설정
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# 색상 팔레트
COLORS = {
    "A*": "#6366f1",            # 인디고
    "Flat RL": "#f59e0b",       # 앰버
    "HRL (Greedy)": "#10b981",  # 에메랄드
    "HRL (Sampling)": "#ef4444", # 레드
    "Worker Only": "#3b82f6",    # 학습 곡선용 파란색
    "Proposed HRL": "#10b981",   # 학습 곡선용 초록색
}


# ============================================================
# 1. Multi-Sampling 롤아웃
# ============================================================

def run_joint_rollout_multisampling(
    worker: Any,
    manager: Any,
    env: Any,
    start_idx: int,
    goal_idx: int,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    temperature: float = DEFAULT_SAMPLE_TEMP,
    measure_time: bool = False,
) -> Dict[str, Any]:
    """N번의 확률적 Joint 롤아웃을 수행하고, 성공한 경로 중 최단 거리를 채택한다.

    Why: GPU 병렬 특성을 활용하여 단일 추론 대비 성공률과 PLR을 극적으로 개선.
    """
    results: List[Dict[str, Any]] = []
    total_time = 0.0

    t0 = time.perf_counter()
    for _ in range(num_samples):
        result = core.run_joint_rollout(
            worker, manager, env, start_idx, goal_idx,
            temperature=temperature,
            mgr_temperature=0.5,
            measure_time=measure_time,
        )
        results.append(result)
    if measure_time:
        if env.device.type == "cuda":
            torch.cuda.synchronize()
        total_time = (time.perf_counter() - t0) * 1000.0

    # 성공한 경로 중 물리적 거리가 가장 짧은 것 선택
    successes = [r for r in results if r["success"]]
    if successes:
        best = min(successes, key=lambda r: r.get("path_distance", float("inf")))
    else:
        # 전부 실패 시 progress가 높은 것 반환
        best = max(results, key=lambda r: r.get("progress", 0.0))

    # 메타 정보 추가
    best["num_samples"] = num_samples
    best["num_successes"] = len(successes)
    best["sampling_success_rate"] = len(successes) / num_samples * 100.0
    if measure_time:
        best["total_sampling_time_ms"] = total_time
        best["per_sample_time_ms"] = total_time / num_samples

    return best


# ============================================================
# 2. A* 최적 경로 복원 (BFS)
# ============================================================

def reconstruct_astar_path(
    env: Any,
    start_idx: int,
    goal_idx: int,
) -> Optional[List[int]]:
    """BFS로 최단 홉 경로를 복원한다."""
    try:
        queue = deque([(start_idx, [start_idx])])
        visited = {start_idx}
        while queue:
            node, path_so_far = queue.popleft()
            if node == goal_idx:
                return path_so_far
            # 내부 노드 매핑을 통해 이웃 노드를 탐색
            raw_key = [k for k, v in env.node_mapping.items() if v == node][0]
            for nb_raw in env.map_core.graph.neighbors(raw_key):
                nb = env.node_mapping[nb_raw]
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, path_so_far + [nb]))
    except Exception:
        pass
    return None


def compute_path_distance(env: Any, path: List[int]) -> float:
    """경로의 물리적 거리를 합산한다."""
    if len(path) < 2:
        return 0.0
    return sum(
        float(env.apsp_matrix[path[i], path[i + 1]].item())
        for i in range(len(path) - 1)
    )


# ============================================================
# 3. OD 쌍 탐색 (긴 거리용)
# ============================================================

def find_long_od_pair(
    env: Any,
    min_hops: int = MIN_HOPS_VIZ,
    seed: int = 42,
    max_attempts: int = 5000,
) -> Tuple[int, int]:
    """hop distance가 큰 OD 쌍을 탐색한다. 시각적으로 경로가 잘 펼쳐지는 쌍 우선."""
    rng = random.Random(seed)
    best_pair = (0, 1)
    best_spread = 0.0
    positions = env.pos_tensor.cpu().numpy()

    for _ in range(max_attempts):
        s = rng.randint(0, env.num_nodes - 1)
        g = rng.randint(0, env.num_nodes - 1)
        if s == g:
            continue
        hd = float(env.hop_matrix[s, g].item())
        if not math.isfinite(hd) or hd < min_hops:
            continue
        # 유클리드 거리 기반 시각적 스프레드 계산
        spread = float(np.linalg.norm(positions[s] - positions[g]))
        if spread > best_spread:
            best_spread = spread
            best_pair = (s, g)

    return best_pair


# ============================================================
# 4. 궤적 시각화
# ============================================================

def draw_map_background(ax: plt.Axes, env: Any, positions: np.ndarray) -> None:
    """도로 네트워크 배경 렌더링."""
    for src, dst in env.map_core.graph.edges():
        u, v = env.node_mapping[src], env.node_mapping[dst]
        ax.plot(
            [positions[u, 0], positions[v, 0]],
            [positions[u, 1], positions[v, 1]],
            color="#6b7280", linewidth=0.6, alpha=0.6,
        )
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
    path: List[int],
    cmap_name: str = "autumn",
    linewidth: float = 3.0,
) -> Optional[plt.cm.ScalarMappable]:
    """경로를 그라데이션으로 렌더링하고 ScalarMappable을 반환한다."""
    if len(path) < 2:
        return None
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=0, vmax=max(len(path) - 2, 1))
    for i in range(len(path) - 1):
        ax.plot(
            [positions[path[i], 0], positions[path[i + 1], 0]],
            [positions[path[i], 1], positions[path[i + 1], 1]],
            color=cmap(norm(i)), linewidth=linewidth, alpha=0.85,
        )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    return sm


def draw_astar_path(
    ax: plt.Axes,
    positions: np.ndarray,
    astar_path: Optional[List[int]],
) -> None:
    """A* 최적 경로를 파선으로 렌더링."""
    if astar_path and len(astar_path) > 1:
        a_arr = np.array(astar_path)
        ax.plot(positions[a_arr, 0], positions[a_arr, 1],
                "--", color="#3b82f6", linewidth=2.5, alpha=0.7, label="A* Optimal")


def generate_trajectory_figures(
    env: Any,
    flat_worker: Any,
    joint_worker: Any,
    manager: Any,
    start_idx: int,
    goal_idx: int,
    output_dir: Path,
    num_samples: int = DEFAULT_NUM_SAMPLES,
) -> Dict[str, Any]:
    """Flat RL + HRL(Multi-Sampling) 궤적 시각화 3장 생성."""
    positions = env.pos_tensor.detach().cpu().numpy()

    # A* 경로 복원
    astar_path = reconstruct_astar_path(env, start_idx, goal_idx)
    optimal_dist = float(env.apsp_matrix[start_idx, goal_idx].item())
    optimal_hops = float(env.hop_matrix[start_idx, goal_idx].item())

    # Flat RL 롤아웃 (Worker-Only 체크포인트 사용)
    flat_result = core.run_worker_rollout(
        flat_worker, env, start_idx, goal_idx, temperature=0.0,
    )

    # HRL Multi-Sampling 롤아웃 (Joint 체크포인트 사용)
    hrl_result = run_joint_rollout_multisampling(
        joint_worker, manager, env, start_idx, goal_idx,
        num_samples=num_samples, temperature=DEFAULT_SAMPLE_TEMP,
    )
    hrl_path = hrl_result["path"]
    hrl_subgoals = hrl_result.get("plan_subgoals", [])

    print(f"\n📍 OD: start={start_idx}, goal={goal_idx}")
    print(f"   Optimal: {optimal_hops:.0f} hops, {optimal_dist:.1f} km")
    print(f"   Flat RL: {'✅' if flat_result['success'] else '❌'}, "
          f"steps={flat_result['actual_steps']}, PLR={flat_result['path_length_ratio']:.3f}")
    print(f"   HRL(N={num_samples}): {'✅' if hrl_result['success'] else '❌'}, "
          f"steps={hrl_result['actual_steps']}, PLR={hrl_result['path_length_ratio']:.3f}, "
          f"sampling_sr={hrl_result['sampling_success_rate']:.1f}%")

    # ── (a) Flat RL ──
    fig_a, ax_a = plt.subplots(figsize=(14, 11), constrained_layout=True)
    draw_map_background(ax_a, env, positions)
    draw_astar_path(ax_a, positions, astar_path)
    sm = draw_path_gradient(ax_a, positions, flat_result["path_nodes"], cmap_name="autumn")
    if sm:
        cbar = fig_a.colorbar(sm, ax=ax_a, orientation='horizontal', fraction=0.046, pad=0.06)
        cbar.set_label('Trajectory Progress (Steps)', fontsize=16, fontweight='bold')
        cbar.ax.tick_params(labelsize=14)
    draw_endpoints(ax_a, positions, start_idx, goal_idx)
    ax_a.legend(fontsize=17, loc="upper right")
    fig_a.savefig(output_dir / "flat_rl_trajectory.png", dpi=300, bbox_inches="tight")
    plt.close(fig_a)
    print(f"  ✅ Saved: flat_rl_trajectory.png")

    # ── (b) HRL + Subgoal 마커 ──
    fig_b, ax_b = plt.subplots(figsize=(14, 11), constrained_layout=True)
    draw_map_background(ax_b, env, positions)
    draw_astar_path(ax_b, positions, astar_path)
    sm_h = draw_path_gradient(ax_b, positions, hrl_path, cmap_name="cool")
    # Subgoal 마커
    if hrl_subgoals:
        sg_valid = [sg for sg in hrl_subgoals if sg < env.num_nodes]
        if sg_valid:
            sg_arr = np.array(sg_valid)
            ax_b.scatter(positions[sg_arr, 0], positions[sg_arr, 1],
                         marker="*", s=600, facecolors="#fbbf24", edgecolors="#92400e",
                         linewidths=1.5, zorder=8, label="Subgoals")
            for idx, sg in enumerate(sg_arr):
                ax_b.text(positions[sg, 0] + 0.001, positions[sg, 1] + 0.001,
                          f"$S_{{{idx+1}}}$", fontsize=16, fontweight="bold",
                          bbox=dict(facecolor="white", edgecolor="#fbbf24",
                                    boxstyle="round,pad=0.2", alpha=0.9))
    if sm_h:
        cbar_h = fig_b.colorbar(sm_h, ax=ax_b, orientation='horizontal', fraction=0.046, pad=0.06)
        cbar_h.set_label('Trajectory Progress (Steps)', fontsize=16, fontweight='bold')
        cbar_h.ax.tick_params(labelsize=14)
    draw_endpoints(ax_b, positions, start_idx, goal_idx)
    ax_b.legend(fontsize=17, loc="upper right")
    fig_b.savefig(output_dir / "hrl_trajectory_with_subgoals.png", dpi=300, bbox_inches="tight")
    plt.close(fig_b)
    print(f"  ✅ Saved: hrl_trajectory_with_subgoals.png")

    # ── (c) HRL Clean (Subgoal 미포함) ──
    fig_c, ax_c = plt.subplots(figsize=(14, 11), constrained_layout=True)
    draw_map_background(ax_c, env, positions)
    draw_astar_path(ax_c, positions, astar_path)
    sm_c = draw_path_gradient(ax_c, positions, hrl_path, cmap_name="cool")
    if sm_c:
        cbar_c = fig_c.colorbar(sm_c, ax=ax_c, orientation='horizontal', fraction=0.046, pad=0.06)
        cbar_c.set_label('Trajectory Progress (Steps)', fontsize=16, fontweight='bold')
        cbar_c.ax.tick_params(labelsize=14)
    draw_endpoints(ax_c, positions, start_idx, goal_idx)
    ax_c.legend(fontsize=17, loc="upper right")
    fig_c.savefig(output_dir / "hrl_trajectory_clean.png", dpi=300, bbox_inches="tight")
    plt.close(fig_c)
    print(f"  ✅ Saved: hrl_trajectory_clean.png")

    # A* 경로 거리
    astar_dist = compute_path_distance(env, astar_path) if astar_path else optimal_dist

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


# ============================================================
# 5. 성과 지표 TXT 생성
# ============================================================

def generate_trajectory_metrics_txt(
    traj_info: Dict[str, Any],
    output_dir: Path,
    env: Any,
) -> None:
    """궤적 비교 성과 지표를 TXT 파일로 저장한다."""
    flat = traj_info["flat_result"]
    hrl = traj_info["hrl_result"]
    opt_dist = traj_info["optimal_dist"]
    opt_hops = traj_info["optimal_hops"]

    flat_dist = flat.get("path_distance", 0.0)
    hrl_dist = hrl.get("path_distance", 0.0)

    summary = (
        f"================================================================\n"
        f"Trajectory Comparison on Goldcoast ({env.num_nodes} nodes)\n"
        f"================================================================\n"
        f"\n"
        f"[Problem Setting]\n"
        f"  Map:              Goldcoast (Australia)\n"
        f"  Nodes:            {env.num_nodes}\n"
        f"  Start Node:       {traj_info['start']}\n"
        f"  Goal Node:        {traj_info['goal']}\n"
        f"  Optimal Hops:     {opt_hops:.0f}\n"
        f"  Optimal Distance: {opt_dist:.1f} km\n"
        f"\n"
        f"[Results]\n"
        f"  {'Metric':<30s} {'A*':>12s} {'Flat RL':>12s} {'HRL(N=16)':>12s}\n"
        f"  {'-'*66}\n"
        f"  {'Success':<30s} {'Yes':>12s} {'Yes' if flat['success'] else 'No':>12s} {'Yes' if hrl['success'] else 'No':>12s}\n"
        f"  {'Actual Steps':<30s} {traj_info['astar_steps']:>12.0f} {flat['actual_steps']:>12d} {hrl['actual_steps']:>12d}\n"
        f"  {'Path Distance (km)':<30s} {traj_info['astar_dist']:>12.1f} {flat_dist:>12.1f} {hrl_dist:>12.1f}\n"
        f"  {'PLR (Distance Ratio)':<30s} {'1.000':>12s} {flat['path_length_ratio']:>12.3f} {hrl['path_length_ratio']:>12.3f}\n"
        f"  {'Subgoals':<30s} {'N/A':>12s} {'N/A':>12s} {len(hrl.get('plan_subgoals', [])):>12d}\n"
        f"  {'Sampling Success Rate':<30s} {'N/A':>12s} {'N/A':>12s} {hrl.get('sampling_success_rate', 0):>11.1f}%\n"
        f"\n"
        f"[Improvement (HRL vs Flat RL)]\n"
    )

    if flat["success"] and hrl["success"]:
        plr_reduction = (flat["path_length_ratio"] - hrl["path_length_ratio"]) / flat["path_length_ratio"] * 100
        step_reduction = (flat["actual_steps"] - hrl["actual_steps"]) / flat["actual_steps"] * 100
        dist_reduction = (flat_dist - hrl_dist) / flat_dist * 100 if flat_dist > 0 else 0
        summary += (
            f"  PLR Reduction:      {flat['path_length_ratio']:.3f} -> {hrl['path_length_ratio']:.3f} ({plr_reduction:.1f}% decrease)\n"
            f"  Step Reduction:     {flat['actual_steps']} -> {hrl['actual_steps']} ({step_reduction:.1f}% decrease)\n"
            f"  Distance Reduction: {flat_dist:.1f} -> {hrl_dist:.1f} km ({dist_reduction:.1f}% decrease)\n"
        )
    else:
        summary += "  (비교 불가: 하나 이상의 방식이 실패)\n"

    summary += f"================================================================\n"

    path = output_dir / "trajectory_metrics.txt"
    path.write_text(summary, encoding="utf-8")
    print(f"  📝 Saved: trajectory_metrics.txt")


# ============================================================
# 6. 배치 평가 (4가지 방식)
# ============================================================

def evaluate_all_methods(
    flat_worker: Any,
    joint_worker: Any,
    manager: Any,
    env: Any,
    num_episodes: int = DEFAULT_EPISODES,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """A*, Flat RL, HRL(Greedy), HRL(Multi-Sampling)를 일괄 평가한다."""
    print(f"\n{'='*60}")
    print(f"📊 배치 평가 시작 ({num_episodes} episodes)")
    print(f"{'='*60}")

    # OD 쌍 생성
    pairs = core.generate_od_pairs(env, num_episodes, min_hops=MIN_HOPS_EVAL, seed=seed)
    if len(pairs) < num_episodes:
        print(f"  ⚠️ {num_episodes}쌍 요청 중 {len(pairs)}쌍만 생성됨")

    results: Dict[str, Dict[str, Any]] = {}

    # (1) A* 벤치마크
    print(f"\n  🔷 A* (Dijkstra) 평가 중...")
    astar_results: List[Dict[str, Any]] = []
    astar_times: List[float] = []
    for s, g in pairs:
        t0 = time.perf_counter()
        try:
            raw_s = [k for k, v in env.node_mapping.items() if v == s][0]
            raw_g = [k for k, v in env.node_mapping.items() if v == g][0]
            nx_path = nx.dijkstra_path(env.map_core.graph, raw_s, raw_g, weight="length")
            mapped_path = [env.node_mapping[n] for n in nx_path]
            success = True
        except (nx.NetworkXNoPath, KeyError, IndexError):
            mapped_path = [s]
            success = False
        t1 = time.perf_counter()
        astar_times.append((t1 - t0) * 1000.0)
        dist = compute_path_distance(env, mapped_path)
        opt_dist = float(env.apsp_matrix[s, g].item())
        plr = dist / max(opt_dist, 1e-6) if success and opt_dist > 0 else float("inf")
        astar_results.append({"success": success, "path_length_ratio": plr, "path_distance": dist})

    astar_sr = sum(1 for r in astar_results if r["success"]) / len(astar_results) * 100
    astar_plrs = [r["path_length_ratio"] for r in astar_results if r["success"] and math.isfinite(r["path_length_ratio"])]
    results["A*"] = {
        "success_rate": astar_sr,
        "path_length_ratio": float(np.mean(astar_plrs)) if astar_plrs else float("inf"),
        "inference_latency_ms": float(np.mean(astar_times)),
        "per_step_latency_ms": float(np.mean(astar_times)),
    }
    print(f"     SR={astar_sr:.1f}%, PLR={results['A*']['path_length_ratio']:.3f}, "
          f"Latency={results['A*']['inference_latency_ms']:.2f}ms")

    # (2) Flat RL (Worker Only)
    print(f"\n  🔶 Flat RL 평가 중...")
    flat_eval = core.evaluate_worker_batch(
        flat_worker, env, len(pairs), temperature=0.0,
        label="Flat RL", seed=seed, min_hops=MIN_HOPS_EVAL,
    )
    results["Flat RL"] = {
        "success_rate": flat_eval["success_rate"],
        "path_length_ratio": flat_eval["path_length_ratio"],
        "inference_latency_ms": flat_eval["inference_latency_ms"],
        "per_step_latency_ms": flat_eval["per_step_latency_ms"],
    }
    print(f"     SR={flat_eval['success_rate']:.1f}%, PLR={flat_eval['path_length_ratio']:.3f}")

    # (3) HRL Greedy
    print(f"\n  🟢 HRL (Greedy) 평가 중...")
    hrl_greedy_eval = core.evaluate_joint_batch(
        joint_worker, manager, env, len(pairs), temperature=0.0,
        mgr_temperature=0.0, label="HRL Greedy", seed=seed, min_hops=MIN_HOPS_EVAL,
    )
    results["HRL (Greedy)"] = {
        "success_rate": hrl_greedy_eval["success_rate"],
        "path_length_ratio": hrl_greedy_eval["path_length_ratio"],
        "inference_latency_ms": hrl_greedy_eval["inference_latency_ms"],
        "per_step_latency_ms": hrl_greedy_eval["per_step_latency_ms"],
    }
    print(f"     SR={hrl_greedy_eval['success_rate']:.1f}%, PLR={hrl_greedy_eval['path_length_ratio']:.3f}")

    # (4) HRL Multi-Sampling
    print(f"\n  🔴 HRL (Multi-Sampling, N={num_samples}) 평가 중...")
    ms_successes = 0
    ms_plrs: List[float] = []
    ms_times: List[float] = []
    rng = random.Random(seed)
    for ep, (s, g) in enumerate(pairs):
        ms_result = run_joint_rollout_multisampling(
            joint_worker, manager, env, s, g,
            num_samples=num_samples, temperature=DEFAULT_SAMPLE_TEMP,
            measure_time=True,
        )
        if ms_result["success"]:
            ms_successes += 1
            plr_val = ms_result["path_length_ratio"]
            if math.isfinite(plr_val):
                ms_plrs.append(plr_val)
        ms_times.append(ms_result.get("total_sampling_time_ms", 0.0))

        if (ep + 1) % 50 == 0:
            sr = ms_successes / (ep + 1) * 100
            print(f"     [{ep+1}/{len(pairs)}] SR={sr:.1f}%")

    ms_sr = ms_successes / len(pairs) * 100
    results["HRL (Sampling)"] = {
        "success_rate": ms_sr,
        "path_length_ratio": float(np.mean(ms_plrs)) if ms_plrs else float("inf"),
        "inference_latency_ms": float(np.mean(ms_times)),
        "per_step_latency_ms": float(np.mean(ms_times)) / num_samples,
    }
    print(f"     SR={ms_sr:.1f}%, PLR={results['HRL (Sampling)']['path_length_ratio']:.3f}")

    return results


# ============================================================
# 7. 배치 평가 결과 TXT 저장
# ============================================================

def save_all_maps_eval_txt(
    all_results: Dict[str, Dict[str, Dict[str, Any]]],
    num_episodes: int,
    output_dir: Path,
) -> None:
    """전체 맵 배치 평가 결과를 하나의 텍스트 파일로 저장한다."""
    lines = [
        f"================================================================",
        f"Batch Evaluation Results — All Maps",
        f"Episodes per map: {num_episodes}",
        f"================================================================",
    ]
    for map_name, results in all_results.items():
        lines.append(f"")
        lines.append(f"  📍 {map_name}")
        lines.append(f"  {'Method':<25s} {'SR (%)':>10s} {'PLR':>10s} {'Latency (ms)':>15s}")
        lines.append(f"  {'-'*60}")
        for method, data in results.items():
            sr = f"{data['success_rate']:.1f}"
            plr = f"{data['path_length_ratio']:.3f}" if math.isfinite(data['path_length_ratio']) else "inf"
            lat = f"{data['inference_latency_ms']:.2f}"
            lines.append(f"  {method:<25s} {sr:>10s} {plr:>10s} {lat:>15s}")
    lines.append(f"================================================================")

    path = output_dir / "batch_eval_results.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  📝 Saved: batch_eval_results.txt")


# ============================================================
# 8. Inference Latency 차트
# ============================================================

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

    # 값 표시
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


# ============================================================
# 9. 학습 곡선 생성 (Worker Only vs HRL)
# ============================================================

def generate_learning_curves(output_dir: Path) -> None:
    """학습 곡선 차트를 생성한다. HRL의 success_ema에 ×100 보정 적용."""
    # CSV 경로 탐색
    worker_csv = Path("logs/rl_worker_stage/2026-04-13_1923_worker_pomo32/debug_metrics.csv")
    joint_csv = Path("logs/rl_joint_stage/2026-04-14_2020_joint_pomo16/debug_metrics.csv")

    if not worker_csv.exists() or not joint_csv.exists():
        print(f"  ⚠️ 학습 곡선 CSV를 찾을 수 없습니다. 건너뜁니다.")
        print(f"     Worker: {worker_csv} ({'✅' if worker_csv.exists() else '❌'})")
        print(f"     Joint:  {joint_csv} ({'✅' if joint_csv.exists() else '❌'})")
        return

    dw = pd.read_csv(worker_csv)
    dj = pd.read_csv(joint_csv)

    w_x = dw["episode"]
    j_x = dj["ep"]

    def smooth(series: pd.Series, frac: float = 0.05) -> pd.Series:
        win = max(int(len(series) * frac), 1)
        return series.rolling(win, min_periods=1).mean()

    # ── Success Rate ──
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    # HRL success_ema는 0-1 범위이므로 ×100 보정
    j_sr = dj["success_ema"] * 100
    ax.plot(w_x, dw["success_ema"], color=COLORS["Worker Only"], alpha=0.12)
    ax.plot(w_x, smooth(dw["success_ema"]), color=COLORS["Worker Only"], linewidth=2.5, label="Worker Only")
    ax.plot(j_x, j_sr, color=COLORS["Proposed HRL"], alpha=0.12)
    ax.plot(j_x, smooth(j_sr), color=COLORS["Proposed HRL"], linewidth=2.5, label="Proposed HRL")
    ax.set_ylabel("Success Rate (EMA, %)", fontsize=18, fontweight="bold")
    ax.set_xlabel("Episode", fontsize=18)
    ax.tick_params(axis="both", labelsize=15)
    ax.legend(fontsize=13, framealpha=0.9)
    fig.savefig(output_dir / "success_rate.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved: success_rate.png")

    # ── Loss ──
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(w_x, dw["loss"], color=COLORS["Worker Only"], alpha=0.12)
    ax.plot(w_x, smooth(dw["loss"]), color=COLORS["Worker Only"], linewidth=2.5, label="Worker Only")
    ax.plot(j_x, dj["loss_mean"], color=COLORS["Proposed HRL"], alpha=0.12)
    ax.plot(j_x, smooth(dj["loss_mean"]), color=COLORS["Proposed HRL"], linewidth=2.5, label="Proposed HRL")
    ax.set_ylabel("Policy Loss", fontsize=18, fontweight="bold")
    ax.set_xlabel("Episode", fontsize=18)
    ax.tick_params(axis="both", labelsize=15)
    ax.legend(fontsize=13, framealpha=0.9)
    fig.savefig(output_dir / "loss.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved: loss.png")


# ============================================================
# 10. 메인 진입점
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="논문용 통합 산출물 생성")
    parser.add_argument("--joint-ckpt", type=str,
                        default="logs/rl_joint_stage/2026-04-15_0545_joint_pomo16/best.pt",
                        help="Joint RL 체크포인트 경로 (Manager+Worker)")
    parser.add_argument("--worker-ckpt", type=str,
                        default="logs/rl_worker_stage/2026-04-13_1923_worker_pomo32/final.pt",
                        help="Worker-Only RL 체크포인트 경로 (Flat RL용)")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES,
                        help="배치 평가 에피소드 수 (기본: 500)")
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES,
                        help="Multi-Sampling N (기본: 16)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path("tests/paper_figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 디바이스 설정: CUDA 우선
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    core.set_seed(args.seed)
    print(f"🖥️ Device: {device}")
    print(f"📂 Joint 체크포인트: {args.joint_ckpt}")
    print(f"📂 Worker 체크포인트: {args.worker_ckpt}")
    print(f"📊 평가 에피소드: {args.episodes}")
    print(f"🎲 Multi-Sampling N: {args.num_samples}")

    # ── 모델 로드 ──
    # Joint 체크포인트에서 Manager + Worker 로드 (HRL용)
    print(f"\n📦 Joint 모델 로드 중... ({args.joint_ckpt})")
    joint_worker, manager, _ = core.load_checkpoint(
        args.joint_ckpt, device=device, load_manager=True,
    )
    if manager is None:
        raise RuntimeError("Joint 체크포인트에서 Manager를 로드할 수 없습니다.")
    print(f"  ✅ Joint Worker + Manager 로드 완료")

    # Worker-Only 체크포인트에서 Worker 로드 (Flat RL용)
    print(f"📦 Worker-Only 모델 로드 중... ({args.worker_ckpt})")
    flat_worker, _, _ = core.load_checkpoint(
        args.worker_ckpt, device=device, load_manager=False,
    )
    print(f"  ✅ Flat Worker 로드 완료")

    # ── Step 1: 전체 맵 배치 평가 (SiouxFalls → Anaheim → ChicagoSketch → Goldcoast) ──
    # Goldcoast를 마지막에 하면 env를 궤적 시각화에 재활용 가능 (APSP 1회만 계산)
    all_map_results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for map_cfg in MAPS_CONFIG:
        map_name = map_cfg["name"]
        print(f"\n{'='*60}")
        print(f"🔬 배치 평가: {map_name} ({map_cfg['nodes']} nodes)")
        print(f"{'='*60}")
        env_map = core.setup_env(map_name, device)
        map_results = evaluate_all_methods(
            flat_worker, joint_worker, manager, env_map,
            num_episodes=args.episodes,
            num_samples=args.num_samples,
            seed=args.seed,
        )
        all_map_results[map_name] = map_results

        # Goldcoast 평가 직후: 같은 env를 재활용하여 궤적 시각화
        if map_name == "Goldcoast":
            print(f"\n{'='*60}")
            print(f"🎨 궤적 시각화 (Goldcoast, APSP 재활용)")
            print(f"{'='*60}")
            start_idx, goal_idx = find_long_od_pair(env_map, min_hops=MIN_HOPS_VIZ, seed=args.seed)
            traj_info = generate_trajectory_figures(
                env_map, flat_worker, joint_worker, manager, start_idx, goal_idx,
                output_dir, num_samples=args.num_samples,
            )
            generate_trajectory_metrics_txt(traj_info, output_dir, env_map)

    # ── Step 2: 전체 맵 결과 TXT 저장 ──
    print(f"\n{'='*60}")
    print(f"📝 전체 맵 평가 결과 저장")
    print(f"{'='*60}")
    save_all_maps_eval_txt(all_map_results, args.episodes, output_dir)

    # ── Step 3: Inference Latency 차트 (Anaheim 기준) ──
    print(f"\n{'='*60}")
    print(f"⏱️ Inference Latency 차트")
    print(f"{'='*60}")
    if "Anaheim" in all_map_results:
        generate_latency_chart(all_map_results["Anaheim"], output_dir)
    else:
        first_map = next(iter(all_map_results))
        generate_latency_chart(all_map_results[first_map], output_dir)

    # ── Step 4: 학습 곡선 ──
    print(f"\n{'='*60}")
    print(f"📈 학습 곡선")
    print(f"{'='*60}")
    generate_learning_curves(output_dir)

    # ── 최종 요약 ──
    print(f"\n{'='*60}")
    print(f"🎉 모든 산출물 생성 완료!")
    print(f"{'='*60}")
    print(f"\n📂 출력 디렉토리: {output_dir.resolve()}")
    print(f"\n📊 전체 맵 평가 요약:")
    for map_name, map_results in all_map_results.items():
        print(f"\n  📍 {map_name}:")
        print(f"    {'Method':<25s} {'SR (%)':>8s} {'PLR':>8s}")
        print(f"    {'-'*41}")
        for method, data in map_results.items():
            sr = f"{data['success_rate']:.1f}"
            plr = f"{data['path_length_ratio']:.3f}" if math.isfinite(data['path_length_ratio']) else "inf"
            print(f"    {method:<25s} {sr:>8s} {plr:>8s}")


if __name__ == "__main__":
    main()
