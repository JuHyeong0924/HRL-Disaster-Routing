"""
summarize_ablation.py — Ablation Study 결과 집계 및 시각화
tests/ablation_results/의 모든 실험 결과를 분석하여 비교표와 차트를 생성한다.

사용법:
  python tests/summarize_ablation.py
"""
import os
import sys
import json
import csv
import re
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ablation_configs import ABLATION_REGISTRY, ALL_EXPERIMENT_IDS


RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ablation_results")


def parse_train_log(log_path: str) -> dict:
    """train_log.txt에서 최종 EMA, Loss 등을 추출한다."""
    result = {
        "final_ema": None,
        "final_loss": None,
        "final_succ": None,
        "ema_history": [],
    }
    if not os.path.exists(log_path):
        return result

    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        # 200ep마다 출력되는 DEBUG_UPDATE 파싱
        if "EMA:" in line or "EMA=" in line:
            # EMA 값 추출
            ema_match = re.search(r"EMA[=:]\s*([\d.]+)%?", line)
            if ema_match:
                ema_val = float(ema_match.group(1))
                result["ema_history"].append(ema_val)
                result["final_ema"] = ema_val

        if "Loss=" in line:
            loss_match = re.search(r"Loss=([\d.]+)", line)
            if loss_match:
                result["final_loss"] = float(loss_match.group(1))

        if "Succ=" in line:
            succ_match = re.search(r"Succ=([\d.]+)%?", line)
            if succ_match:
                result["final_succ"] = float(succ_match.group(1))

    return result


def parse_debug_csv(csv_path: str) -> dict:
    """debug_metrics.csv에서 주요 지표를 추출한다."""
    result = {
        "success_rates": [],
        "entropies": [],
        "episodes": [],
    }
    if not os.path.exists(csv_path):
        return result

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "success_rate" in row and row["success_rate"]:
                result["success_rates"].append(float(row["success_rate"]))
            if "worker_entropy" in row and row["worker_entropy"]:
                result["entropies"].append(float(row["worker_entropy"]))
            if "episode" in row and row["episode"]:
                result["episodes"].append(int(row["episode"]))

    return result


def collect_all_results() -> dict:
    """모든 실험 결과를 수집한다."""
    results = {}

    for exp_id in ALL_EXPERIMENT_IDS:
        exp_dir = os.path.join(RESULTS_DIR, exp_id)
        if not os.path.isdir(exp_dir):
            continue

        # 메타데이터
        meta_path = os.path.join(exp_dir, "experiment_meta.json")
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

        # 학습 로그 파싱
        log_data = parse_train_log(os.path.join(exp_dir, "train_log.txt"))

        # CSV 파싱
        csv_data = parse_debug_csv(os.path.join(exp_dir, "debug_metrics.csv"))

        config = ABLATION_REGISTRY.get(exp_id, {})
        results[exp_id] = {
            "description": config.get("description", ""),
            "category": "Architecture" if exp_id.startswith("A") else
                        "State" if exp_id.startswith("S") else
                        "Reward" if exp_id.startswith("R") else "Baseline",
            "final_ema": log_data["final_ema"],
            "final_loss": log_data["final_loss"],
            "ema_history": log_data["ema_history"],
            "success_rates": csv_data["success_rates"],
            "entropies": csv_data["entropies"],
            "episodes": csv_data["episodes"],
            "meta": meta,
        }

    return results


def print_comparison_table(results: dict) -> None:
    """콘솔에 비교표를 출력한다."""
    print(f"\n{'='*80}")
    print(f"📊 Worker Ablation Study 결과 비교")
    print(f"{'='*80}")

    # 헤더
    print(f"{'ID':>10} | {'Category':>12} | {'Final EMA':>10} | {'Final Loss':>10} | {'Description'}")
    print(f"{'-'*10}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*40}")

    baseline_ema = results.get("BASELINE", {}).get("final_ema")

    for exp_id in ALL_EXPERIMENT_IDS:
        if exp_id not in results:
            continue
        r = results[exp_id]
        ema_str = f"{r['final_ema']:.1f}%" if r['final_ema'] is not None else "N/A"
        loss_str = f"{r['final_loss']:.3f}" if r['final_loss'] is not None else "N/A"

        # Baseline 대비 차이 표시
        delta = ""
        if baseline_ema is not None and r['final_ema'] is not None and exp_id != "BASELINE":
            diff = r['final_ema'] - baseline_ema
            delta = f" ({'+' if diff >= 0 else ''}{diff:.1f}%)"

        print(f"{exp_id:>10} | {r['category']:>12} | {ema_str:>10}{delta:>8} | {loss_str:>10} | {r['description'][:40]}")


def save_comparison_csv(results: dict) -> str:
    """비교표를 CSV로 저장한다."""
    csv_path = os.path.join(RESULTS_DIR, "ablation_summary.csv")
    baseline_ema = results.get("BASELINE", {}).get("final_ema")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Category", "Description", "Final_EMA(%)", "Delta_vs_Baseline(%)", "Final_Loss"])

        for exp_id in ALL_EXPERIMENT_IDS:
            if exp_id not in results:
                continue
            r = results[exp_id]
            delta = ""
            if baseline_ema and r['final_ema'] is not None and exp_id != "BASELINE":
                delta = f"{r['final_ema'] - baseline_ema:+.1f}"

            writer.writerow([
                exp_id, r['category'], r['description'],
                f"{r['final_ema']:.1f}" if r['final_ema'] else "N/A",
                delta,
                f"{r['final_loss']:.3f}" if r['final_loss'] else "N/A",
            ])

    print(f"\n📄 비교표 저장: {csv_path}")
    return csv_path


def plot_ema_bar_chart(results: dict) -> str:
    """최종 EMA 바 차트를 생성한다."""
    chart_path = os.path.join(RESULTS_DIR, "ablation_ema_chart.png")

    ids = [eid for eid in ALL_EXPERIMENT_IDS if eid in results and results[eid]['final_ema'] is not None]
    emas = [results[eid]['final_ema'] for eid in ids]
    categories = [results[eid]['category'] for eid in ids]

    # 카테고리별 색상
    color_map = {
        "Baseline": "#4CAF50",
        "Architecture": "#2196F3",
        "State": "#FF9800",
        "Reward": "#E91E63",
    }
    colors = [color_map.get(c, "#9E9E9E") for c in categories]

    fig, ax = plt.subplots(figsize=(16, 6))
    bars = ax.bar(range(len(ids)), emas, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)

    # Baseline 기준선
    baseline_ema = results.get("BASELINE", {}).get("final_ema")
    if baseline_ema:
        ax.axhline(y=baseline_ema, color='red', linestyle='--', alpha=0.7, label=f'Baseline ({baseline_ema:.1f}%)')

    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(ids, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel("Success Rate EMA (%)", fontsize=12)
    ax.set_title("Worker Ablation Study — Final EMA Comparison", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 105)

    # 값 표시
    for bar, val in zip(bars, emas):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"📈 바 차트 저장: {chart_path}")
    return chart_path


def plot_learning_curves_overlay(results: dict) -> str:
    """모든 실험의 EMA 학습 곡선을 오버레이한다."""
    chart_path = os.path.join(RESULTS_DIR, "ablation_curves.png")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = ["Architecture (A1~A7)", "State (S1~S5)", "Reward (R1~R5)"]
    prefixes = ["A", "S", "R"]

    for ax, title, prefix in zip(axes, titles, prefixes):
        # Baseline 항상 표시
        if "BASELINE" in results and results["BASELINE"]["ema_history"]:
            ax.plot(results["BASELINE"]["ema_history"], 'k-', linewidth=2,
                    alpha=0.8, label="BASELINE")

        for exp_id in ALL_EXPERIMENT_IDS:
            if not exp_id.startswith(prefix):
                continue
            if exp_id not in results or not results[exp_id]["ema_history"]:
                continue
            ax.plot(results[exp_id]["ema_history"], linewidth=1.5, alpha=0.7, label=exp_id)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("Checkpoint (×200ep)")
        ax.set_ylabel("EMA (%)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 105)

    plt.suptitle("Worker Ablation Study — Learning Curves", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"📈 학습 곡선 오버레이 저장: {chart_path}")
    return chart_path


def generate_judgment(results: dict) -> None:
    """Baseline 대비 판정을 출력한다."""
    baseline_ema = results.get("BASELINE", {}).get("final_ema")
    if baseline_ema is None:
        print("\n⚠️ Baseline 결과가 없어 판정을 생략합니다.")
        return

    print(f"\n{'='*80}")
    print(f"📋 Ablation 판정 (Baseline EMA: {baseline_ema:.1f}%)")
    print(f"{'='*80}")

    for exp_id in ALL_EXPERIMENT_IDS:
        if exp_id == "BASELINE" or exp_id not in results:
            continue
        r = results[exp_id]
        if r['final_ema'] is None:
            continue

        diff = r['final_ema'] - baseline_ema
        if diff > 3.0:
            verdict = "⬆️ 기존 설정이 오히려 방해 → 제거 권장"
        elif diff >= -3.0:
            verdict = "➡️ 해당 요소는 불필요 → 경량화 가능"
        elif diff >= -10.0:
            verdict = "⬇️ 해당 요소가 일부 기여 → 유지 검토"
        else:
            verdict = "🔻 해당 요소는 필수 → 유지 확정"

        print(f"  {exp_id:>8} | Δ{diff:+6.1f}% | {verdict}")


def main():
    if not os.path.isdir(RESULTS_DIR):
        print(f"❌ 결과 디렉토리가 없습니다: {RESULTS_DIR}")
        print(f"   먼저 run_ablation.py를 실행하세요.")
        return

    results = collect_all_results()

    if not results:
        print("❌ 수집된 결과가 없습니다.")
        return

    print(f"✅ {len(results)}개 실험 결과 수집 완료")

    # 1. 비교표 출력/저장
    print_comparison_table(results)
    save_comparison_csv(results)

    # 2. 시각화
    plot_ema_bar_chart(results)
    plot_learning_curves_overlay(results)

    # 3. 판정
    generate_judgment(results)

    print(f"\n✅ 분석 완료! 결과: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
