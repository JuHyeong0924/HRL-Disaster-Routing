"""
run_ablation.py — Worker Ablation Study 자동 실행 러너
GPU 2개를 활용하여 병렬 실행하고, 결과를 tests/ablation_results/에 저장한다.

사용법:
  # 전체 실험 실행 (GPU 2개 병렬, 각 3000ep)
  python tests/run_ablation.py --episodes 3000 --pomo 48

  # 단일 실험 실행
  python tests/run_ablation.py --only A1 --episodes 3000

  # 빠른 smoke test (5ep)
  python tests/run_ablation.py --smoke
"""
import os
import sys
import argparse
import subprocess
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ablation_configs import ABLATION_REGISTRY, ABLATION_ROUNDS, ALL_EXPERIMENT_IDS


def get_results_dir() -> str:
    """결과 저장 디렉토리 반환."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "ablation_results")


def run_single_experiment(
    ablation_id: str,
    episodes: int,
    pomo: int,
    gpu_id: int,
    results_dir: str,
    debug: bool = True,
    map_name: str = "Anaheim",
) -> subprocess.Popen:
    """단일 ablation 실험을 서브프로세스로 실행한다."""
    exp_dir = os.path.join(results_dir, ablation_id)
    os.makedirs(exp_dir, exist_ok=True)

    # train_rl.py 기반 실행 (결과를 exp_dir에 저장)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [
        sys.executable, os.path.join(project_root, "train_rl.py"),
        "--stage", "worker",
        "--episodes", str(episodes),
        "--num_pomo", str(pomo),
        "--map", map_name,
        "--ablation", ablation_id,
        "--disable_tqdm",
    ]
    if debug:
        cmd.append("--debug")

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}

    # stdout을 로그 파일로 저장
    log_path = os.path.join(exp_dir, "train_log.txt")
    log_file = open(log_path, "w", encoding="utf-8")

    print(f"  🚀 [{ablation_id}] GPU {gpu_id} | {episodes}ep | POMO {pomo}")

    proc = subprocess.Popen(
        cmd, env=env, cwd=project_root,
        stdout=log_file, stderr=subprocess.STDOUT,
        text=True,
    )

    # 메타데이터 저장
    meta = {
        "ablation_id": ablation_id,
        "config": ABLATION_REGISTRY.get(ablation_id.upper(), {}),
        "episodes": episodes,
        "pomo": pomo,
        "gpu_id": gpu_id,
        "map": map_name,
        "started_at": datetime.now().isoformat(),
        "pid": proc.pid,
    }
    with open(os.path.join(exp_dir, "experiment_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return proc, log_file


def run_round(
    experiments: list,
    episodes: int,
    pomo: int,
    results_dir: str,
    debug: bool,
    map_name: str,
) -> list:
    """GPU 2개에 실험 2개를 병렬로 배치한다."""
    procs = []
    log_files = []

    for gpu_id, exp_id in enumerate(experiments):
        if gpu_id >= 2:
            break  # GPU 2개만 사용
        proc, log_file = run_single_experiment(
            exp_id, episodes, pomo, gpu_id, results_dir, debug, map_name
        )
        procs.append((exp_id, proc))
        log_files.append(log_file)

    # 모든 프로세스 완료 대기
    results = []
    for exp_id, proc in procs:
        rc = proc.wait()
        results.append({"id": exp_id, "exit_code": rc})
        status = "✅" if rc == 0 else "❌"
        print(f"  {status} [{exp_id}] 완료 (exit code: {rc})")

    # 로그 파일 닫기
    for lf in log_files:
        lf.close()

    return results


def copy_rl_results(results_dir: str, exp_id: str) -> None:
    """train_rl.py가 저장한 결과를 ablation_results/로 복사한다."""
    import shutil
    # train_rl.py는 logs/rl_worker_stage/최신폴더/에 결과를 저장
    rl_base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "logs", "rl_worker_stage"
    )
    if not os.path.exists(rl_base):
        return

    # 가장 최근 디렉토리 찾기
    subdirs = [
        os.path.join(rl_base, d) for d in os.listdir(rl_base)
        if os.path.isdir(os.path.join(rl_base, d))
    ]
    if not subdirs:
        return

    latest = max(subdirs, key=os.path.getmtime)
    exp_dir = os.path.join(results_dir, exp_id)

    # 핵심 파일만 복사
    for fname in ["final.pt", "debug_metrics.csv", "rl_learning_curve.png",
                   "rl_debug_log.txt", "runtime_config.json"]:
        src = os.path.join(latest, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(exp_dir, fname))


def main():
    parser = argparse.ArgumentParser(description="Worker Ablation Study 자동 실행")
    parser.add_argument("--episodes", type=int, default=3000,
                        help="실험당 에피소드 수 (기본: 3000)")
    parser.add_argument("--pomo", type=int, default=48,
                        help="POMO 배치 크기 (기본: 48)")
    parser.add_argument("--map", type=str, default="Anaheim")
    parser.add_argument("--only", type=str, default=None,
                        help="특정 실험만 실행 (예: --only A1)")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: 5ep으로 전체 실험 검증")
    parser.add_argument("--no-debug", action="store_true",
                        help="디버그 모드 비활성화")
    parser.add_argument("--sequential", action="store_true",
                        help="순차 실행 (병렬 대신)")
    args = parser.parse_args()

    if args.smoke:
        args.episodes = 5

    results_dir = get_results_dir()
    os.makedirs(results_dir, exist_ok=True)
    debug = not args.no_debug

    print(f"{'='*60}")
    print(f"🧪 Worker Ablation Study")
    print(f"{'='*60}")
    print(f"  에피소드: {args.episodes}")
    print(f"  POMO: {args.pomo}")
    print(f"  결과 저장: {results_dir}")
    print(f"  모드: {'Smoke Test' if args.smoke else '병렬' if not args.sequential else '순차'}")
    print()

    start_time = time.time()
    all_results = []

    if args.only:
        # 단일 실험 실행
        exp_id = args.only.upper()
        if exp_id not in ABLATION_REGISTRY:
            print(f"❌ Unknown experiment: {exp_id}")
            print(f"   Available: {list(ABLATION_REGISTRY.keys())}")
            return
        results = run_round([exp_id], args.episodes, args.pomo, results_dir, debug, args.map)
        all_results.extend(results)
        copy_rl_results(results_dir, exp_id)
    elif args.sequential:
        # 순차 실행
        for exp_id in ALL_EXPERIMENT_IDS:
            print(f"\n--- Round: {exp_id} ---")
            results = run_round([exp_id], args.episodes, args.pomo, results_dir, debug, args.map)
            all_results.extend(results)
            copy_rl_results(results_dir, exp_id)
    else:
        # GPU 2개 병렬 실행 (라운드별)
        for round_idx, round_exps in enumerate(ABLATION_ROUNDS):
            print(f"\n--- Round {round_idx+1}/{len(ABLATION_ROUNDS)}: {round_exps} ---")
            results = run_round(round_exps, args.episodes, args.pomo, results_dir, debug, args.map)
            all_results.extend(results)
            for exp_id in round_exps:
                copy_rl_results(results_dir, exp_id)

    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    mins = int((elapsed % 3600) // 60)

    # 최종 결과 저장
    summary = {
        "total_experiments": len(all_results),
        "elapsed_time": f"{hours}h {mins}m",
        "episodes_per_experiment": args.episodes,
        "results": all_results,
        "completed_at": datetime.now().isoformat(),
    }
    with open(os.path.join(results_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"🏁 전체 Ablation Study 완료!")
    print(f"   총 소요 시간: {hours}h {mins}m")
    print(f"   성공: {sum(1 for r in all_results if r['exit_code'] == 0)}/{len(all_results)}")
    print(f"   결과: {results_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
