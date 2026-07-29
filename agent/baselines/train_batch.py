"""
Batch launcher for baseline algorithm training across multiple months.

Runs any baseline algorithm (ddpg/td3/sac/a2c) in any mode
(short/long/hybrid/fixed_hybrid) for each specified month.

Usage:
  # DDPG long mode, all 12 months, 2 parallel workers
  python -m agent.baselines.train_batch --algorithm ddpg --mode long

  # A2C short mode, only months 6 and 9, 4 workers
  python -m agent.baselines.train_batch --algorithm a2c --mode short --months 6,9 --workers 4

  # TD3 hybrid mode, month 9, custom episodes
  python -m agent.baselines.train_batch --algorithm td3 --mode hybrid --months 9 --episodes 1000

  # Dry-run: preview what would be executed
  python -m agent.baselines.train_batch --algorithm sac --mode fixed_hybrid --dry-run
"""

from __future__ import annotations

import argparse
import importlib
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

VALID_ALGORITHMS = ("ddpg", "td3", "sac", "a2c")
VALID_MODES = ("short", "long", "hybrid", "fixed_hybrid")
ALL_MONTHS = list(range(1, 13))


def month_seed(base_seed: int, month: int) -> int:
    return int(base_seed) + int(month) - 1


def build_training_args(module, mode: str, month: int, episodes: int | None = None):
    parser = module.build_parser()
    args, _ = parser.parse_known_args([])
    args.mode = mode
    args.month = month
    args.seed = month_seed(args.seed, month)
    if episodes is not None:
        args.episodes = int(episodes)
    return args


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-launch baseline algorithm training across months."
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        choices=VALID_ALGORITHMS,
        help=f"Baseline algorithm: {', '.join(VALID_ALGORITHMS)}.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=VALID_MODES,
        help=f"Training mode: {', '.join(VALID_MODES)}.",
    )
    parser.add_argument(
        "--months",
        type=str,
        default="1,2,3,4,5,6,7,8,9,10,11,12",
        help="Comma-separated month numbers (default: 1..12).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override the number of training episodes (default: use config value).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of parallel training processes (default: 2).",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=2,
        help="PyTorch threads per worker (default: 2).",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU mode even if CUDA is available.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print per-month parameters without running training.",
    )
    return parser


def parse_list_arg(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker_init(cpu_threads: int, force_cpu: bool) -> None:
    sys.path.insert(0, str(REPO_ROOT))

    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_threads)

    import torch

    torch.set_num_threads(cpu_threads)
    if force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


def run_one(algorithm: str, mode: str, month: int, episodes: int | None = None) -> dict:
    """
    Run a single baseline training job for a given algorithm / mode / month.
    Each worker process loads its own module independently.
    """
    label = f"{algorithm.upper()} {mode} m{month:02d}"

    start = time.time()
    t0_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{t0_str}] START  {label}", flush=True)

    try:
        # Dynamically import the algorithm module
        module = importlib.import_module(f"agent.baselines.{algorithm}")
        args = build_training_args(module, mode, month, episodes)

        module.train(args)

        elapsed = time.time() - start
        t1_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{t1_str}] OK     {label}  ({elapsed:.0f}s)", flush=True)
        return {"algorithm": algorithm, "mode": mode, "month": month, "success": True, "elapsed": elapsed}
    except Exception as exc:
        elapsed = time.time() - start
        t1_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{t1_str}] FAIL   {label}  ({elapsed:.0f}s)", flush=True)
        print(f"  {type(exc).__name__}: {exc}", flush=True)
        return {"algorithm": algorithm, "mode": mode, "month": month, "success": False, "elapsed": elapsed}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()
    months_raw = parse_list_arg(args.months)
    months = [int(m) for m in months_raw]
    for m in months:
        if m < 1 or m > 12:
            raise ValueError(f"Invalid month: {m}")

    tasks = [
        {"algorithm": args.algorithm, "mode": args.mode, "month": m, "episodes": args.episodes}
        for m in months
    ]

    cpu_count = os.cpu_count() or 1
    print(f"Algorithm:  {args.algorithm.upper()}")
    print(f"Mode:       {args.mode}")
    print(f"Months:     {months}")
    print(f"Episodes:   {args.episodes or 'config default'}")
    print(f"CPU cores:  {cpu_count}")
    print(f"Workers:    {args.workers}")
    print(f"CPU threads/worker: {args.cpu_threads}")
    print(f"Force CPU:  {args.force_cpu}")
    print(f"Dry run:    {args.dry_run}")

    total_threads = args.workers * args.cpu_threads
    if total_threads > cpu_count:
        print(f"⚠  WARNING: {args.workers} workers × {args.cpu_threads} threads "
              f"= {total_threads} logical threads, but only {cpu_count} CPU cores.")
        print(f"   Consider --cpu-threads 1 or fewer --workers.")
    print("-" * 60)

    # --- Dry-run ---
    if args.dry_run:
        from agent.baselines.common import DEFAULT_PARAMETERS

        base_seed = int(DEFAULT_PARAMETERS["experiment"]["seed"])
        print(
            f"{'#':>3}  {'algorithm':>10}  {'mode':>14}  "
            f"{'month':>5}  {'seed':>6}  {'episodes':>9}"
        )
        print("-" * 65)
        for i, t in enumerate(tasks, 1):
            ep_str = str(t["episodes"]) if t["episodes"] else "default"
            task_seed = month_seed(base_seed, t["month"])
            print(
                f"{i:>3}  {t['algorithm']:>10}  {t['mode']:>14}  "
                f"{t['month']:>5}  {task_seed:>6}  {ep_str:>9}"
            )
        return

    # --- Sequential ---
    if args.workers <= 1:
        _worker_init(args.cpu_threads, args.force_cpu)
        results = [run_one(t["algorithm"], t["mode"], t["month"], t["episodes"]) for t in tasks]
    else:
        # --- Parallel ---
        ctx = multiprocessing.get_context("spawn")
        results = []
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(args.cpu_threads, args.force_cpu),
        ) as pool:
            futures = {
                pool.submit(run_one, t["algorithm"], t["mode"], t["month"], t["episodes"]): t
                for t in tasks
            }
            for future in as_completed(futures):
                results.append(future.result())

    # --- Summary ---
    ok = sum(1 for r in results if r["success"])
    fail = len(results) - ok
    total_elapsed = sum(r["elapsed"] for r in results)
    print("\n" + "=" * 60)
    print(f"SUMMARY: {ok} succeeded, {fail} failed, {len(results)} total")
    print(f"Total wall-clock time: {total_elapsed/60:.1f} min")
    if fail:
        for r in results:
            if not r["success"]:
                print(f"  FAILED: {r['algorithm']} {r['mode']} month={r['month']}")


if __name__ == "__main__":
    main()
