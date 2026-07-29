"""
Batch launcher for PPO training across all months and modes.

Runs train() for every combination of:
  - month ∈ {1..12}
  - mode ∈ {short, long, hybrid}

Usage:
  # sequential (one at a time)
  python -m agent.ppo.train_batch

  # parallel: 4 workers, each limited to 2 CPU threads
  python -m agent.ppo.train_batch --workers 4 --cpu-threads 2

  # 12 workers, single-threaded PyTorch (safe for CPU-heavy workloads)
  python -m agent.ppo.train_batch --workers 12 --cpu-threads 1

  # filter specific months / modes
  python -m agent.ppo.train_batch --months 1,3,5 --modes short,long

  # dry-run (print what would run, don't run)
  python -m agent.ppo.train_batch --dry-run
"""

from __future__ import annotations

import argparse
import itertools
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from environment.class_public import load_config

ALL_MODES = ("short", "long", "hybrid")
ALL_MONTHS = list(range(1, 13))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-launch PPO training across months × modes."
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
        help="PyTorch threads per worker. 1 = safest for high parallelism (default: 2).",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU mode even if CUDA is available (avoids GPU contention).",
    )
    parser.add_argument(
        "--months",
        type=str,
        default="1,2,3,4,5,6,7,8,9,10,11,12",
        help="Comma-separated month numbers (default: 1..12).",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="short,long,hybrid",
        help="Comma-separated modes (default: short,long,hybrid).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the task list without executing.",
    )
    return parser


def parse_list_arg(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def build_task_list(months: list[int], modes: list[str]) -> list[dict]:
    return [{"month": m, "mode": mode} for m, mode in itertools.product(months, modes)]


def _worker_init(cpu_threads: int, force_cpu: bool) -> None:
    """Called once per worker process to configure PyTorch threading."""
    # Ensure the worker can import from the project root (spawn context)
    sys.path.insert(0, str(REPO_ROOT))

    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_threads)

    import torch

    torch.set_num_threads(cpu_threads)
    if force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


def run_one(month: int, mode: str) -> dict:
    """
    Run a single training job.  Executed inside a worker process —
    each process has its own Python interpreter, memory space, and
    PyTorch context, so there is no shared-state interference.
    """
    label = f"m{month:02d}_{mode}"
    start = time.time()
    t0_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{t0_str}] START  {label}", flush=True)

    try:
        from agent.ppo.train import train

        parameters = load_config()
        parameters["experiment"]["month"] = month
        parameters["experiment"]["mode"] = mode
        csv_path = train(parameters)

        elapsed = time.time() - start
        t1_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{t1_str}] OK     {label}  → {csv_path}  ({elapsed:.0f}s)", flush=True)
        return {"month": month, "mode": mode, "success": True, "elapsed": elapsed}
    except Exception as exc:
        elapsed = time.time() - start
        t1_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{t1_str}] FAIL   {label}  ({elapsed:.0f}s)", flush=True)
        print(f"  {type(exc).__name__}: {exc}", flush=True)
        return {"month": month, "mode": mode, "success": False, "elapsed": elapsed}


def main() -> None:
    args = build_parser().parse_args()
    months_raw = parse_list_arg(args.months)
    modes_raw = parse_list_arg(args.modes)

    months = [int(m) for m in months_raw]
    for m in months:
        if m < 1 or m > 12:
            raise ValueError(f"Invalid month: {m}")
    for mode in modes_raw:
        if mode not in ALL_MODES:
            raise ValueError(f"Invalid mode: {mode} (choose from {ALL_MODES})")

    tasks = build_task_list(months, modes_raw)

    cpu_count = os.cpu_count() or 1
    print(f"CPU cores:  {cpu_count}")
    print(f"Total tasks: {len(tasks)}  (months={months}, modes={modes_raw})")
    print(f"Workers:    {args.workers}")
    print(f"CPU threads/worker: {args.cpu_threads}")
    print(f"Force CPU:  {args.force_cpu}")
    print(f"Dry run:    {args.dry_run}")

    # Warn if oversubscription is likely
    total_threads = args.workers * args.cpu_threads
    if total_threads > cpu_count:
        print(f"⚠  WARNING: {args.workers} workers × {args.cpu_threads} threads = {total_threads} "
              f"logical threads, but only {cpu_count} CPU cores available.")
        print(f"   Consider --cpu-threads 1 or fewer --workers to avoid slowdown.")
    print("-" * 60)

    if args.dry_run:
        for i, t in enumerate(tasks, 1):
            print(f"  {i:>3}. month={t['month']:>2}  mode={t['mode']}")
        return

    # --- Sequential ---
    if args.workers <= 1:
        _worker_init(args.cpu_threads, args.force_cpu)
        results = [run_one(t["month"], t["mode"]) for t in tasks]
    else:
        # --- Parallel ---
        # Use "spawn" context — required on Windows, safe everywhere.
        # Each worker gets a fresh interpreter → no shared PyTorch/CUDA state.
        ctx = multiprocessing.get_context("spawn")
        results = []
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(args.cpu_threads, args.force_cpu),
        ) as pool:
            futures = {
                pool.submit(run_one, t["month"], t["mode"]): t for t in tasks
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
                print(f"  FAILED: month={r['month']}, mode={r['mode']}")


if __name__ == "__main__":
    main()
