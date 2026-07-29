"""
Batch launcher for fixed_hybrid PPO training across 12 months.

Runs train() with mode="fixed_hybrid" for each month,
with per-month long_term_robots and long_term_pickers_area
read from a JSON config file.

Usage:
  # Use default config file (configs/fixed_hybrid_batch.json)
  python -m agent.ppo.train_fixed_hybrid

  # Custom config file
  python -m agent.ppo.train_fixed_hybrid --config path/to/my_config.json

  # 4 parallel workers
  python -m agent.ppo.train_fixed_hybrid --workers 4 --cpu-threads 1

  # Only specific months
  python -m agent.ppo.train_fixed_hybrid --months 1,6,11,12

  # Dry-run: preview parameters for each month
  python -m agent.ppo.train_fixed_hybrid --dry-run
"""

from __future__ import annotations

import argparse
import json
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

DEFAULT_BATCH_CONFIG = REPO_ROOT / "configs" / "fixed_hybrid_batch.json"
ALL_MONTHS = list(range(1, 13))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-launch fixed_hybrid PPO training across months."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_BATCH_CONFIG),
        help=f"Path to per-month fixed_hybrid JSON config (default: {DEFAULT_BATCH_CONFIG}).",
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
        help="Force CPU mode even if CUDA is available.",
    )
    parser.add_argument(
        "--months",
        type=str,
        default="1,2,3,4,5,6,7,8,9,10,11,12",
        help="Comma-separated month numbers (default: 1..12).",
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
# Per-month config loading
# ---------------------------------------------------------------------------

def load_fixed_hybrid_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load the per-month fixed_hybrid config file.

    Expected format:
    {
        "default": {"long_term_robots": 3, "long_term_pickers_area": [1,1,1]},
        "months": {
            "1":  {"long_term_robots": 5, "long_term_pickers_area": [3,2,1]},
            "6":  {"long_term_robots": 4, "long_term_pickers_area": [2,2,2]},
            ...
        }
    }

    Returns the parsed dict.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Fixed-hybrid batch config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    if "default" not in cfg:
        raise KeyError("Config must contain a 'default' key")
    if "long_term_robots" not in cfg["default"]:
        raise KeyError("default.long_term_robots is required")
    if "long_term_pickers_area" not in cfg["default"]:
        raise KeyError("default.long_term_pickers_area is required")

    # Validate every month entry if present
    months_cfg = cfg.get("months", {})
    for month_key, month_val in months_cfg.items():
        if "long_term_robots" not in month_val:
            raise KeyError(f"months.{month_key}.long_term_robots is required")
        if "long_term_pickers_area" not in month_val:
            raise KeyError(f"months.{month_key}.long_term_pickers_area is required")

    return cfg


def get_fixed_hybrid_params(
    batch_config: dict[str, Any],
    month: int,
) -> dict[str, int | list[int]]:
    """
    Resolve per-month fixed_hybrid parameters.
    Falls back to 'default' if the month is not explicitly listed.
    """
    months_cfg = batch_config.get("months", {})
    month_str = str(month)

    if month_str in months_cfg:
        entry = months_cfg[month_str]
    else:
        entry = batch_config["default"]

    return {
        "long_term_robots": int(entry["long_term_robots"]),
        "long_term_pickers_area": [int(v) for v in entry["long_term_pickers_area"]],
    }


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker_init(cpu_threads: int, force_cpu: bool) -> None:
    # Ensure the worker can import from the project root (spawn context)
    sys.path.insert(0, str(REPO_ROOT))

    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_threads)

    import torch

    torch.set_num_threads(cpu_threads)
    if force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


def run_one(month: int, fixed_hybrid_params: dict) -> dict:
    """
    Run a single fixed_hybrid training job for a given month.
    Each worker process loads config & model independently.
    """
    robots = fixed_hybrid_params["long_term_robots"]
    pickers = fixed_hybrid_params["long_term_pickers_area"]
    pickers_str = ",".join(str(p) for p in pickers)
    label = f"m{month:02d}_fixed_hybrid  (robots={robots}, pickers=[{pickers_str}])"

    start = time.time()
    t0_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{t0_str}] START  {label}", flush=True)

    try:
        from agent.ppo.train import train
        from environment.class_public import load_config

        parameters = load_config()
        parameters["experiment"]["month"] = month
        parameters["experiment"]["mode"] = "fixed_hybrid"
        parameters["experiment"]["fixed_hybrid"] = {
            "long_term_robots": robots,
            "long_term_pickers_area": pickers,
        }
        csv_path = train(parameters)

        elapsed = time.time() - start
        t1_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{t1_str}] OK     {label}  → {csv_path}  ({elapsed:.0f}s)", flush=True)
        return {"month": month, "success": True, "elapsed": elapsed}
    except Exception as exc:
        elapsed = time.time() - start
        t1_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{t1_str}] FAIL   {label}  ({elapsed:.0f}s)", flush=True)
        print(f"  {type(exc).__name__}: {exc}", flush=True)
        return {"month": month, "success": False, "elapsed": elapsed}


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

    # Load per-month fixed_hybrid config
    batch_config = load_fixed_hybrid_config(args.config)

    # Build task list: one task per month, each with its own fixed_hybrid params
    tasks = []
    for m in months:
        params = get_fixed_hybrid_params(batch_config, m)
        tasks.append({"month": m, "fixed_hybrid": params})

    cpu_count = os.cpu_count() or 1
    print(f"Config:     {args.config}")
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

    # --- Dry-run: print each month's parameters ---
    if args.dry_run:
        print(f"{'#':>3}  {'month':>5}  {'robots':>7}  pickers")
        print("-" * 40)
        for i, t in enumerate(tasks, 1):
            p = t["fixed_hybrid"]
            pickers_str = str(p["long_term_pickers_area"])
            print(f"{i:>3}  {t['month']:>5}  {p['long_term_robots']:>7}  {pickers_str}")
        return

    # --- Sequential ---
    if args.workers <= 1:
        _worker_init(args.cpu_threads, args.force_cpu)
        results = [run_one(t["month"], t["fixed_hybrid"]) for t in tasks]
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
                pool.submit(run_one, t["month"], t["fixed_hybrid"]): t
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
                print(f"  FAILED: month={r['month']}")


if __name__ == "__main__":
    main()
