"""GCP Monte Carlo driver with checkpoint/resume for Spot VMs.

Wraps the same simulation path as ``run_monte_carlo.py`` but:
- uses ``multiprocessing.Pool`` with ``maxtasksperchild=1`` so worker memory
  is released between draws;
- writes each draw to Parquet immediately (parent never holds zone frames);
- saves ``checkpoint.json`` for resume after preemption.

For a 30 arc-second production run:

    python scripts/run_monte_carlo_gcp.py \\
        --input data/GIS/extracted/master_30as.parquet \\
        --workers 1 \\
        --seed 20260808
"""

import argparse
import json
import multiprocessing
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import src.gis_global_module as gis  # noqa: E402
import src.gis_monte_carlo as mc  # noqa: E402
from src.monte_carlo_parameters import sample_parameters  # noqa: E402
from scripts.run_monte_carlo import (  # noqa: E402
    CHUNK_ROWS_30AS,
    MAX_WORKERS_30AS,
    ZONE_RESULT_COLUMNS,
    _init_worker,
    format_parameter_draws,
)

RESULTS_PATH = project_root / "results"
PARQUET_PATH = RESULTS_PATH / "parquet_files"
CHECKPOINT_FILE = PARQUET_PATH / "checkpoint.json"

# Give up on a single iteration if no result arrives within this many seconds.
ITERATION_TIMEOUT = 2 * 60 * 60

shutdown_requested = False


def handle_preemption(signum, frame):
    """Handle Spot VM preemption signal (30 second warning)."""
    global shutdown_requested
    print("\n⚠️  Preemption signal received! Saving checkpoint...")
    shutdown_requested = True


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="number of Monte Carlo draws (default 1000)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="zone frame (CSV or Parquet; default the 30 arc-second master grid)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"simultaneous worker processes (default {MAX_WORKERS_30AS})",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=None,
        help=f"chunk budget (default {CHUNK_ROWS_30AS:,}; 0 disables)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="fix the parameter draws (default: unseeded)",
    )
    parser.add_argument(
        "--save-zone-parquet",
        dest="save_zone_parquet",
        action="store_true",
        default=None,
        help="force per-draw zone Parquet on",
    )
    parser.add_argument(
        "--no-save-zone-parquet",
        dest="save_zone_parquet",
        action="store_false",
        help="force zone Parquet off (default)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_PATH,
        help="directory for results/ and parquet_files/",
    )
    return parser.parse_args(argv)


def _find_gsutil():
    found = shutil.which("gsutil")
    if found:
        return found
    for candidate in (
        "/snap/bin/gsutil",
        "/usr/lib/google-cloud-sdk/bin/gsutil",
        "/root/google-cloud-sdk/bin/gsutil",
        "/usr/bin/gsutil",
    ):
        if Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def sync_results_to_gcs():
    """Push ``results/`` to GCS when ``GCS_RESULTS_URI`` is set (Spot VMs).

    Cron alone is unreliable: default cron PATH often omits ``gsutil``
    (commonly under ``/snap/bin``), so syncs failed silently with ``-q``.
    Calling this after each completed draw is the durable path; cron remains
    a backup.
    """
    uri = os.environ.get("GCS_RESULTS_URI", "").strip().rstrip("/")
    if not uri:
        return
    gsutil = _find_gsutil()
    if gsutil is None:
        print("⚠️  gsutil not found; skipping GCS sync")
        return
    result = subprocess.run(
        [gsutil, "-m", "rsync", "-r", f"{RESULTS_PATH}/", f"{uri}/"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()[-500:]
        print(f"⚠️  GCS sync failed: {err}")
    else:
        print(f"☁️  Synced results/ → {uri}/")


def save_checkpoint(completed_iterations, parameters, run_config):
    checkpoint = {
        "completed_iterations": completed_iterations,
        "total_iterations": run_config["iterations"],
        "parameters": parameters,
        "run_config": run_config,
        "timestamp": time.time(),
    }
    PARQUET_PATH.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)
    print(f"✅ Checkpoint saved: {len(completed_iterations)} iterations completed")
    sync_results_to_gcs()


def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return None


def get_completed_iterations(num_iterations, require_zone):
    completed = []
    for i in range(num_iterations):
        district_file = PARQUET_PATH / f"district_simulation_result_{i}.parquet"
        countries_file = PARQUET_PATH / f"countries_simulation_result_{i}.parquet"
        zone_file = PARQUET_PATH / f"zone_simulation_result_{i}.parquet"
        if not district_file.exists() or not countries_file.exists():
            continue
        if require_zone and not zone_file.exists():
            continue
        completed.append(i)
    return completed


def process_saved_results(num_iterations, require_zone):
    print("\n📊 Processing saved results...")
    import pandas as pd

    districts_simulation_results = []
    countries_simulation_results = []
    completed = get_completed_iterations(
        num_iterations=num_iterations, require_zone=require_zone
    )

    for i in completed:
        district_file = PARQUET_PATH / f"district_simulation_result_{i}.parquet"
        countries_file = PARQUET_PATH / f"countries_simulation_result_{i}.parquet"
        if district_file.exists():
            districts_simulation_results.append(pd.read_parquet(district_file))
        if countries_file.exists():
            countries_simulation_results.append(pd.read_parquet(countries_file))

    if districts_simulation_results and countries_simulation_results:
        mc.process_mc_results(
            countries_simulation_results,
            plot=False,
            output_dir=str(RESULTS_PATH),
        )
        mc.process_districts_results(
            districts_simulation_results,
            output_dir=str(RESULTS_PATH),
        )
        print(f"✅ Processed {len(completed)} simulation results")
    else:
        print("⚠️  No results to process yet")


def run_and_save_simulation(job):
    """Worker entry point: one draw, write Parquet, return draw index."""
    i = job["index"]
    _init_worker(job["zone_file"])

    zone_output_path = None
    if job["save_zone_parquet"] and (job["chunk_rows"] or job["chunk_plan"] is not None):
        zone_output_path = PARQUET_PATH / f"zone_simulation_result_{i}.parquet"

    countries_result, district_result, zone_result = mc.run_simulation(
        crr_adjustment=job["crr_adjustment"],
        time_gathering_water=job["time_gathering_water"],
        practical_limit_bicycle=job["practical_limit_bicycle"],
        practical_limit_buckets=job["practical_limit_buckets"],
        met=job["met"],
        watts=job["watts"],
        hill_polarity=job["hill_polarity"],
        urban_adjustment=job["urban_adjustment"],
        rural_adjustment=job["rural_adjustment"],
        urban_borehole_lose_fraction=job["urban_borehole_lose_fraction"],
        rural_borehole_lose_fraction=job["rural_borehole_lose_fraction"],
        unallocated_vulnerable_fraction=job["unallocated_vulnerable_fraction"],
        use_sample_data=False,
        zone_columns=ZONE_RESULT_COLUMNS if job["save_zone_parquet"] else [],
        chunk_rows=job["chunk_rows"] or None,
        chunk_plan=job["chunk_plan"],
        zone_output_path=zone_output_path,
    )

    district_result.to_parquet(
        PARQUET_PATH / f"district_simulation_result_{i}.parquet", index=False
    )
    countries_result.to_parquet(
        PARQUET_PATH / f"countries_simulation_result_{i}.parquet", index=False
    )
    if job["save_zone_parquet"]:
        if zone_output_path is not None:
            if not zone_output_path.exists():
                raise RuntimeError(
                    f"chunked zone run finished but {zone_output_path} was not written"
                )
        elif not zone_result.empty:
            zone_result[ZONE_RESULT_COLUMNS].to_parquet(
                PARQUET_PATH / f"zone_simulation_result_{i}.parquet", index=False
            )

    return i


def _require_borehole_params(params):
    for key in (
        "urban_borehole_lose_fractions",
        "rural_borehole_lose_fractions",
        "unallocated_vulnerable_fractions",
    ):
        if key not in params:
            raise RuntimeError(
                f"checkpoint.json predates {key}; delete the checkpoint and restart"
            )


def main(argv=None):
    global RESULTS_PATH, PARQUET_PATH, CHECKPOINT_FILE

    args = parse_args(argv)
    workers = args.workers if args.workers is not None else MAX_WORKERS_30AS
    save_zone_parquet = (
        args.save_zone_parquet if args.save_zone_parquet is not None else False
    )

    if args.input is not None:
        gis.URB_DATA_FILE = args.input

    RESULTS_PATH = args.output_dir
    PARQUET_PATH = RESULTS_PATH / "parquet_files"
    CHECKPOINT_FILE = PARQUET_PATH / "checkpoint.json"
    PARQUET_PATH.mkdir(parents=True, exist_ok=True)

    chunk_rows = args.chunk_rows if args.chunk_rows is not None else CHUNK_ROWS_30AS
    chunk_plan = None
    if chunk_rows:
        import src.chunked_analysis as chunked

        print(f"Planning country chunks (budget {chunk_rows:,} cells) ...", flush=True)
        chunk_plan = chunked.plan_chunks(
            gis.URB_DATA_FILE,
            max_rows_per_chunk=chunk_rows,
        )
        print(f"  {chunk_plan.describe()}", flush=True)

    run_config = {
        "iterations": args.iterations,
        "zone_file": str(gis.URB_DATA_FILE),
        "workers": workers,
        "chunk_rows": chunk_rows,
        "seed": args.seed,
        "save_zone_parquet": save_zone_parquet,
    }

    print(f"Zone frame: {gis.URB_DATA_FILE}")
    print(f"Workers: {workers}")

    completed_iterations = get_completed_iterations(
        args.iterations, require_zone=save_zone_parquet
    )
    checkpoint = load_checkpoint()

    if completed_iterations and len(completed_iterations) >= args.iterations:
        print("✅ All iterations complete! Processing results...")
        process_saved_results(args.iterations, save_zone_parquet)
        return 0

    if checkpoint and "parameters" in checkpoint:
        params = checkpoint["parameters"]
        _require_borehole_params(params)
        parameters = params
        print("📂 Parameters restored from checkpoint")
    else:
        if args.seed is not None:
            np.random.seed(args.seed)
            print(f"Parameter seed: {args.seed}")
        else:
            print("Parameter seed: none (unseeded draws)")
        drawn = sample_parameters(args.iterations)
        for line in format_parameter_draws(drawn, args.iterations):
            print(line)
        parameters = {key: value.tolist() for key, value in drawn.items()}
        save_checkpoint(completed_iterations, parameters, run_config)

    def array(name):
        return np.array(parameters[name])

    crr_adjustments = array("crr_adjustments")
    time_gatherings = array("time_gatherings")
    practical_limits_bicycle = array("practical_limits_bicycle")
    practical_limits_buckets = array("practical_limits_buckets")
    mets = array("mets")
    watts_values = array("watts_values")
    hill_polarities = array("hill_polarities")
    urban_adjustments = array("urban_adjustments")
    rural_adjustments = array("rural_adjustments")
    urban_borehole_lose_fractions = array("urban_borehole_lose_fractions")
    rural_borehole_lose_fractions = array("rural_borehole_lose_fractions")
    unallocated_vulnerable_fractions = array("unallocated_vulnerable_fractions")

    iterations_to_run = [
        i for i in range(args.iterations) if i not in completed_iterations
    ]
    if not iterations_to_run:
        process_saved_results(args.iterations, save_zone_parquet)
        return 0

    signal.signal(signal.SIGTERM, handle_preemption)

    started = time.time()
    print(f"\n🚀 Starting Monte Carlo simulations...")
    print(
        f"📊 Running {len(iterations_to_run)} remaining iterations "
        f"(out of {args.iterations} total)"
    )
    print(f"🕐 Start time: {time.strftime('%H:%M:%S', time.localtime())}\n")

    jobs = [
        {
            "index": idx,
            "zone_file": str(gis.URB_DATA_FILE),
            "crr_adjustment": crr_adjustments[idx],
            "time_gathering_water": time_gatherings[idx],
            "practical_limit_bicycle": practical_limits_bicycle[idx],
            "practical_limit_buckets": practical_limits_buckets[idx],
            "met": mets[idx],
            "watts": watts_values[idx],
            "hill_polarity": hill_polarities[idx],
            "urban_adjustment": urban_adjustments[idx],
            "rural_adjustment": rural_adjustments[idx],
            "urban_borehole_lose_fraction": urban_borehole_lose_fractions[idx],
            "rural_borehole_lose_fraction": rural_borehole_lose_fractions[idx],
            "unallocated_vulnerable_fraction": unallocated_vulnerable_fractions[idx],
            "save_zone_parquet": save_zone_parquet,
            "chunk_rows": chunk_rows or None,
            "chunk_plan": chunk_plan,
        }
        for idx in iterations_to_run
    ]

    pool = multiprocessing.Pool(processes=workers, maxtasksperchild=1)
    pending = [(job["index"], pool.apply_async(run_and_save_simulation, (job,))) for job in jobs]
    pool.close()

    progress = tqdm(total=len(iterations_to_run), desc="Simulating")
    for idx, async_result in pending:
        if shutdown_requested:
            pool.terminate()
            break
        try:
            completed_index = async_result.get(timeout=ITERATION_TIMEOUT)
            completed_iterations.append(completed_index)
            if len(completed_iterations) % 10 == 0:
                save_checkpoint(completed_iterations, parameters, run_config)
            progress.update()
        except multiprocessing.TimeoutError:
            print(
                f"\n❌ Iteration {idx}: no result within {ITERATION_TIMEOUT}s "
                "(worker may have been killed); continuing"
            )
        except Exception as exc:
            print(f"\n❌ Error in iteration {idx}: {exc}")

    progress.close()
    pool.terminate()
    pool.join()

    if not shutdown_requested:
        save_checkpoint(completed_iterations, parameters, run_config)
        process_saved_results(args.iterations, save_zone_parquet)

    elapsed = time.time() - started
    print(
        f"\n⏱️  Session time: {elapsed / 60:.2f} minutes ({elapsed / 3600:.2f} hours)"
    )
    print(f"✅ Completed {len(completed_iterations)} total iterations")
    if shutdown_requested:
        print("\n⚠️  Interrupted but progress saved. Run again to continue.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
