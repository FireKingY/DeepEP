#!/usr/bin/env python3
"""
Wait for all GPUs to be idle, then run DeepEP intranode performance test.

This script monitors GPU utilization across all B200 GPUs and waits until
they are completely idle (no resource contention), then executes the DeepEP
intranode performance test to obtain accurate NVLink bandwidth measurements.
"""

import argparse
import datetime
import os
import re
import subprocess
import sys
import time


def get_gpu_status() -> list[tuple[int, int, int]]:
    """
    Query GPU utilization and process count for all GPUs.

    Returns:
        List of tuples (gpu_id, utilization_percent, process_count)
    """
    try:
        # Query GPU utilization
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )

        utilization = {}
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split(',')
                if len(parts) >= 2:
                    try:
                        gpu_id = int(parts[0].strip())
                        util = int(parts[1].strip())
                        utilization[gpu_id] = util
                    except ValueError:
                        continue

        # Query compute processes
        process_counts = {gpu_id: 0 for gpu_id in utilization}
        try:
            result = subprocess.run(
                ["nvidia-smi", "pmon", "-s", "u", "-c", "1"],
                capture_output=True,
                text=True
            )

            for line in result.stdout.strip().split('\n'):
                if line.strip() and not line.startswith('#') and not 'gpu' in line.lower():
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            gpu_id = int(parts[0])
                            # Check if there's an active process (pid != '-')
                            if parts[2] != '-' and gpu_id in process_counts:
                                process_counts[gpu_id] += 1
                        except ValueError:
                            continue
        except subprocess.SubprocessError:
            # pmon might fail if no processes, that's ok
            pass

        # Build status list
        return [(gpu_id, utilization[gpu_id], process_counts[gpu_id])
                for gpu_id in sorted(utilization.keys())]

    except subprocess.CalledProcessError as e:
        print(f"[{timestamp()}] [ERROR] Failed to query GPU status: {e}")
        return []


def are_all_gpus_idle(status: list[tuple[int, int, int]], max_utilization: int = 5) -> bool:
    """Check if all GPUs have utilization < max_utilization and no processes."""
    if not status:
        return False
    return all(util < max_utilization and proc_count == 0
               for _, util, proc_count in status)


def format_status(status: list[tuple[int, int, int]]) -> str:
    """Format GPU status for display."""
    return " | ".join(f"GPU{gpu_id}:{util}%/{proc_count}procs"
                      for gpu_id, util, proc_count in status)


def wait_for_idle_gpus(max_utilization: int = 5,
                       poll_interval: int = 10,
                       settling_time: int = 30) -> None:
    """
    Wait until all GPUs are idle.

    Args:
        max_utilization: Maximum utilization percentage to consider idle
        poll_interval: Seconds between GPU status checks
        settling_time: Seconds of continuous idle before confirming
    """
    print(f"[{timestamp()}] Starting GPU idle wait...")
    print(f"[{timestamp()}] Idle threshold: <{max_utilization}% utilization, 0 processes")
    print(f"[{timestamp()}] Poll interval: {poll_interval} seconds")
    print(f"[{timestamp()}] Settling time: {settling_time} seconds")
    print(f"[{timestamp()}] Press Ctrl+C to interrupt\n")

    idle_start_time = None
    wait_start_time = time.time()

    while True:
        status = get_gpu_status()

        if not status:
            print(f"[{timestamp()}] [WARNING] Failed to get GPU status, retrying...")
            time.sleep(poll_interval)
            continue

        current_time = time.time()
        elapsed = current_time - wait_start_time

        if are_all_gpus_idle(status, max_utilization):
            if idle_start_time is None:
                idle_start_time = current_time
                print(f"[{timestamp()}] [{format_elapsed(elapsed)}] GPUs went idle: {format_status(status)}")
            else:
                idle_duration = current_time - idle_start_time
                if idle_duration >= settling_time:
                    print(f"\n[{timestamp()}] [SUCCESS] All GPUs idle for {settling_time}s!")
                    print(f"[{timestamp()}] Total wait time: {format_elapsed(elapsed)}")
                    return
                else:
                    remaining = settling_time - idle_duration
                    print(f"[{timestamp()}] [{format_elapsed(elapsed)}] Idle for {idle_duration:.0f}s, waiting {remaining:.0f}s more... {format_status(status)}")
        else:
            if idle_start_time is not None:
                print(f"[{timestamp()}] [{format_elapsed(elapsed)}] GPUs became busy again: {format_status(status)}")
                idle_start_time = None
            elif int(elapsed) % 60 < poll_interval:
                # Only print every minute to reduce log spam
                print(f"[{timestamp()}] [{format_elapsed(elapsed)}] Waiting... {format_status(status)}")

        time.sleep(poll_interval)


def run_deepep_test(log_file: str, baseline_gb_s: float) -> int:
    """
    Run DeepEP intranode test and capture output.

    Args:
        log_file: Path to log file for output
        baseline_gb_s: Baseline performance for comparison

    Returns:
        Return code from test process
    """
    print(f"\n[{timestamp()}] Running DeepEP intranode test...")
    print(f"[{timestamp()}] Logging to: {log_file}\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))

    cmd = [
        sys.executable, "tests/test_intranode.py",
        "--num-processes", "8",
        "--num-tokens", "4096",
        "--hidden", "7168"
    ]

    try:
        env = os.environ.copy()
        env['PYTHONPATH'] = script_dir + ':' + env.get('PYTHONPATH', '')

        with open(log_file, 'w') as f:
            f.write(f"DeepEP Intranode Test - {timestamp()}\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Baseline: {baseline_gb_s} GB/s\n")
            f.write("=" * 80 + "\n\n")
            f.flush()

            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=script_dir,
                env=env
            )

            f.write(f"\n{'=' * 80}\n")
            f.write(f"Exit code: {result.returncode}\n")
            f.write(f"Completed: {timestamp()}\n")

        return result.returncode

    except Exception as e:
        print(f"[ERROR] Failed to run test: {e}")
        return 1


def timestamp() -> str:
    """Get current timestamp string."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_elapsed(seconds: float) -> str:
    """Format elapsed time as HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def parse_performance_results(log_file: str) -> dict[str, float]:
    """Parse performance numbers from DeepEP test log."""
    results = {}
    try:
        with open(log_file, 'r') as f:
            content = f.read()

        # Parse dispatch BF16 results
        match = re.search(r'Best dispatch \(BF16\):.*?([\d.]+) GB/s', content)
        if match:
            results['dispatch_bf16_gb_s'] = float(match.group(1))

        # Parse dispatch FP8 results
        match = re.search(r'Best dispatch \(FP8\):.*?([\d.]+) GB/s', content)
        if match:
            results['dispatch_fp8_gb_s'] = float(match.group(1))

        # Parse combine results
        match = re.search(r'Best combine:.*?([\d.]+) GB/s', content)
        if match:
            results['combine_gb_s'] = float(match.group(1))

        return results
    except Exception as e:
        print(f"[WARNING] Failed to parse results: {e}")
        return {}


def compare_with_baseline(results: dict[str, float], baseline_gb_s: float = 42.0) -> str:
    """Compare performance results against baseline."""
    if not results or 'dispatch_bf16_gb_s' not in results:
        return "[WARNING] Could not extract performance numbers for comparison"

    new_value = results['dispatch_bf16_gb_s']
    improvement_pct = ((new_value - baseline_gb_s) / baseline_gb_s) * 100

    lines = [
        "=" * 80,
        "PERFORMANCE COMPARISON REPORT",
        "=" * 80,
        f"Baseline (contested conditions): {baseline_gb_s} GB/s",
        f"Current (idle conditions):       {new_value:.2f} GB/s",
        f"Improvement:                     {improvement_pct:+.1f}%",
        ""
    ]

    if improvement_pct > 20:
        lines.append(f"Significant improvement: {improvement_pct:.1f}% faster than baseline!")
        lines.append("   GPUs were likely uncontested during test.")
    elif improvement_pct > 0:
        lines.append(f"Marginal improvement: {improvement_pct:.1f}% (may still have some contention)")
    else:
        lines.append(f"Degradation: {abs(improvement_pct):.1f}% slower than baseline")
        lines.append("   Check for other system load or configuration issues.")

    if 'dispatch_fp8_gb_s' in results:
        lines.append(f"\nFP8 Dispatch: {results['dispatch_fp8_gb_s']:.2f} GB/s")
    if 'combine_gb_s' in results:
        lines.append(f"Combine:      {results['combine_gb_s']:.2f} GB/s")

    lines.append("=" * 80)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Wait for GPUs to be idle, then run DeepEP performance test"
    )
    parser.add_argument(
        "--max-utilization",
        type=int,
        default=5,
        help="Maximum GPU utilization percentage to consider idle (default: 5)"
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=10,
        help="Seconds between GPU status checks (default: 10)"
    )
    parser.add_argument(
        "--settling-time",
        type=int,
        default=30,
        help="Seconds of continuous idle before starting test (default: 30)"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Log file path (default: deepep_idle_test_YYYYMMDD_HHMMSS.log)"
    )
    parser.add_argument(
        "--baseline-gb-s",
        type=float,
        default=42.0,
        help="Baseline performance in GB/s for comparison (default: 42.0)"
    )

    args = parser.parse_args()

    # Generate log filename if not provided
    if args.log_file is None:
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_file = f"deepep_idle_test_{timestamp_str}.log"

    # Wait for GPUs to be idle
    wait_for_idle_gpus(
        max_utilization=args.max_utilization,
        poll_interval=args.poll_interval,
        settling_time=args.settling_time
    )

    # Run DeepEP test
    exit_code = run_deepep_test(args.log_file, args.baseline_gb_s)

    if exit_code == 0:
        print(f"\n[{timestamp()}] [SUCCESS] Test completed successfully")
        print(f"[{timestamp()}] Results saved to: {args.log_file}")

        # Parse and compare results
        results = parse_performance_results(args.log_file)
        comparison_report = compare_with_baseline(results, baseline_gb_s=args.baseline_gb_s)

        # Print to console
        print("\n" + comparison_report)

        # Append to log file
        with open(args.log_file, 'a') as f:
            f.write("\n" + comparison_report + "\n")
    else:
        print(f"\n[{timestamp()}] [FAILED] Test exited with code {exit_code}")
        print(f"[{timestamp()}] Check log file: {args.log_file}")

    return exit_code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n[{timestamp()}] [ABORTED] Interrupted by user")
        sys.exit(1)
