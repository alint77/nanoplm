#!/usr/bin/env python3
"""
Analyze chrome_trace.json files to compare backward kernel performance
for attnres kernels on thread 7 (main GPU compute thread).
"""

import json
import sys
from collections import defaultdict


BASELINE_PATH = "/workspace/nanoplm/output/pretraining_checkpoints/run-17031118-4/profiler_traces/chrome_trace.json"
LATEST_PATH = "/workspace/nanoplm/output/pretraining_checkpoints/run-17031247/profiler_traces/chrome_trace.json"

FOCUS_STEPS = ["ProfilerStep#11", "ProfilerStep#12", "ProfilerStep#13", "ProfilerStep#14"]

KERNEL_CATEGORIES = [
    "_block_attnres_state_bwd_kernel",
    "_block_attnres_state_fwd_kernel",
]


def load_trace(path):
    """Load a chrome trace JSON file."""
    print(f"Loading {path} ...")
    with open(path) as f:
        data = json.load(f)
    return data["traceEvents"]


def analyze_trace(events, label):
    """Analyze a single trace's events on tid 7."""
    # 1. Collect ProfilerStep events on tid 7
    profiler_steps = []
    for e in events:
        if (
            e.get("tid") == 7
            and e.get("ph") == "X"
            and e.get("name", "").startswith("ProfilerStep#")
        ):
            profiler_steps.append(e)

    # Sort by timestamp
    profiler_steps.sort(key=lambda e: e["ts"])

    # Build step ranges: {step_name: (start_ts, end_ts, dur)}
    step_ranges = {}
    for step in profiler_steps:
        name = step["name"]
        ts_start = step["ts"]
        ts_end = ts_start + step["dur"]
        step_ranges[name] = (ts_start, ts_end, step["dur"])

    # 2. Collect all attnres kernel events on tid 7
    attnres_events = []
    for e in events:
        if (
            e.get("tid") == 7
            and e.get("ph") == "X"
            and "attnres" in e.get("name", "").lower()
        ):
            attnres_events.append(e)

    # 3. Assign each attnres event to a profiler step
    # An event belongs to a step if its start timestamp falls within the step's range
    step_kernel_durations = defaultdict(lambda: defaultdict(list))

    for e in attnres_events:
        e_ts = e["ts"]
        e_name = e["name"]
        e_dur = e["dur"]
        for step_name, (s_start, s_end, _) in step_ranges.items():
            if s_start <= e_ts < s_end:
                step_kernel_durations[step_name][e_name].append(e_dur)
                break

    return step_ranges, step_kernel_durations, attnres_events


def print_report(label, step_ranges, step_kernel_durations):
    """Print a detailed report for one trace."""
    print()
    print("=" * 90)
    print(f"  TRACE: {label}")
    print("=" * 90)

    # Print step durations
    print()
    print(f"  {'Step':<20} {'Step Duration (ms)':>20}")
    print(f"  {'-'*20} {'-'*20}")
    step_durations_ms = {}
    for step_name in FOCUS_STEPS:
        if step_name in step_ranges:
            dur_ms = step_ranges[step_name][2] / 1000.0
            step_durations_ms[step_name] = dur_ms
            print(f"  {step_name:<20} {dur_ms:>20.3f}")
    if step_durations_ms:
        avg = sum(step_durations_ms.values()) / len(step_durations_ms)
        print(f"  {'AVERAGE':<20} {avg:>20.3f}")

    # Print kernel-level breakdown per step
    for kernel_name in KERNEL_CATEGORIES:
        print()
        print(f"  --- {kernel_name} ---")
        print(f"  {'Step':<20} {'Count':>8} {'Total (ms)':>14} {'Avg/call (us)':>16} {'% of step':>12}")
        print(f"  {'-'*20} {'-'*8} {'-'*14} {'-'*16} {'-'*12}")

        totals_ms = []
        counts = []
        for step_name in FOCUS_STEPS:
            durs = step_kernel_durations.get(step_name, {}).get(kernel_name, [])
            count = len(durs)
            total_us = sum(durs)
            total_ms = total_us / 1000.0
            avg_us = total_us / count if count > 0 else 0.0
            step_dur_ms = step_durations_ms.get(step_name, 1.0)
            pct = (total_ms / step_dur_ms) * 100.0 if step_dur_ms > 0 else 0.0

            totals_ms.append(total_ms)
            counts.append(count)
            print(
                f"  {step_name:<20} {count:>8} {total_ms:>14.3f} {avg_us:>16.3f} {pct:>11.2f}%"
            )

        if totals_ms:
            avg_total = sum(totals_ms) / len(totals_ms)
            avg_count = sum(counts) / len(counts)
            avg_per_call = (sum(totals_ms) * 1000.0 / sum(counts)) if sum(counts) > 0 else 0
            avg_step = sum(step_durations_ms.values()) / len(step_durations_ms) if step_durations_ms else 1.0
            avg_pct = (avg_total / avg_step) * 100.0
            print(
                f"  {'AVERAGE':<20} {avg_count:>8.1f} {avg_total:>14.3f} {avg_per_call:>16.3f} {avg_pct:>11.2f}%"
            )

    # Combined attnres total
    print()
    print(f"  --- ALL attnres kernels combined ---")
    print(f"  {'Step':<20} {'Count':>8} {'Total (ms)':>14} {'% of step':>12}")
    print(f"  {'-'*20} {'-'*8} {'-'*14} {'-'*12}")
    combined_totals = []
    combined_counts = []
    for step_name in FOCUS_STEPS:
        all_durs = []
        for kernel_name in KERNEL_CATEGORIES:
            all_durs.extend(step_kernel_durations.get(step_name, {}).get(kernel_name, []))
        count = len(all_durs)
        total_ms = sum(all_durs) / 1000.0
        step_dur_ms = step_durations_ms.get(step_name, 1.0)
        pct = (total_ms / step_dur_ms) * 100.0 if step_dur_ms > 0 else 0.0
        combined_totals.append(total_ms)
        combined_counts.append(count)
        print(f"  {step_name:<20} {count:>8} {total_ms:>14.3f} {pct:>11.2f}%")

    if combined_totals:
        avg_total = sum(combined_totals) / len(combined_totals)
        avg_count = sum(combined_counts) / len(combined_counts)
        avg_step = sum(step_durations_ms.values()) / len(step_durations_ms) if step_durations_ms else 1.0
        avg_pct = (avg_total / avg_step) * 100.0
        print(f"  {'AVERAGE':<20} {avg_count:>8.1f} {avg_total:>14.3f} {avg_pct:>11.2f}%")

    return step_durations_ms


def print_comparison(baseline_ranges, baseline_kernels, latest_ranges, latest_kernels,
                     baseline_step_ms, latest_step_ms):
    """Print side-by-side comparison."""
    print()
    print()
    print("=" * 100)
    print("  COMPARISON: BASELINE vs LATEST")
    print("=" * 100)

    # Step time comparison
    print()
    print(f"  {'Step':<20} {'Baseline (ms)':>15} {'Latest (ms)':>15} {'Delta (ms)':>12} {'Delta %':>10}")
    print(f"  {'-'*20} {'-'*15} {'-'*15} {'-'*12} {'-'*10}")
    base_vals = []
    latest_vals = []
    for step_name in FOCUS_STEPS:
        b = baseline_step_ms.get(step_name, 0)
        l = latest_step_ms.get(step_name, 0)
        delta = l - b
        pct = (delta / b * 100.0) if b > 0 else 0
        base_vals.append(b)
        latest_vals.append(l)
        print(f"  {step_name:<20} {b:>15.3f} {l:>15.3f} {delta:>12.3f} {pct:>9.2f}%")
    if base_vals and latest_vals:
        avg_b = sum(base_vals) / len(base_vals)
        avg_l = sum(latest_vals) / len(latest_vals)
        delta = avg_l - avg_b
        pct = (delta / avg_b * 100.0) if avg_b > 0 else 0
        print(f"  {'AVERAGE':<20} {avg_b:>15.3f} {avg_l:>15.3f} {delta:>12.3f} {pct:>9.2f}%")

    # Per-kernel comparison
    for kernel_name in KERNEL_CATEGORIES:
        print()
        print(f"  --- {kernel_name} ---")
        print(f"  {'Step':<20} {'Base Tot(ms)':>14} {'Late Tot(ms)':>14} {'Delta(ms)':>11} {'Delta %':>10}  {'Base #':>7} {'Late #':>7}")
        print(f"  {'-'*20} {'-'*14} {'-'*14} {'-'*11} {'-'*10}  {'-'*7} {'-'*7}")

        b_totals = []
        l_totals = []
        b_counts_list = []
        l_counts_list = []
        for step_name in FOCUS_STEPS:
            b_durs = baseline_kernels.get(step_name, {}).get(kernel_name, [])
            l_durs = latest_kernels.get(step_name, {}).get(kernel_name, [])
            b_ms = sum(b_durs) / 1000.0
            l_ms = sum(l_durs) / 1000.0
            delta = l_ms - b_ms
            pct = (delta / b_ms * 100.0) if b_ms > 0 else 0
            b_totals.append(b_ms)
            l_totals.append(l_ms)
            b_counts_list.append(len(b_durs))
            l_counts_list.append(len(l_durs))
            print(
                f"  {step_name:<20} {b_ms:>14.3f} {l_ms:>14.3f} {delta:>11.3f} {pct:>9.2f}%  {len(b_durs):>7} {len(l_durs):>7}"
            )
        if b_totals and l_totals:
            avg_b = sum(b_totals) / len(b_totals)
            avg_l = sum(l_totals) / len(l_totals)
            delta = avg_l - avg_b
            pct = (delta / avg_b * 100.0) if avg_b > 0 else 0
            avg_bc = sum(b_counts_list) / len(b_counts_list)
            avg_lc = sum(l_counts_list) / len(l_counts_list)
            print(
                f"  {'AVERAGE':<20} {avg_b:>14.3f} {avg_l:>14.3f} {delta:>11.3f} {pct:>9.2f}%  {avg_bc:>7.1f} {avg_lc:>7.1f}"
            )

    # Combined attnres comparison
    print()
    print(f"  --- ALL attnres kernels combined ---")
    print(f"  {'Step':<20} {'Base Tot(ms)':>14} {'Late Tot(ms)':>14} {'Delta(ms)':>11} {'Delta %':>10}")
    print(f"  {'-'*20} {'-'*14} {'-'*14} {'-'*11} {'-'*10}")
    b_combined = []
    l_combined = []
    for step_name in FOCUS_STEPS:
        b_all = []
        l_all = []
        for kernel_name in KERNEL_CATEGORIES:
            b_all.extend(baseline_kernels.get(step_name, {}).get(kernel_name, []))
            l_all.extend(latest_kernels.get(step_name, {}).get(kernel_name, []))
        b_ms = sum(b_all) / 1000.0
        l_ms = sum(l_all) / 1000.0
        delta = l_ms - b_ms
        pct = (delta / b_ms * 100.0) if b_ms > 0 else 0
        b_combined.append(b_ms)
        l_combined.append(l_ms)
        print(f"  {step_name:<20} {b_ms:>14.3f} {l_ms:>14.3f} {delta:>11.3f} {pct:>9.2f}%")
    if b_combined and l_combined:
        avg_b = sum(b_combined) / len(b_combined)
        avg_l = sum(l_combined) / len(l_combined)
        delta = avg_l - avg_b
        pct = (delta / avg_b * 100.0) if avg_b > 0 else 0
        print(f"  {'AVERAGE':<20} {avg_b:>14.3f} {avg_l:>14.3f} {delta:>11.3f} {pct:>9.2f}%")

    # Per-call average comparison
    print()
    print(f"  --- Per-call average duration (us) ---")
    for kernel_name in KERNEL_CATEGORIES:
        b_all_durs = []
        l_all_durs = []
        for step_name in FOCUS_STEPS:
            b_all_durs.extend(baseline_kernels.get(step_name, {}).get(kernel_name, []))
            l_all_durs.extend(latest_kernels.get(step_name, {}).get(kernel_name, []))
        b_avg = sum(b_all_durs) / len(b_all_durs) if b_all_durs else 0
        l_avg = sum(l_all_durs) / len(l_all_durs) if l_all_durs else 0
        delta = l_avg - b_avg
        pct = (delta / b_avg * 100.0) if b_avg > 0 else 0
        print(f"  {kernel_name}")
        print(f"    Baseline: {b_avg:>10.3f} us/call  ({len(b_all_durs)} calls across steps 11-14)")
        print(f"    Latest:   {l_avg:>10.3f} us/call  ({len(l_all_durs)} calls across steps 11-14)")
        print(f"    Delta:    {delta:>10.3f} us/call  ({pct:>+.2f}%)")
        print()


def main():
    # Load traces
    baseline_events = load_trace(BASELINE_PATH)
    latest_events = load_trace(LATEST_PATH)

    # Analyze each trace
    baseline_ranges, baseline_kernels, baseline_attnres = analyze_trace(baseline_events, "BASELINE")
    latest_ranges, latest_kernels, latest_attnres = analyze_trace(latest_events, "LATEST")

    # Print individual reports
    baseline_step_ms = print_report("BASELINE (run-17031118-4)", baseline_ranges, baseline_kernels)
    latest_step_ms = print_report("LATEST (run-17031247)", latest_ranges, latest_kernels)

    # Print comparison
    print_comparison(
        baseline_ranges, baseline_kernels,
        latest_ranges, latest_kernels,
        baseline_step_ms, latest_step_ms,
    )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
