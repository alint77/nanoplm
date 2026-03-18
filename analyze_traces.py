"""Acceptance-oriented analysis for Block AttnRes chrome traces.

Default behavior compares:
- post-phase0 AttnRes-on baseline
- latest current AttnRes-on trace
- latest current AttnRes-off trace

The report focuses on the regression vectors identified in the phase5 traces:
- tiny all_gather_into_tensor_coalesced storms
- FSDP::cast_forward_inputs cost
- kernel-launch pressure
- host synchronizations
- stream-7 compute busy/idle time
- NCCL overlap, especially stream 22 vs stream 7
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_TRACES = {
    "post_phase0_on": Path(
        "output/pretraining_checkpoints/run-17031621/profiler_traces/chrome_trace.json"
    ),
    "current_on": Path(
        "output/pretraining_checkpoints/run-17032053/profiler_traces/chrome_trace.json"
    ),
    "current_off": Path(
        "output/pretraining_checkpoints/run-17032103-2/profiler_traces/chrome_trace.json"
    ),
}
DEFAULT_STEPS = (11, 12, 13, 14)
DEFAULT_MAIN_STREAM = 7
DEFAULT_COMM_STREAM = 22

HOST_SYNC_NAMES = {
    "cudaStreamSynchronize",
    "cudaEventSynchronize",
    "cudaDeviceSynchronize",
}
LAUNCH_NAMES = {
    "cudaLaunchKernel",
    "cudaLaunchKernelExC",
}
ATTNRES_KERNEL_PATTERNS = (
    "attnres",
    "batched_completed",
    "merge_partial",
    "completed_dreduction",
)
STEP_RE = re.compile(r"ProfilerStep#(\d+)")


@dataclass(frozen=True)
class StepWindow:
    step_num: int
    start_us: float
    end_us: float


@dataclass
class TraceSummary:
    label: str
    path: Path
    step_numbers: tuple[int, ...]
    step_ms: float
    stream7_busy_ms: float
    stream7_idle_ms: float
    launch_calls: float
    launch_cpu_ms: float
    host_sync_calls: float
    host_sync_ms: float
    fsdp_cast_calls: float
    fsdp_cast_ms: float
    tiny_allgather_calls: float
    tiny_allgather_ms: float
    nccl_total_ms: float
    nccl_overlap_ms: float
    nccl_uncovered_ms: float
    stream22_nccl_ms: float
    stream22_overlap_pct: float
    top_kernels: list[tuple[str, float, float]]
    attnres_kernels: list[tuple[str, float, float]]


def _load_events(path: Path) -> list[dict]:
    return json.loads(path.read_text())["traceEvents"]


def _parse_step_windows(
    events: list[dict],
    preferred_steps: tuple[int, ...],
) -> list[StepWindow]:
    windows_by_step: dict[int, tuple[int, StepWindow]] = {}
    for ev in events:
        if ev.get("ph") != "X":
            continue
        match = STEP_RE.fullmatch(ev.get("name", ""))
        if match is None:
            continue
        step_num = int(match.group(1))
        start_us = float(ev.get("ts", 0.0))
        dur_us = float(ev.get("dur", 0.0))
        window = StepWindow(
            step_num=step_num,
            start_us=start_us,
            end_us=start_us + dur_us,
        )
        # Prefer the CPU/user-annotation ProfilerStep track (pid != 0) over the
        # mirrored GPU track (pid == 0, tid == stream id).
        priority = 1 if int(ev.get("pid", 0)) != 0 else 0
        prior = windows_by_step.get(step_num)
        if prior is None or priority > prior[0]:
            windows_by_step[step_num] = (priority, window)
    windows = [window for _priority, window in windows_by_step.values()]
    windows.sort(key=lambda window: window.step_num)
    selected = [window for window in windows if window.step_num in preferred_steps]
    if len(selected) >= 2:
        return selected
    return windows[-min(4, len(windows)) :]


def _overlaps_window(event: dict, windows: list[StepWindow]) -> bool:
    start_us = float(event.get("ts", 0.0))
    end_us = start_us + float(event.get("dur", 0.0))
    return any(start_us < window.end_us and end_us > window.start_us for window in windows)


def _events_in_windows(events: list[dict], windows: list[StepWindow]) -> list[dict]:
    return [
        ev
        for ev in events
        if ev.get("ph") == "X" and float(ev.get("dur", 0.0)) >= 0.0 and _overlaps_window(ev, windows)
    ]


def _clipped_intervals(
    events: Iterable[dict],
    window: StepWindow,
    predicate,
) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    for ev in events:
        if not predicate(ev):
            continue
        start_us = max(float(ev.get("ts", 0.0)), window.start_us)
        end_us = min(
            float(ev.get("ts", 0.0)) + float(ev.get("dur", 0.0)),
            window.end_us,
        )
        if end_us > start_us:
            intervals.append((start_us, end_us))
    return intervals


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for start_us, end_us in intervals[1:]:
        last_start, last_end = merged[-1]
        if start_us <= last_end:
            merged[-1] = (last_start, max(last_end, end_us))
        else:
            merged.append((start_us, end_us))
    return merged


def _interval_duration_us(intervals: list[tuple[float, float]]) -> float:
    return sum(end_us - start_us for start_us, end_us in intervals)


def _overlap_duration_us(
    left: list[tuple[float, float]],
    right: list[tuple[float, float]],
) -> float:
    left = _merge_intervals(left)
    right = _merge_intervals(right)
    i = 0
    j = 0
    overlap_us = 0.0
    while i < len(left) and j < len(right):
        start_us = max(left[i][0], right[j][0])
        end_us = min(left[i][1], right[j][1])
        if end_us > start_us:
            overlap_us += end_us - start_us
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return overlap_us


def _flatten_dims(obj) -> Iterable[int]:
    if isinstance(obj, int):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _flatten_dims(item)


def _max_reported_dim(event: dict) -> int:
    dims = list(_flatten_dims(event.get("args", {}).get("Input Dims", [])))
    dims = [int(dim) for dim in dims if isinstance(dim, int)]
    return max(dims) if dims else 0


def _is_kernel(event: dict) -> bool:
    return event.get("cat") == "kernel"


def _is_nccl_kernel(event: dict) -> bool:
    args = event.get("args", {})
    name = event.get("name", "").lower()
    return _is_kernel(event) and (
        "Collective name" in args or name.startswith("nccl")
    )


def _is_tiny_allgather(event: dict) -> bool:
    if event.get("cat") != "user_annotation":
        return False
    name = event.get("name", "").lower()
    if "all_gather_into_tensor_coalesced" not in name and "allgather_into_tensor_coalesced" not in name:
        return False
    max_dim = _max_reported_dim(event)
    return max_dim == 0 or max_dim <= 2048


def _aggregate_event_metric(events: Iterable[dict], predicate) -> tuple[int, float]:
    count = 0
    dur_us = 0.0
    for ev in events:
        if predicate(ev):
            count += 1
            dur_us += float(ev.get("dur", 0.0))
    return count, dur_us


def _aggregate_kernel_table(
    events: Iterable[dict],
    *,
    include_predicate,
    step_count: int,
    limit: int,
) -> list[tuple[str, float, float]]:
    totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for ev in events:
        if not include_predicate(ev):
            continue
        name = ev.get("name", "unknown")
        totals[name][0] += float(ev.get("dur", 0.0))
        totals[name][1] += 1.0
    rows = [
        (name, dur_us / 1000.0 / step_count, count / step_count)
        for name, (dur_us, count) in totals.items()
    ]
    rows.sort(key=lambda row: row[1], reverse=True)
    return rows[:limit]


def summarize_trace(
    label: str,
    path: Path,
    *,
    preferred_steps: tuple[int, ...],
    main_stream: int,
    comm_stream: int,
) -> TraceSummary:
    events = _load_events(path)
    windows = _parse_step_windows(events, preferred_steps)
    if not windows:
        raise RuntimeError(f"No ProfilerStep windows found in {path}")
    window_events = _events_in_windows(events, windows)
    step_count = len(windows)
    step_ms = sum((window.end_us - window.start_us) for window in windows) / 1000.0 / step_count

    stream7_busy_us = 0.0
    nccl_total_us = 0.0
    nccl_overlap_us = 0.0
    stream22_nccl_us = 0.0
    stream22_overlap_us = 0.0
    for window in windows:
        step_events = [
            ev
            for ev in window_events
            if float(ev.get("ts", 0.0)) < window.end_us
            and float(ev.get("ts", 0.0)) + float(ev.get("dur", 0.0)) > window.start_us
        ]
        main_intervals = _merge_intervals(
            _clipped_intervals(
                step_events,
                window,
                lambda ev: _is_kernel(ev) and int(ev.get("tid", -1)) == main_stream,
            )
        )
        nccl_intervals = _merge_intervals(
            _clipped_intervals(step_events, window, _is_nccl_kernel)
        )
        stream22_intervals = _merge_intervals(
            _clipped_intervals(
                step_events,
                window,
                lambda ev: _is_nccl_kernel(ev) and int(ev.get("tid", -1)) == comm_stream,
            )
        )
        stream7_busy_us += _interval_duration_us(main_intervals)
        nccl_total_us += _interval_duration_us(nccl_intervals)
        nccl_overlap_us += _overlap_duration_us(nccl_intervals, main_intervals)
        stream22_nccl_us += _interval_duration_us(stream22_intervals)
        stream22_overlap_us += _overlap_duration_us(stream22_intervals, main_intervals)

    launch_calls, launch_dur_us = _aggregate_event_metric(
        window_events,
        lambda ev: ev.get("name") in LAUNCH_NAMES,
    )
    host_sync_calls, host_sync_dur_us = _aggregate_event_metric(
        window_events,
        lambda ev: ev.get("name") in HOST_SYNC_NAMES,
    )
    fsdp_cast_calls, fsdp_cast_dur_us = _aggregate_event_metric(
        window_events,
        lambda ev: ev.get("name") == "FSDP::cast_forward_inputs",
    )
    tiny_allgather_calls, tiny_allgather_dur_us = _aggregate_event_metric(
        window_events,
        _is_tiny_allgather,
    )

    return TraceSummary(
        label=label,
        path=path,
        step_numbers=tuple(window.step_num for window in windows),
        step_ms=step_ms,
        stream7_busy_ms=stream7_busy_us / 1000.0 / step_count,
        stream7_idle_ms=step_ms - (stream7_busy_us / 1000.0 / step_count),
        launch_calls=launch_calls / step_count,
        launch_cpu_ms=launch_dur_us / 1000.0 / step_count,
        host_sync_calls=host_sync_calls / step_count,
        host_sync_ms=host_sync_dur_us / 1000.0 / step_count,
        fsdp_cast_calls=fsdp_cast_calls / step_count,
        fsdp_cast_ms=fsdp_cast_dur_us / 1000.0 / step_count,
        tiny_allgather_calls=tiny_allgather_calls / step_count,
        tiny_allgather_ms=tiny_allgather_dur_us / 1000.0 / step_count,
        nccl_total_ms=nccl_total_us / 1000.0 / step_count,
        nccl_overlap_ms=nccl_overlap_us / 1000.0 / step_count,
        nccl_uncovered_ms=(nccl_total_us - nccl_overlap_us) / 1000.0 / step_count,
        stream22_nccl_ms=stream22_nccl_us / 1000.0 / step_count,
        stream22_overlap_pct=(
            100.0 * stream22_overlap_us / stream22_nccl_us if stream22_nccl_us > 0 else 0.0
        ),
        top_kernels=_aggregate_kernel_table(
            window_events,
            include_predicate=lambda ev: _is_kernel(ev) and not _is_nccl_kernel(ev),
            step_count=step_count,
            limit=10,
        ),
        attnres_kernels=_aggregate_kernel_table(
            window_events,
            include_predicate=lambda ev: _is_kernel(ev)
            and any(pattern in ev.get("name", "").lower() for pattern in ATTNRES_KERNEL_PATTERNS),
            step_count=step_count,
            limit=10,
        ),
    )


def _print_summary(summary: TraceSummary) -> None:
    print(f"\n{'=' * 96}")
    print(f"{summary.label}: {summary.path}")
    print(
        f"steps={summary.step_numbers} step_ms={summary.step_ms:.2f} "
        f"stream7_busy_ms={summary.stream7_busy_ms:.2f} stream7_idle_ms={summary.stream7_idle_ms:.2f}"
    )
    print(
        f"launches/step={summary.launch_calls:.1f} launch_cpu_ms/step={summary.launch_cpu_ms:.2f} "
        f"host_sync_calls/step={summary.host_sync_calls:.1f} host_sync_ms/step={summary.host_sync_ms:.2f}"
    )
    print(
        f"FSDP::cast_forward_inputs calls/step={summary.fsdp_cast_calls:.1f} "
        f"ms/step={summary.fsdp_cast_ms:.2f}"
    )
    print(
        f"tiny_allgather/step={summary.tiny_allgather_calls:.1f} "
        f"tiny_allgather_ms/step={summary.tiny_allgather_ms:.2f}"
    )
    print(
        f"NCCL total/step={summary.nccl_total_ms:.2f} overlap_with_stream7={summary.nccl_overlap_ms:.2f} "
        f"uncovered={summary.nccl_uncovered_ms:.2f}"
    )
    print(
        f"stream22 NCCL/step={summary.stream22_nccl_ms:.2f} "
        f"overlap_pct={summary.stream22_overlap_pct:.1f}%"
    )

    print("\nTop direct kernels (ms/step, calls/step):")
    for name, ms_per_step, calls_per_step in summary.top_kernels:
        print(f"  {ms_per_step:8.2f} ms  {calls_per_step:7.1f} calls  {name}")

    if summary.attnres_kernels:
        print("\nAttnRes-related kernels (ms/step, calls/step):")
        for name, ms_per_step, calls_per_step in summary.attnres_kernels:
            print(f"  {ms_per_step:8.2f} ms  {calls_per_step:7.1f} calls  {name}")


def _print_delta(title: str, left: TraceSummary, right: TraceSummary) -> None:
    print(f"\n{'-' * 96}")
    print(f"{title}: {right.label} - {left.label}")
    rows = [
        ("step_ms", left.step_ms, right.step_ms),
        ("stream7_busy_ms", left.stream7_busy_ms, right.stream7_busy_ms),
        ("stream7_idle_ms", left.stream7_idle_ms, right.stream7_idle_ms),
        ("launch_calls", left.launch_calls, right.launch_calls),
        ("launch_cpu_ms", left.launch_cpu_ms, right.launch_cpu_ms),
        ("host_sync_ms", left.host_sync_ms, right.host_sync_ms),
        ("fsdp_cast_ms", left.fsdp_cast_ms, right.fsdp_cast_ms),
        ("tiny_allgather_calls", left.tiny_allgather_calls, right.tiny_allgather_calls),
        ("nccl_overlap_ms", left.nccl_overlap_ms, right.nccl_overlap_ms),
        ("nccl_uncovered_ms", left.nccl_uncovered_ms, right.nccl_uncovered_ms),
        ("stream22_overlap_pct", left.stream22_overlap_pct, right.stream22_overlap_pct),
    ]
    for name, left_value, right_value in rows:
        print(f"  {name:<22} {left_value:>10.2f} -> {right_value:>10.2f}   delta={right_value - left_value:+.2f}")


def _evaluate_acceptance(summary: TraceSummary) -> None:
    checks = [
        ("step_ms <= 650", summary.step_ms <= 650.0),
        ("tiny_allgather_calls <= 1", summary.tiny_allgather_calls <= 1.0),
        ("FSDP::cast_forward_inputs ms <= 1", summary.fsdp_cast_ms <= 1.0),
        ("launch_calls <= 2500", summary.launch_calls <= 2500.0),
        ("stream22 overlap >= 25%", summary.stream22_overlap_pct >= 25.0),
    ]
    print(f"\n{'-' * 96}")
    print(f"Acceptance check for {summary.label}")
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")


def _parse_trace_arg(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"Invalid --trace value {spec!r}; expected label=/path/to/chrome_trace.json"
        )
    label, raw_path = spec.split("=", 1)
    return label, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Custom trace to analyze. If omitted, uses the built-in post-phase0/current-on/current-off set.",
    )
    parser.add_argument(
        "--main-stream",
        type=int,
        default=DEFAULT_MAIN_STREAM,
        help="Main compute stream/thread id. Default: 7",
    )
    parser.add_argument(
        "--comm-stream",
        type=int,
        default=DEFAULT_COMM_STREAM,
        help="Communication stream/thread id to track for overlap. Default: 22",
    )
    parser.add_argument(
        "--steps",
        nargs="*",
        type=int,
        default=list(DEFAULT_STEPS),
        help="ProfilerStep numbers to treat as steady-state. Default: 11 12 13 14",
    )
    args = parser.parse_args()

    trace_specs = (
        dict(_parse_trace_arg(spec) for spec in args.trace)
        if args.trace
        else DEFAULT_TRACES
    )

    summaries: list[TraceSummary] = []
    for label, path in trace_specs.items():
        if not path.exists():
            raise FileNotFoundError(path)
        summaries.append(
            summarize_trace(
                label,
                path,
                preferred_steps=tuple(args.steps),
                main_stream=args.main_stream,
                comm_stream=args.comm_stream,
            )
        )

    for summary in summaries:
        _print_summary(summary)

    if len(summaries) >= 2:
        for idx in range(1, len(summaries)):
            _print_delta("Comparison", summaries[0], summaries[idx])

    summary_by_label = {summary.label: summary for summary in summaries}
    candidate = summary_by_label.get("current_on", summaries[-1])
    _evaluate_acceptance(candidate)


if __name__ == "__main__":
    main()
