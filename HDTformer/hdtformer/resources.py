from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import asdict, dataclass

import psutil
import torch


@dataclass
class ResourceStats:
    elapsed_seconds: float = 0.0
    peak_process_cpu_percent: float = 0.0
    peak_system_cpu_percent: float = 0.0
    peak_process_rss_mb: float = 0.0
    peak_process_tree_pss_mb: float = 0.0
    peak_gpu_utilization_percent: float = 0.0
    peak_gpu_memory_used_mb_device: float = 0.0
    peak_torch_memory_allocated_mb: float = 0.0
    peak_torch_memory_reserved_mb: float = 0.0


class ResourceMonitor:
    """Sample process/host resources and CUDA usage during one measured phase."""

    def __init__(self, interval: float = 0.2) -> None:
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.stats = ResourceStats()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time = 0.0
        self._last_tree_cpu_seconds: float | None = None
        self._last_tree_sample_time: float | None = None

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self._prime_process_tree()
        psutil.cpu_percent(None)
        self._start_time = time.perf_counter()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.stats.elapsed_seconds = time.perf_counter() - self._start_time
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval * 2))
        self._sample_once()
        if torch.cuda.is_available():
            self.stats.peak_torch_memory_allocated_mb = (
                torch.cuda.max_memory_allocated() / 1024**2
            )
            self.stats.peak_torch_memory_reserved_mb = (
                torch.cuda.max_memory_reserved() / 1024**2
            )

    def _sample_loop(self) -> None:
        while not self._stop_event.wait(self.interval):
            self._sample_once()

    def _sample_once(self) -> None:
        try:
            now = time.perf_counter()
            processes = [self.process, *self.process.children(recursive=True)]
            cpu_seconds = 0.0
            tree_pss_bytes = 0
            for process in processes:
                try:
                    cpu = process.cpu_times()
                    cpu_seconds += cpu.user + cpu.system
                    memory = process.memory_full_info()
                    tree_pss_bytes += getattr(memory, "pss", memory.rss)
                except (psutil.Error, OSError):
                    continue
            if self._last_tree_cpu_seconds is not None and self._last_tree_sample_time is not None:
                elapsed = now - self._last_tree_sample_time
                if elapsed > 0:
                    tree_cpu_percent = max(
                        0.0,
                        100.0 * (cpu_seconds - self._last_tree_cpu_seconds) / elapsed,
                    )
                    self.stats.peak_process_cpu_percent = max(
                        self.stats.peak_process_cpu_percent, tree_cpu_percent
                    )
            self._last_tree_cpu_seconds = cpu_seconds
            self._last_tree_sample_time = now
            self.stats.peak_process_cpu_percent = max(
                self.stats.peak_process_cpu_percent,
                0.0,
            )
            self.stats.peak_system_cpu_percent = max(
                self.stats.peak_system_cpu_percent,
                psutil.cpu_percent(None),
            )
            rss_mb = self.process.memory_info().rss / 1024**2
            self.stats.peak_process_rss_mb = max(
                self.stats.peak_process_rss_mb, rss_mb
            )
            tree_pss_mb = tree_pss_bytes / 1024**2
            self.stats.peak_process_tree_pss_mb = max(
                self.stats.peak_process_tree_pss_mb, tree_pss_mb
            )
        except (psutil.Error, OSError):
            pass
        self._sample_nvidia_smi()

    def _prime_process_tree(self) -> None:
        now = time.perf_counter()
        cpu_seconds = 0.0
        for process in [self.process, *self.process.children(recursive=True)]:
            try:
                cpu = process.cpu_times()
                cpu_seconds += cpu.user + cpu.system
            except (psutil.Error, OSError):
                continue
        self._last_tree_cpu_seconds = cpu_seconds
        self._last_tree_sample_time = now

    def _sample_nvidia_smi(self) -> None:
        if not torch.cuda.is_available():
            return
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
            device_index = torch.cuda.current_device()
            line = result.stdout.strip().splitlines()[device_index]
            utilization, memory = (float(value.strip()) for value in line.split(","))
            self.stats.peak_gpu_utilization_percent = max(
                self.stats.peak_gpu_utilization_percent, utilization
            )
            self.stats.peak_gpu_memory_used_mb_device = max(
                self.stats.peak_gpu_memory_used_mb_device, memory
            )
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            pass

    def to_dict(self) -> dict[str, float]:
        return asdict(self.stats)
