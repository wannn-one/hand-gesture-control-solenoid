"""
resource_monitor.py
-------------------
Modul pemantau resource perangkat (CPU%, RAM%) menggunakan psutil.
Berjalan di background thread agar tidak memblokir loop pengujian.

Penggunaan:
    monitor = ResourceMonitor(interval=0.2)
    monitor.start()
    # ... jalankan proses yang ingin diukur ...
    stats = monitor.stop()
    print(stats)
    # {'cpu_avg_pct': 45.2, 'cpu_max_pct': 78.1,
    #  'ram_avg_pct': 62.3, 'ram_max_pct': 64.0}
"""

import threading
import time
from typing import Dict

import psutil


class ResourceMonitor:
    """
    Merekam penggunaan CPU% dan RAM% secara periodik di background thread.
    Mulai dengan start(), hentikan dan ambil statistik dengan stop().
    """

    def __init__(self, interval: float = 0.2) -> None:
        """
        Args:
            interval: Jarak antar sampling dalam detik.
        """
        self.interval = interval
        self._cpu_samples: list[float] = []
        self._ram_samples: list[float] = []
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Mulai sampling resource di background thread."""
        self._cpu_samples = []
        self._ram_samples = []
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def _sample_loop(self) -> None:
        # Panggil sekali dulu agar psutil warm-up (nilai pertama sering 0.0)
        psutil.cpu_percent(interval=None)
        while self._running:
            self._cpu_samples.append(psutil.cpu_percent(interval=None))
            self._ram_samples.append(psutil.virtual_memory().percent)
            time.sleep(self.interval)

    def stop(self) -> Dict[str, float]:
        """
        Hentikan sampling dan kembalikan statistik agregat.

        Returns:
            Dict berisi:
                cpu_avg_pct  – rata-rata CPU usage (%)
                cpu_max_pct  – puncak CPU usage (%)
                ram_avg_pct  – rata-rata RAM usage (%)
                ram_max_pct  – puncak RAM usage (%)
        """
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        def _avg(lst: list[float]) -> float:
            return round(sum(lst) / len(lst), 2) if lst else 0.0

        def _max(lst: list[float]) -> float:
            return round(max(lst), 2) if lst else 0.0

        return {
            "cpu_avg_pct": _avg(self._cpu_samples),
            "cpu_max_pct": _max(self._cpu_samples),
            "ram_avg_pct": _avg(self._ram_samples),
            "ram_max_pct": _max(self._ram_samples),
        }
