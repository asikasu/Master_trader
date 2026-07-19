import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


class WalkForwardSplitter:
    def __init__(self, total_bars: int, train_ratio: float = 0.6,
                 val_ratio: float = 0.2, num_windows: int = 3):
        if total_bars < 100:
            raise ValueError(f"total_bars ({total_bars}) too low for walk-forward")
        if train_ratio + val_ratio >= 1.0:
            raise ValueError("train_ratio + val_ratio must be < 1.0")
        self.total_bars = total_bars
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.num_windows = num_windows

    def get_windows(self) -> List[Tuple[int, int, int, int]]:
        train_bars = int(self.total_bars * self.train_ratio)
        val_bars = int(self.total_bars * self.val_ratio)
        step = max(1, (self.total_bars - train_bars - val_bars) // max(1, self.num_windows - 1))

        windows = []
        for i in range(self.num_windows):
            ts = i * step
            te = ts + train_bars
            vs = te
            ve = min(vs + val_bars, self.total_bars)
            if te > self.total_bars or vs > self.total_bars:
                break
            windows.append((ts, te, vs, ve))
        if not windows:
            windows.append((0, train_bars, train_bars, min(train_bars + val_bars, self.total_bars)))
        return windows
