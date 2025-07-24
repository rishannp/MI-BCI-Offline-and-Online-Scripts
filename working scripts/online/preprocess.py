# preprocess.py

import numpy as np
from scipy.signal import butter, filtfilt
from config import SAMPLING_RATE

class Preprocessor:
    def __init__(self, fs=SAMPLING_RATE, lowcut=8, highcut=30, order=4, artifact_threshold=100.0):
        nyq = 0.5 * fs
        low, high = lowcut/nyq, highcut/nyq
        self.b, self.a = butter(order, [low, high], btype="band")
        self.artifact_thresh = artifact_threshold

    def process(self, window: np.ndarray) -> np.ndarray:
        filtered = filtfilt(self.b, self.a, window, axis=0)
        ptp = filtered.max(axis=0) - filtered.min(axis=0)
        if np.any(ptp > self.artifact_thresh):
            return None
        return filtered

# helper for main_online.py
_pre = Preprocessor()
def preprocess_window(window):
    return _pre.process(window)
