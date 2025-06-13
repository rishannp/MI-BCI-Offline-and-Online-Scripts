import numpy as np
from scipy.signal import butter, filtfilt

class Preprocessor:
    def __init__(self, fs, lowcut=8, highcut=30, order=4, artifact_threshold=100.0):
        """
        I set up a Butterworth bandpass filter (default 8–30 Hz)
        and an artifact rejection threshold in microvolts.
        """
        nyq = 0.5 * fs
        low, high = lowcut/nyq, highcut/nyq
        self.b, self.a = butter(order, [low, high], btype="band")
        # threshold (µV) above which windows are rejected
        self.artifact_thresh = artifact_threshold

    def process(self, window: np.ndarray) -> np.ndarray:
        """
        I apply bandpass filtering to each channel, then
        reject the window if any sample exceeds the threshold.

        Args:
            window: shape (n_samples, n_channels)

        Returns:
            filtered window, or None if artifact detected
        """
        # bandpass filter
        filtered = filtfilt(self.b, self.a, window, axis=0)

        # artifact check: peak‐to‐peak amplitude
        peak = np.max(filtered, axis=0)
        trough = np.min(filtered, axis=0)
        ptp = peak - trough

        # if any channel’s p-to-p amplitude > threshold, reject
        if np.any(ptp > self.artifact_thresh):
            # drop this window
            return None

        return filtered
