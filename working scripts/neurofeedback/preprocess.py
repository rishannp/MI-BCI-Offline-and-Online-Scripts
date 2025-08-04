import numpy as np
from config import SAMPLING_RATE  # Uncomment if needed

class Preprocessor:
    def __init__(self, artifact_threshold=3000.0):
        self.artifact_thresh = artifact_threshold

        # Full 64-channel headset layout
        self.headset_electrodes = [
            'FP1', 'FPz', 'FP2', 'AF7', 'AF3', 'AF4', 'AF8', 'F7', 'F5', 'F3',
            'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz',
            'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4',
            'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
            'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7',
            'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'F9', 'F10', 'A1', 'A2'
        ]

        # 58 shared electrodes used for training
        self.shared_stieger_electrodes = [
            'FP1', 'FPz', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz',
            'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2',
            'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4',
            'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
            'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7',
            'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2'
        ]

        self.subset_indices = [self.headset_electrodes.index(e) for e in self.shared_stieger_electrodes]

    def process(self, window: np.ndarray) -> np.ndarray:
        """
        Takes raw EEG window of shape [samples, 64],
        returns subset [samples, 58], or None if artifact detected.
        """
        ptp = window.max(axis=0) - window.min(axis=0)
        if np.any(ptp > self.artifact_thresh):
            return None

        subset_window = window[:, self.subset_indices]
        return subset_window


# --------------
# Convenience wrapper for use in main_online.py
# --------------
_pre = Preprocessor()
n_channels = len(_pre.subset_indices)  # For use in PLV visualisation

def preprocess_window(window):
    """
    Wrapper to preprocess a window. Input: [samples, 64]. Output: [samples, 58] or None.
    """
    return _pre.process(window)
