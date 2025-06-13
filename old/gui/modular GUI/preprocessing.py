"""
preprocessing.py – pure NumPy/SciPy transforms for data shaped
(trials, samples, channels) **or** (samples, channels).
"""

from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt

def _as_3d(arr):
    if arr.ndim == 2:          # (samples, ch) → (1, samples, ch)
        return arr[None, ...]
    return arr                 # already 3‑D

# ------------------------------------------------------------------ filters
def bandpass(arr, **p):
    fs, lo, hi = p["sampling_rate"], p["bpf_low"], p["bpf_high"]
    sos = butter(4, [lo, hi], btype="band", fs=fs, output="sos")
    out = _as_3d(arr).copy()
    out = sosfiltfilt(sos, out, axis=1)      # filter along samples
    return out if arr.ndim == 3 else out[0]

def notch(arr, **p):
    fs, freq = p["sampling_rate"], p["notch_freq"]
    b, a = iirnotch(freq, 30, fs)
    out = _as_3d(arr).copy()
    out = filtfilt(b, a, out, axis=1)
    return out if arr.ndim == 3 else out[0]

def laplacian(arr, **p):             # placeholder
    return arr

# ------------------------------------------------------------------ artefact + z‑score
def artefact_rejection(arr, **p):
    """Hard‑clip any |uV| > 100 to the channel median of that trial."""
    thr = 100.0
    data = _as_3d(arr).copy()
    med = np.median(data, axis=1, keepdims=True)
    mask = np.abs(data) > thr
    data[mask] = np.broadcast_to(med, data.shape)[mask]
    return data if arr.ndim == 3 else data[0]

def zscore(arr, **p):
    data = _as_3d(arr).astype(float)
    m = data.mean(axis=1, keepdims=True)
    s = data.std(axis=1, keepdims=True)
    data = (data - m) / np.where(s == 0, 1, s)
    return data if arr.ndim == 3 else data[0]

# registry
PREPROC_FUNCS = {
    "Band‑pass (BPF)": bandpass,
    "Notch": notch,
    "Laplacian": laplacian,
    "Artifact‑rej.": artefact_rejection,
    "Z‑score": zscore,
}
