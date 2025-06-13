from __future__ import annotations
import numpy as np
from scipy.signal import welch

def psd(arr, **p):
    fs = p["sampling_rate"]; fmin, fmax = p.get("fmin", 8), p.get("fmax", 30)
    arr = arr.mean(0) if arr.ndim == 3 else arr             # collapse trials
    freqs, pxx = welch(arr.T, fs=fs, nperseg=256, axis=0)   # welch expects (samples, ch)
    band = pxx[(freqs >= fmin) & (freqs <= fmax)]
    return band.mean(0)

def csp(arr, **p): return arr.var(axis=(0,1))
def riemann(arr, **p): return arr.mean(axis=(0,1))

FEAT_FUNCS = {"PSD": psd, "CSP": csp, "Riemann": riemann, "None": lambda a, **k: a.mean(0)}
