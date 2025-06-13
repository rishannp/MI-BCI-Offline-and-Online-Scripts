"""
utils.py – LSL helpers, resource logging, generic I/O.
"""
from __future__ import annotations
import numpy as np, threading, pickle
from time import time
from pathlib import Path
import psutil, pylsl

def resolve_stream(idx=0):
    streams = pylsl.resolve_streams()
    if not streams: raise RuntimeError("No LSL streams.")
    inlet = pylsl.StreamInlet(streams[idx])
    inlet.time_offset = inlet.time_correction()
    return inlet

def pull_chunk(inlet, n_samp, timeout=0.0):
    n_ch = inlet.info().channel_count()
    data = np.zeros((n_ch, n_samp)); ts = np.zeros(n_samp)
    for i in range(n_samp):
        s, t = inlet.pull_sample(timeout=timeout)
        if s: data[:, i], ts[i] = s, t + inlet.time_offset
        else: i -= 1
    return data, ts

class ResourceMonitor(threading.Thread):
    def __init__(self): super().__init__(daemon=True); self.log=[]
    def run(self): self._run=True; 
    def run(self):
        self._run=True
        while self._run:
            self.log.append(dict(t=time(),
                                 cpu=psutil.cpu_percent(interval=1),
                                 ram=psutil.virtual_memory().percent))
    def stop(self): self._run=False

def save_pickle(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f: pickle.dump(obj, f, protocol=4)
