#!/usr/bin/env python3
"""
trainer.py – Psychopy + SciPy modular runner.
"""
from __future__ import annotations
import random, numpy as np
from time import time
from pathlib import Path

from utils                import resolve_stream, ResourceMonitor, save_pickle
from preprocessing        import PREPROC_FUNCS
from feature_extractors   import FEAT_FUNCS
from models               import MODEL_FUNCS
from outputs              import OUTPUT_FUNCS

# ------------------------------------------------------------------ PSYCHOPY
from psychopy import visual, core, event                                # docs :contentReference[oaicite:0]{index=0}
import pylsl                                                            # inlet docs :contentReference[oaicite:1]{index=1}

def run_psychopy_acquisition(inlet, pipe, params):
    fs     = params.get("sampling_rate") or inlet.info().nominal_srate()   # fall‑back to stream SR
    n_ch   = inlet.info().channel_count()
    cue_s  = params.get("cue_duration")
    iti_s  = params.get("inter_trial_interval")
    n_cls  = params.get("num_trials_per_class")

    samples_per_cue = int(cue_s * fs)
    total_trials    = n_cls * 3
    data       = np.zeros((total_trials, samples_per_cue, n_ch))
    latencies  = []
    ts_all     = np.zeros((total_trials, samples_per_cue))
    markers    = np.zeros((total_trials, samples_per_cue))

    # === Psychopy stimuli ===
    win = visual.Window(fullscr=False, backend='pygame')
    #win = visual.Window(fullscr=True, color="black")                     # full-screen window :contentReference[oaicite:2]{index=2}
    instruct = visual.TextStim(win, text="Focus on cues.\nPress any key.", color="white")
    fix = visual.TextStim(win, text="+", color="white", height=0.2)
    left  = visual.TextStim(win, text="←", color="white", height=0.5)
    right = visual.TextStim(win, text="→", color="white", height=0.5)
    rest  = visual.TextStim(win, text="Rest", color="white", height=0.5)

    instruct.draw(); win.flip(); event.waitKeys()
    cues = [(left,"Left"),(right,"Right"),(rest,"Rest")]
    label_map = {"Left":1,"Right":2,"Rest":0}

    trial = 0
    for _ in range(n_cls):
        random.shuffle(cues)
        for stim,name in cues:
            print(f"Trial {trial+1}/{total_trials} – {name}")
            if event.getKeys(["escape"]): raise KeyboardInterrupt           # escape key check :contentReference[oaicite:3]{index=3}
            fix.draw(); win.flip(); core.wait(1)                            # fixation cross (1 s) :contentReference[oaicite:4]{index=4}
            t0 = time(); samp = 0
            while samp < samples_per_cue and (time()-t0) < cue_s:
                if event.getKeys(["escape"]): raise KeyboardInterrupt
                stim.draw(); win.flip()
                t_before = pylsl.local_clock()
                s, ts = inlet.pull_sample(timeout=1.0/fs)                  # pull_sample timeout docs :contentReference[oaicite:5]{index=5}
                t_after  = pylsl.local_clock()
                if s:
                    data[trial,samp] = s
                    ts_all[trial,samp] = ts + inlet.time_offset
                    markers[trial,samp] = label_map[name]
                    latencies.append(t_after - t_before)
                    samp += 1
            win.flip(); core.wait(iti_s)
            trial += 1

    win.close(); core.quit()
    return data, ts_all, markers, np.array(latencies)

# ------------------------------------------------------------------ MAIN PIPE
def run_pipeline(pipe, params):
    inlet = resolve_stream(params.get("stream_index",0))
    monitor = ResourceMonitor(); monitor.start()

    raw, ts, markers, lat = run_psychopy_acquisition(inlet, pipe, params)
    monitor.stop()

    # ---- preprocessing chain (works on full trials array) ----
    arr = raw.copy()                     # shape (trials, samples, ch)
    for step in pipe.pre:
        arr = PREPROC_FUNCS[step](arr, **params)

    feats = FEAT_FUNCS[pipe.feat](arr, **params).reshape(arr.shape[0], -1)
    labels = markers[:,0]               # one label per trial
    model, score_fn = MODEL_FUNCS[pipe.model](feats, labels, **params)
    OUTPUT_FUNCS[pipe.output](score_fn(feats, labels), **params)

    out = Path("models")/f"{pipe.model}_{int(time())}.pkl"
    save_pickle(dict(model=model, params=params, pipe=pipe.__dict__,
                     data=arr, timestamps=ts, markers=markers,
                     latencies=lat, resources=monitor.log), out)
    return out, monitor.log
