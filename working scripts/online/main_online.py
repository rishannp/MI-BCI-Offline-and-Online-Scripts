# main_online.py

import json
import time
import collections
from queue import Queue
from threading import Thread

from config import METHOD, SUBJECT_DIR, SESSION_DIR, VISUALISE_PLV
from lsl_stream import stream_data
from preprocess import preprocess_window
from featandclass import BCIPipeline, n_channels
from game import run_game

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    # write out config snapshot
    with open(f"{SESSION_DIR}/config.json", 'w') as j:
        json.dump({k:repr(v) for k,v in vars(__import__('config')).items()
                   if k.isupper()}, j, indent=2)

    pipeline = BCIPipeline(method=METHOD)
    action_queue = Queue()
    label_queue  = Queue()
    adapt_queue  = Queue()
    game_states  = []
    raw_eeg_log  = collections.deque(maxlen=1)

    # PLV visualiser
    def plv_visualiser():
        fig, ax = plt.subplots(figsize=(6,6))
        im = ax.imshow(np.zeros((n_channels,n_channels)), cmap='viridis', vmin=0, vmax=1)
        ax.set_title("Live PLV")
        plt.colorbar(im, ax=ax)
        plt.ion(); plt.show()
        last = None
        while True:
            plv = pipeline.latest_plv
            if plv is not None and not np.array_equal(plv,last):
                im.set_data(plv); last = plv.copy()
                fig.canvas.draw_idle(); fig.canvas.flush_events()
            time.sleep(0.1)

    # BCI loop
    def bci_loop():
        while True:
            win = stream_data()
            proc = preprocess_window(win)
            if proc is None: continue
            raw_eeg_log.clear(); raw_eeg_log.append(proc)
            cmd = pipeline.predict(proc); action_queue.put(cmd)
            if pipeline.adaptive and not label_queue.empty():
                lbl, windows = label_queue.get()
                for w in windows:
                    pipeline._win_buf.append(w); pipeline._lab_buf.append(lbl)
                if len(pipeline._win_buf) >= pipeline.adapt_N:
                    t0 = time.perf_counter(); pipeline.adapt(); t1 = time.perf_counter()
                    adapt_queue.put(int((t1-t0)*1000))

    Thread(target=bci_loop, daemon=True).start()
    if VISUALISE_PLV and METHOD=='plv':
        Thread(target=plv_visualiser, daemon=True).start()

    run_game(action_queue, adapt_queue, game_states, label_queue, raw_eeg_log)

if __name__=='__main__':
    main()
