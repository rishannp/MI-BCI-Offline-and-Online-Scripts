import json
import time
import collections
from threading import Thread
from queue import Queue

from config import SESSION_DIR, VISUALISE_PLV
from lsl_stream import stream_data
from preprocess import preprocess_window, n_channels
from game import run_game

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

def compute_plv(eeg_window: np.ndarray) -> np.ndarray:
    phase = np.angle(hilbert(eeg_window, axis=0))
    C = eeg_window.shape[1]
    plv = np.zeros((C, C))
    for i in range(C):
        for j in range(i+1, C):
            dphi = phase[:, j] - phase[:, i]
            plv[i, j] = plv[j, i] = abs(np.exp(1j * dphi).mean())
    return plv

def main():
    # Save config snapshot
    with open(f"{SESSION_DIR}/config.json", 'w') as j:
        import config
        json.dump({k: repr(v) for k, v in vars(config).items() if k.isupper()}, j, indent=2)

    action_queue = Queue()
    label_queue = Queue()
    raw_eeg_log = collections.deque(maxlen=1)
    latest_plv   = collections.deque(maxlen=1)

    # EEG streaming thread
    def eeg_loop():
        while True:
            win = stream_data()
            print(f"[DEBUG] Raw window shape: {win.shape} | Mean: {win.mean():.2f}")

            proc = preprocess_window(win)
            if proc is None:
                print("[DEBUG] Window rejected due to artifacts.")
                continue
            print(f"[DEBUG] Preprocessed window shape: {proc.shape}")

            raw_eeg_log.clear()
            raw_eeg_log.append(proc)

            plv = compute_plv(proc)
            latest_plv.clear()
            latest_plv.append(plv)

            print(f"[DEBUG] PLV min={plv.min():.3f}, max={plv.max():.3f}, mean={plv.mean():.3f}")

    # Live visualiser
    def plv_visualiser():
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(np.zeros((n_channels, n_channels)), cmap='hot', vmin=0, vmax=1)
        ax.set_title("Live PLV Matrix")
        plt.colorbar(im, ax=ax)
        plt.ion()
        plt.show()
        last = None

        while True:
            if latest_plv:
                plv = latest_plv[0]
                if not np.array_equal(plv, last):
                    im.set_data(plv)
                    last = plv.copy()
                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()
            plt.pause(0.001)
            time.sleep(0.1)

    # Start threads
    Thread(target=eeg_loop, daemon=True).start()
    if VISUALISE_PLV:
        time.sleep(1)
        Thread(target=plv_visualiser, daemon=True).start()

    run_game(action_queue, label_queue, raw_eeg_log)

if __name__ == '__main__':
    main()
