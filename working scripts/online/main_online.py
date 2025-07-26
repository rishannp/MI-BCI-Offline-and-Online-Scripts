# main_online.py

from game import run_game
from lsl_stream import stream_data
from preprocess import preprocess_window
from featandclass import BCIPipeline, n_channels
from queue import Queue
from threading import Thread
import collections
import time

# for PLV visualisation
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from config import VISUALISE_PLV

def main():
    METHOD = 'plv'  # 'plv' or 'csp'
    pipeline = BCIPipeline(method=METHOD)

    action_queue = Queue()
    label_queue  = Queue()
    adapt_queue  = Queue()

    game_states  = []
    raw_eeg_log  = collections.deque(maxlen=1)

    # ─── Live PLV Visualisation Thread ─────────────────────────────
    def plv_visualiser():
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(np.zeros((n_channels, n_channels)), 
                       cmap='viridis', vmin=0, vmax=1)
        ax.set_title("Live PLV Matrix")
        plt.colorbar(im, ax=ax)
        plt.ion()
        plt.show()

        last = None
        while True:
            plv = pipeline.latest_plv
            if plv is not None and not np.array_equal(plv, last):
                im.set_data(plv)
                last = plv.copy()
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
            time.sleep(0.1)

    # ─── BCI Loop Thread ───────────────────────────────────────────
    def bci_loop():
        while True:
            window = stream_data()
            processed = preprocess_window(window)
            if processed is None:
                continue

            # snapshot for game
            raw_eeg_log.clear()
            raw_eeg_log.append(processed)

            # prediction
            cmd = pipeline.predict(processed)
            action_queue.put(cmd)

            # adaptation on all windows of correct trials
            if pipeline.adaptive and not label_queue.empty():
                label, trial_windows = label_queue.get()
                # use all windows from that trial
                for w in trial_windows:
                    pipeline._win_buf.append(w)
                    pipeline._lab_buf.append(label)
                if len(pipeline._win_buf) >= pipeline.adapt_N:
                    t0 = time.perf_counter()
                    pipeline.adapt()
                    t1 = time.perf_counter()
                    adapt_queue.put(int((t1 - t0) * 1000))

    # start threads
    Thread(target=bci_loop, daemon=True).start()
    if VISUALISE_PLV:
        Thread(target=plv_visualiser, daemon=True).start()

    # launch game
    run_game(action_queue, adapt_queue, game_states, label_queue, raw_eeg_log)

if __name__ == '__main__':
    main()
