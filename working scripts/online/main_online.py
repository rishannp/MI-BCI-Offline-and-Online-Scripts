from game import run_game
from lsl_stream import stream_data
from preprocess import preprocess_window
from featandclass import BCIPipeline, n_channels
from queue import Queue
from threading import Thread
import collections
import numpy as np
import time

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from config import VISUALISE_PLV

def main():
    METHOD = 'plv'  # 'plv' or 'csp'
    pipeline = BCIPipeline(method=METHOD)
    action_queue = Queue()
    label_queue  = Queue()
    adapt_queue  = Queue()

    game_states  = []
    raw_eeg_log  = collections.deque(maxlen=1)
    latest_processed = None

    def bci_loop():
        nonlocal latest_processed
        while True:
            window = stream_data()
            processed = preprocess_window(window)
            if processed is None:
                continue
            latest_processed = processed

            cmd = pipeline.predict(processed)
            action_queue.put(cmd)

            if pipeline.adaptive and not label_queue.empty():
                label = label_queue.get()
                if cmd == label:
                    pipeline._win_buf.append(processed)
                    pipeline._lab_buf.append(label)
                if len(pipeline._win_buf) >= pipeline.adapt_N:
                    pipeline.adapt()
                    adapt_queue.put(True)

            raw_eeg_log.clear()
            raw_eeg_log.append(processed)

    def plv_visualiser():
        fig, ax = plt.subplots(figsize=(6, 6))  # width, height in inches
        im = ax.imshow(np.zeros((n_channels, n_channels)), cmap='viridis', vmin=0, vmax=1)
        ax.set_title("Live PLV Matrix")
        cbar = plt.colorbar(im, ax=ax)
        plt.show(block=False)

        last_plv = None
        while True:
            if pipeline.latest_plv is not None and not np.array_equal(pipeline.latest_plv, last_plv):
                im.set_data(pipeline.latest_plv)
                last_plv = pipeline.latest_plv.copy()
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
            time.sleep(0.1)

    Thread(target=bci_loop, daemon=True).start()

    if VISUALISE_PLV:
        Thread(target=plv_visualiser, daemon=True).start()

    run_game(action_queue, adapt_queue, game_states, label_queue, raw_eeg_log)

if __name__ == '__main__':
    main()
