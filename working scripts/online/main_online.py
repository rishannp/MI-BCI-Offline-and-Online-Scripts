# main_online.py

from game import run_game
from lsl_stream import stream_data
from preprocess import preprocess_window
from featandclass import BCIPipeline
from queue import Queue
from threading import Thread
import collections

def main():
    METHOD = 'plv'  # 'plv' or 'csp'
    pipeline = BCIPipeline(method=METHOD)
    action_queue = Queue()
    label_queue  = Queue()
    adapt_queue  = Queue()   # NEW: notify game.py when adapt() runs

    # Shared logs
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

            # Predict & send command
            cmd = pipeline.predict(processed)
            action_queue.put(cmd)

            # Adaptation only on correct trials
            if pipeline.adaptive and not label_queue.empty():
                label = label_queue.get()
                if cmd == label:
                    pipeline._win_buf.append(processed)
                    pipeline._lab_buf.append(label)
                if len(pipeline._win_buf) >= pipeline.adapt_N:
                    pipeline.adapt()
                    adapt_queue.put(True)    # signal adaptation event

            # keep latest window for game.py
            raw_eeg_log.clear()
            raw_eeg_log.append(processed)

    Thread(target=bci_loop, daemon=True).start()
    # pass adapt_queue into run_game
    run_game(action_queue, adapt_queue, game_states, label_queue, raw_eeg_log)

if __name__ == '__main__':
    main()
