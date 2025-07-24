# main_online.py

import numpy as np
from threading import Thread
from queue import Queue
from collections import deque

from config import WINDOW_SIZE, STEP_SIZE, BUFFER_SIZE, THRESHOLD
from lsl_stream import LSLStreamHandler
from preprocess import Preprocessor
from control_logic import ControlLogic
from game import run_game
from featandclass import BCIPipeline

# store raw EEG if you want
raw_eeg = []

def bci_loop(action_queue, pipeline):
    stream = LSLStreamHandler()
    prep   = Preprocessor(fs=stream.sampling_rate)
    ctrl   = ControlLogic(buffer_size=BUFFER_SIZE, threshold=THRESHOLD)

    buf = deque(maxlen=WINDOW_SIZE)
    while True:
        sample, ts = stream.pull_sample()
        if sample is None: continue
        raw_eeg.append((ts, sample))
        buf.append(sample)

        if len(buf)==WINDOW_SIZE and len(buf)%STEP_SIZE==0:
            window = np.array(buf)
            proc   = prep.process(window)
            if proc is None: continue
            pred   = pipeline.predict(proc)
            act    = ctrl.update(pred)
            if act is not None:
                # for CSP+LDA, store pseudo-label
                if pipeline.method=='csp':
                    # heuristic: if obstacle > center, label=0 else 1
                    # you'd need shared game x-position here
                    label = 0
                    pipeline._win_buf.append(proc)
                    pipeline._lab_buf.append(label)
                action_queue.put(act)

def main():
    # choose your method:
    METHOD = 'plv'  # 'csp' or 'plv'
    pipeline = BCIPipeline(method=METHOD)

    action_queue = Queue()
    t = Thread(target=bci_loop, args=(action_queue, pipeline), daemon=True)
    t.start()

    run_game(action_queue, pipeline)

if __name__=='__main__':
    main()
