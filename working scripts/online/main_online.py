import numpy as np
from threading import Thread
from queue import Queue
from collections import deque

from config import WINDOW_SIZE, STEP_SIZE, BUFFER_SIZE, THRESHOLD, SAMPLING_RATE
from lsl_stream import LSLStreamHandler
from preprocess import Preprocessor
from control_logic import ControlLogic
from game import run_game
from featandclass import BCIPipeline  # imported pipeline


def bci_loop(action_queue: Queue, pipeline: BCIPipeline):
    """Run EEG→classification loop and send actions to `action_queue`."""
    stream = LSLStreamHandler()
    prep   = Preprocessor(fs=SAMPLING_RATE)
    ctrl   = ControlLogic(buffer_size=BUFFER_SIZE, threshold=THRESHOLD)

    buf = deque(maxlen=WINDOW_SIZE)

    while True:
        sample, ts = stream.pull_sample()
        if sample is None:
            continue

        buf.append(sample)

        if len(buf) == WINDOW_SIZE and len(buf) % STEP_SIZE == 0:
            window = np.array(buf)
            proc   = prep.process(window)
            if proc is None:
                continue

            pred   = pipeline.predict(proc)
            action = ctrl.update(pred)
            if action is not None:
                action_queue.put(action)

def main():
    # Choose pipeline and supply pre-trained model paths
    METHOD = 'plv'  # 'csp', 'plv', or 'riemann'
    CSP_PATH = r'C:\Users\uceerjp\Desktop\PhD\Year 2\online experiments\development\online\models\csp.pkl'
    LDA_PATH = r'C:\Users\uceerjp\Desktop\PhD\Year 2\online experiments\development\online\models\lda.pkl'
    GAT_PATH = r'C:\Users\uceerjp\Desktop\PhD\Year 2\online experiments\development\online\models\gat_model.pth'
    RIEMANN_TS_PATH = r'C:\Users\uceerjp\Desktop\PhD\Year 2\online experiments\development\online\models\riemann_ts.pkl'
    MDR_PATH = r'C:\Users\uceerjp\Desktop\PhD\Year 2\online experiments\development\online\models\mdr_model.pkl'

    pipeline = BCIPipeline(
        method=METHOD,
        fs=SAMPLING_RATE,
        csp_path=CSP_PATH,
        lda_path=LDA_PATH,
        gat_path=GAT_PATH,
        riemann_ts_path=RIEMANN_TS_PATH,
        mdr_path=MDR_PATH
    )

    action_queue = Queue()
    eeg_thread = Thread(target=bci_loop, args=(action_queue, pipeline), daemon=True)
    eeg_thread.start()

    run_game(action_queue)

if __name__ == '__main__':
    main()
