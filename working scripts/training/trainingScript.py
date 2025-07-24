import random
import numpy as np
import pickle
from time import time
import pylsl
from psychopy import visual, core, event
import psutil  # For system resource monitoring
import threading  # For running resource monitoring in parallel

# Resolve LSL streams
streams = pylsl.resolve_streams()
if not streams:
    raise RuntimeError("No LSL streams found. Ensure the EEG stream is active.")

# Print available streams
print("\nAvailable Streams:\n")
for i, stream in enumerate(streams):
    print(f"{i}: {stream.name()}")

# Select stream
val = int(input("Please select stream (enter number)>> "))
if val not in range(len(streams)):
    raise ValueError("Invalid stream selection.")

# Connect to chosen stream
instream = pylsl.StreamInlet(streams[val])
print(f"Connected to stream: {streams[val].name()}")

# Compute time offset using LSL's built-in time_correction
time_offset = instream.time_correction()
print(f"Time offset computed: {time_offset}")

# Stream information (for saving system configs)
stream_info = instream.info()
channel_names = []

# Parse the XML metadata for channel names
channel_xml = stream_info.desc().child("channels").child("channel")
for _ in range(stream_info.channel_count()):
    channel_names.append(channel_xml.child_value("label"))
    channel_xml = channel_xml.next_sibling()

# Add channel names to stream details
stream_details = {
    "name": stream_info.name(),
    "type": stream_info.type(),
    "channel_count": stream_info.channel_count(),
    "sampling_rate": stream_info.nominal_srate(),
    "stream_id": stream_info.source_id(),
    "channel_names": channel_names,
}

# PsychoPy setup
win = visual.Window(fullscr=True, color="black")
instructions = visual.TextStim(win, text="Focus on the cues. Press any key to start.", color="white")
fixation = visual.TextStim(win, text="+", color="white", height=0.2)
cue_left = visual.TextStim(win, text="←", color="white", height=0.5)
cue_right = visual.TextStim(win, text="→", color="white", height=0.5)
cue_rest = visual.TextStim(win, text="Rest", color="white", height=0.5)

# Display instructions
instructions.draw()
win.flip()
event.waitKeys()

# Training configuration
cue_duration = 5  # seconds
inter_trial_interval = 5  # seconds
num_trials_per_class = 3
sampling_rate = int(stream_details["sampling_rate"])  # e.g., 256 Hz
samples_per_cue = int(cue_duration * sampling_rate)

# Preallocate arrays
num_trials = num_trials_per_class * 3  # Total trials for Left, Right, and Rest
data = np.zeros((num_trials, samples_per_cue, stream_details["channel_count"]))
timestamps = np.zeros((num_trials, samples_per_cue))
event_markers = np.zeros((num_trials, samples_per_cue))

# List to store latency values per sample (only the duration of pull_sample calls)
latency_log = []

# Cue labels
cues = [(cue_left, "Left"), (cue_right, "Right"), (cue_rest, "Rest")]
labels = {"Left": 1, "Right": 2, "Rest": 0}

# Resource monitoring variables
resource_stats = []
monitoring = True

def monitor_resources():
    """Monitor CPU and memory usage in a separate thread."""
    while monitoring:
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        resource_stats.append({"time": time(), "cpu": cpu_usage, "memory": memory_info.percent})

# Start resource monitoring in a separate thread
monitor_thread = threading.Thread(target=monitor_resources)
monitor_thread.start()

# Trial loop
try:
    trial_idx = 0
    for _ in range(num_trials_per_class):
        random.shuffle(cues)  # Shuffle cue order for each repetition
        for selected_cue, cue_name in cues:
            print(f"Starting trial {trial_idx + 1}/{num_trials} - Cue: {cue_name}")

            # Check for ESC key press to terminate
            if event.getKeys(["escape"]):
                print("Escape key pressed. Terminating...")
                raise KeyboardInterrupt

            # Fixation cross
            fixation.draw()
            win.flip()
            core.wait(1)  # 1-second fixation

            # Present cue
            label = labels[cue_name]
            trial_samples = 0

            # Collect EEG data until we reach the target sample count or cue duration
            trial_start_time = time()
            while trial_samples < samples_per_cue and (time() - trial_start_time) < cue_duration:
                # Check for ESC key press during cue presentation
                if event.getKeys(["escape"]):
                    print("Escape key pressed. Terminating...")
                    raise KeyboardInterrupt

                selected_cue.draw()
                win.flip()

                # Measure the duration of the pull_sample call
                local_time_before = pylsl.local_clock()
                sample, timestamp = instream.pull_sample(timeout=1.0 / sampling_rate)
                local_time_after = pylsl.local_clock()
                sample_call_latency = local_time_after - local_time_before

                if sample:
                    # Apply time correction (if needed) to the timestamp
                    corrected_timestamp = timestamp + time_offset

                    # Record the sample and its call latency
                    latency_log.append(sample_call_latency)
                    data[trial_idx, trial_samples, :] = sample
                    timestamps[trial_idx, trial_samples] = corrected_timestamp
                    event_markers[trial_idx, trial_samples] = label
                    trial_samples += 1
                else:
                    print("Sample missed or timed out. Attempting again.")

            # Handle case where data is not collected as expected (e.g., missing data)
            while trial_samples < samples_per_cue:
                local_time_before = pylsl.local_clock()
                sample, timestamp = instream.pull_sample(timeout=0.01)  # Short timeout to avoid blocking
                local_time_after = pylsl.local_clock()
                sample_call_latency = local_time_after - local_time_before

                if sample:
                    corrected_timestamp = timestamp + time_offset
                    latency_log.append(sample_call_latency)
                    data[trial_idx, trial_samples, :] = sample
                    timestamps[trial_idx, trial_samples] = corrected_timestamp
                    event_markers[trial_idx, trial_samples] = label
                    trial_samples += 1
                else:
                    print("Waiting for more samples...")

            # Inter-trial interval
            win.flip()  # Blank screen
            core.wait(inter_trial_interval)

            trial_idx += 1

except KeyboardInterrupt:
    print("Experiment terminated by user.")

# Stop resource monitoring
monitoring = False
monitor_thread.join()

# Save data including latency log
output = {
    "data": data,
    "timestamps": timestamps,
    "event_markers": event_markers,
    "stream_details": stream_details,
    "parameters": {
        "cue_duration": cue_duration,
        "inter_trial_interval": inter_trial_interval,
        "num_trials": num_trials,
        "sampling_rate": sampling_rate,
        "num_trials_per_class": num_trials_per_class,
    },
    "resource_stats": resource_stats,
    "latency_log": latency_log,
}

with open("training_data.pkl", "wb") as f:
    pickle.dump(output, f)

print("Training completed. Data saved to training_data.pkl.")

# Close PsychoPy window
win.close()
core.quit()

# ---------------------------------------------------------
# Profiling in CMD
#   python -m cProfile -s cumtime trainingScript.py
# ---------------------------------------------------------

#%% Train model from data

import pickle

# Specify the path to your pkl file
file_path = "training_data.pkl"

# Open and load the pkl file
with open(file_path, 'rb') as file:
    data = pickle.load(file)
    
#%%O Output models: CSP LDA, PLVGAT

import numpy as np
import scipy.signal as sig
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import joblib


def train_csp_lda(eeg_data: np.ndarray,
                  event_markers: np.ndarray,
                  sfreq: float,
                  classes: tuple = (1, 2)) -> dict:
    """
    Train CSP + LDA on calibration EEG data, labeling each trial by the first marker sample.

    Args:
        eeg_data: array shape (n_trials, n_times, n_channels)
        event_markers: array shape (n_trials, n_times), values {0,1,2}
        sfreq: sampling frequency in Hz
        n_components: number of spatial patterns to retain per class
        classes: tuple of class labels to include (ignore rest=0)

    Returns:
        model_dict containing:
          - 'csp': trained CSP object
          - 'lda': trained LDA object
          - 'filters': spatial filters (patterns)
          - 'lda_coef': LDA weight vector
          - 'lda_intercept': LDA intercept
    """
    first_labels = event_markers[:, 0]
    mask = np.isin(first_labels, classes)
    X = eeg_data[mask]
    X = np.transpose(X, (0, 2, 1))  # to shape (n_epochs, n_channels, n_times)
    y = first_labels[mask]

    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
    csp.fit(X, y)
    X_csp = csp.transform(X)

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_csp, y)

    return {
        'csp': csp,
        'lda': lda,
        'filters': csp.filters_,
        'lda_coef': lda.coef_,
        'lda_intercept': lda.intercept_,
    }


def save_model(model: dict, filename: str) -> None:
    """
    Save CSP + LDA model components to disk.
    """
    joblib.dump(model, filename)


def load_model(filename: str) -> dict:
    """
    Load saved CSP + LDA model.
    """
    return joblib.load(filename)


def segment_signal(eeg_data: np.ndarray,
                   sfreq: float,
                   window_sec: float = 3.0,
                   hop_sec: float = 0.5) -> np.ndarray:
    """
    Segment continuous EEG into overlapping windows.
    """
    win_samples = int(window_sec * sfreq)
    hop_samples = int(hop_sec * sfreq)
    n_times, n_channels = eeg_data.shape
    segments = []
    for start in range(0, n_times - win_samples + 1, hop_samples):
        end = start + win_samples
        segments.append(eeg_data[start:end, :])
    return np.stack(segments, axis=0)


def compute_plv_matrix(eeg_segment: np.ndarray, n_elec: int = 62) -> np.ndarray:
    """
    Compute PLV matrix for a single EEG segment.
    """
    data = eeg_segment[:, :n_elec]
    n_times, n_channels = data.shape
    analytic = sig.hilbert(data, axis=0)
    phase = np.angle(analytic)
    plv = np.zeros((n_channels, n_channels))
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            diff = phase[:, j] - phase[:, i]
            plv_val = np.abs(np.sum(np.exp(1j * diff)) / n_times)
            plv[i, j] = plv_val
            plv[j, i] = plv_val
    return plv


def compute_plv_series(eeg_data: np.ndarray,
                       sfreq: float,
                       window_sec: float = 3.0,
                       hop_sec: float = 0.5,
                       n_electrodes: int = 62) -> np.ndarray:
    """
    Compute PLV matrices for all overlapping windows in continuous EEG.
    """
    segments = segment_signal(eeg_data, sfreq, window_sec, hop_sec)
    return np.stack([compute_plv_matrix(seg, n_elec=n_electrodes) for seg in segments], axis=0)

# Execute training and PLV computation using pre-loaded data dict
sfreq = 256.0  # update to your sampling frequency

# Train CSP+LDA model with 4 spatial filters
model = train_csp_lda(
    data['data'],               # EEG data array
    data['event_markers'],       # event markers array (correct key)
    sfreq=sfreq
)
filters = model['filters']
lda_coef = model['lda_coef']
lda_intercept = model['lda_intercept']
print(f"CSP filters shape: {filters.shape}  # should be (4, n_channels)")
print(f"LDA coefficients shape: {lda_coef.shape}")
print(f"LDA intercepts shape: {lda_intercept.shape}")

# Compute PLV series and labels for all windows across trials
all_plv = []
all_labels = []
first_labels = data['event_markers'][:, 0]
for trial_idx, trial in enumerate(data['data']):
    plv_series = compute_plv_series(trial, sfreq=sfreq)
    n_windows = plv_series.shape[0]
    all_plv.append(plv_series)
    label = first_labels[trial_idx]
    all_labels.extend([label] * n_windows)
    print(f"Trial {trial_idx}: {n_windows} windows of {plv_series.shape[1]}×{plv_series.shape[2]} PLV matrices")

# Stack into arrays: (#samples, n_elec, n_elec) and labels (#samples,)
plv_array = np.concatenate(all_plv, axis=0)
label_vector = np.array(all_labels)

print(f"Final PLV array shape: {plv_array.shape}")
print(f"Label vector shape: {label_vector.shape}")

# Overwrite training_data.pkl with augmented data dict
import pickle
try:
    with open('training_data.pkl', 'rb') as f:
        saved = pickle.load(f)
except FileNotFoundError:
    saved = {}
# Update saved dict
saved.update({
    'filters': filters,
    'lda_coef': lda_coef,
    'lda_intercept': lda_intercept,
    'plv_array': plv_array,
    'label_vector': label_vector,
})
# Write back to pickle
with open('training_data.pkl', 'wb') as f:
    pickle.dump(saved, f)
print("training_data.pkl updated with CSP, LDA and PLV entries.")


#%% Finetune GAT

import os
import pickle

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, GraphNorm, global_mean_pool
from torch_geometric.utils import add_self_loops
from torch_geometric.seed import seed_everything

# ---------------------------
# CONFIG
# ---------------------------
pretrained_path = r'C:\Users\uceerjp\Desktop\PhD\Year 2\online experiments\working scripts\training\best_gat_model.pt'
finetune_epochs = 15
batch_size      = 32
lr              = 1e-4
dropout         = 0.1
h1, h2, h3      = 32, 16, 8
heads           = 7
topk_percent    = 0.4
use_subset      = False

# Force CPU usage only
seed_everything(12345)
device = torch.device('cpu')  # use CPU only

# ---------------------------
# YOUR ELECTRODE INDICES
# ---------------------------
subset_indices = None  # e.g. [0,2,5,7] if using subset else None

# ---------------------------
# GAT MODEL DEFINITION
# ---------------------------
class SimpleGAT(nn.Module):
    def __init__(self, in_channels, h1, h2, h3, num_heads, dropout):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, h1, heads=num_heads,
                               concat=True, dropout=dropout)
        self.gn1   = GraphNorm(h1 * num_heads)
        self.conv2 = GATv2Conv(h1 * num_heads, h2, heads=num_heads,
                               concat=True, dropout=dropout)
        self.gn2   = GraphNorm(h2 * num_heads)
        self.conv3 = GATv2Conv(h2 * num_heads, h3, heads=num_heads,
                               concat=False, dropout=dropout)
        self.gn3   = GraphNorm(h3)
        self.lin   = nn.Linear(h3, 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.gn1(self.conv1(x, edge_index)))
        x = F.relu(self.gn2(self.conv2(x, edge_index)))
        x = F.relu(self.gn3(self.conv3(x, edge_index)))
        x = global_mean_pool(x, batch)
        return self.lin(x)

# ---------------------------
# GRAPH PREPROCESSING
# ---------------------------
def preprocess_graph(data, topk_percent=0.4):
    plv = data.x.clone().detach()
    if subset_indices is not None:
        plv = plv[subset_indices][:, subset_indices]
    num_nodes = plv.size(0)
    plv.fill_diagonal_(0.0)

    triu = torch.triu_indices(num_nodes, num_nodes, offset=1)
    weights = plv[triu[0], triu[1]]
    k = int(weights.numel() * topk_percent)
    topk = torch.topk(weights, k=k, sorted=False).indices

    row = triu[0][topk]
    col = triu[1][topk]
    edge_index = torch.cat([
        torch.stack([row, col], dim=0),
        torch.stack([col, row], dim=0)
    ], dim=1)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    data.edge_index = edge_index
    return data

# ---------------------------
# LOAD YOUR CALIBRATION DATA
# ---------------------------
with open('training_data.pkl', 'rb') as f:
    saved = pickle.load(f)
plv_array    = saved['plv_array']    # shape = (N_windows, n_elec, n_elec)
label_vector = saved['label_vector'] # shape = (N_windows,)

# ---------------------------
# BUILD PyG DATA LIST (filtering only classes 1 and 2)
new_graphs = []
for i in range(plv_array.shape[0]):
    raw_label = label_vector[i]
    # skip rest class (0)
    if raw_label not in (1, 2):
        continue
    plv = torch.tensor(plv_array[i], dtype=torch.float)
    # remap labels from {1,2} to {0,1}
    y = torch.tensor(raw_label - 1, dtype=torch.long)
    data = Data(x=plv, y=y)
    new_graphs.append(preprocess_graph(data, topk_percent))

# ---------------------------
# DATA LOADER
loader = DataLoader(new_graphs, batch_size=batch_size, shuffle=True)
# ---------------------------
# infer in_channels from one graph after preprocessing
in_feats = new_graphs[0].x.size(1)
model = SimpleGAT(in_feats, h1, h2, h3, heads, dropout).to(device)
# load weights onto CPU
state = torch.load(pretrained_path, map_location='cpu')
model.load_state_dict(state)
model.train()

opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
crit = nn.CrossEntropyLoss()

best_acc = 0.0
best_model_path = 'best_finetuned_model.pt'

for epoch in range(1, finetune_epochs+1):
    correct = total = 0
    for batch in loader:
        batch = batch.to(device)
        opt.zero_grad()
        logits = model(batch)
        loss = crit(logits, batch.y)
        loss.backward()
        opt.step()
        preds = logits.argmax(dim=1)
        correct += (preds == batch.y).sum().item()
        total += batch.num_graphs
    epoch_acc = correct / total
    # save best model
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        torch.save(model.state_dict(), best_model_path)
    print(f"Epoch {epoch}/{finetune_epochs}  Acc: {epoch_acc:.2%}  Best: {best_acc:.2%}")

print(f"Best fine-tuned model saved to {best_model_path} with Acc: {best_acc:.2%}")
