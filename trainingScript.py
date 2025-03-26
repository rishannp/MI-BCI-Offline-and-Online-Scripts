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
num_trials_per_class = 10
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
# Proiling in CMD
#   python -m cProfile -s cumtime trainingScript.py
# ---------------------------------------------------------

#%% Train model from data

import pickle

# Specify the path to your pkl file
file_path = "training_data.pkl"

# Open and load the pkl file
with open(file_path, 'rb') as file:
    data = pickle.load(file)
    

