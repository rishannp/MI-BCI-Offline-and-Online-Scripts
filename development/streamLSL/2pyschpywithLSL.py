import random
from time import time, sleep
import pylsl
from psychopy import visual, core, event

# Resolve LSL streams
streams = pylsl.resolve_streams()
if not streams:
    print("No LSL streams found.")
    exit()

# Print available streams
print("\nAvailable Streams:\n")
for i, stream in enumerate(streams):
    print(f"{i}: {stream.name()}")

# Select stream
val = int(input("Please select stream (enter number)>> "))
if val not in range(len(streams)):
    print("Invalid stream selection.")
    exit()

# Connect to chosen stream
instream = pylsl.StreamInlet(streams[val])
print(f"Connected to stream: {streams[val].name()}")

# Initialize PsychoPy
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
cue_duration = 3  # seconds
inter_trial_interval = 2  # seconds
num_trials = 10

# Trial loop
cues = [cue_left, cue_right, cue_rest]
labels = ["Left", "Right", "Rest"]
data_log = []

for trial in range(num_trials):
    # Randomly select a cue
    cue_idx = random.randint(0, len(cues) - 1)
    selected_cue = cues[cue_idx]
    label = labels[cue_idx]

    # Fixation cross
    fixation.draw()
    win.flip()
    core.wait(1)  # 1-second fixation

    # Present cue
    selected_cue.draw()
    win.flip()

    start_time = time()
    trial_data = []
    while time() - start_time < cue_duration:
        chunk, timestamp = instream.pull_sample(timeout=0.01)
        if chunk:
            trial_data.append((timestamp, chunk))
    data_log.append({"label": label, "data": trial_data})

    # Inter-trial interval
    win.flip()  # Blank screen
    core.wait(inter_trial_interval)

# Save collected data
import json
with open("lsl_training_data.json", "w") as f:
    json.dump(data_log, f)

print("Training completed. Data saved to lsl_training_data.json.")

# Close PsychoPy window
win.close()
core.quit()
