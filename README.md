# Documentation: EEG Data Collection with Latency Measurement using LSL and PsychoPy
This script is designed for EEG data collection in an experimental training session. It integrates EEG acquisition via Lab Streaming Layer (LSL), visual cue presentation with PsychoPy, and real‐time measurement of the data acquisition latency. Additionally, the script monitors system resources during the experiment and saves all relevant data for post-session analysis.

---

## Overview

- **Purpose:**  
  The script collects EEG data while presenting visual cues (Left, Right, Rest) to the participant. For each EEG sample acquired from the LSL stream, the script measures the latency of the data acquisition call (i.e. the duration of the `pull_sample` call) using high-resolution local clock readings.

- **Key Features:**  
  - Resolves available LSL streams and allows user selection.
  - Computes a time offset using LSL’s built-in `time_correction` function.
  - Presents visual stimuli and instructions via PsychoPy.
  - Acquires a fixed number of EEG samples per cue while measuring and logging the call latency.
  - Monitors CPU and memory usage in a separate thread.
  - Saves the EEG data, timestamps, event markers, resource statistics, and latency measurements to a pickle file.

---

## Dependencies

Make sure the following Python libraries are installed:
- **pylsl:** For resolving and connecting to EEG streams.
- **psychopy:** For displaying visual stimuli and managing experimental flow.
- **numpy:** For numerical data storage and processing.
- **pickle:** For saving collected data.
- **psutil:** For monitoring CPU and memory usage.
- **threading:** For running resource monitoring concurrently.
- **time:** For time measurements.

Install the required packages using pip if needed:
```bash
pip install pylsl psychopy numpy psutil
```

---

## Detailed Functionality

### 1. LSL Stream Setup
- **Stream Resolution and Selection:**  
  The script first resolves available LSL streams. It then prints a list of stream names and prompts the user to select one by entering its corresponding number.
  
- **Connection and Time Offset:**  
  Once a stream is selected, an inlet is created to receive data. The script computes a time offset using `instream.time_correction()`, which estimates the difference between the EEG device’s clock and the local computer’s clock. This offset is used to correct the timestamps of the acquired samples.

- **Stream Metadata:**  
  Metadata (such as stream name, type, channel count, sampling rate, and channel labels) is parsed from the stream’s XML description and stored for later reference.

### 2. PsychoPy Experimental Setup
- **Window and Stimuli Initialization:**  
  A full-screen PsychoPy window with a black background is created. Several text stimuli are defined:
  - Instructions for the participant.
  - A fixation cross ("+").
  - Visual cues for "Left", "Right", and "Rest" conditions.

- **Starting the Experiment:**  
  The instructions are displayed, and the script waits for a key press to begin the experiment.

### 3. Experiment Configuration
- **Timing Parameters:**  
  - `cue_duration`: Duration (in seconds) for which each cue is presented.
  - `inter_trial_interval`: Pause (in seconds) between trials.
  - `num_trials_per_class`: Number of trials for each cue condition.
  - `samples_per_cue`: Number of EEG samples to acquire per cue, computed as cue duration multiplied by the sampling rate.

- **Data Preallocation:**  
  Arrays are preallocated for EEG data, timestamps, and event markers. A list (`latency_log`) is also initialized to record the duration of each `pull_sample` call.

### 4. Data Acquisition and Latency Measurement
- **Trial Loop:**  
  The experiment proceeds through trials. In each trial:
  - The order of cues is shuffled.
  - A 1-second fixation cross is shown before cue presentation.
  - For each cue, data is collected until the number of required samples is reached or the cue duration expires.
  
- **Latency Measurement:**  
  For every sample, the script records the local time immediately before and after calling `instream.pull_sample()`. The duration of the sample acquisition call (i.e. `local_time_after - local_time_before`) is computed and stored in `latency_log`. This value reflects the latency of the data acquisition function independent of the experiment’s waiting periods.

- **Handling Missed Samples:**  
  If a sample is missed or times out, the script prints a warning and continues attempting to collect the required number of samples.

### 5. Resource Monitoring
- **Background Monitoring Thread:**  
  A separate thread continuously monitors CPU and memory usage using `psutil`. The resource statistics are appended to the `resource_stats` list at 1-second intervals.

### 6. Data Saving and Cleanup
- **Data Aggregation:**  
  At the end of the experiment (or if terminated early by pressing the ESC key), all collected data—including the EEG data, corrected timestamps, event markers, resource statistics, and latency log—is aggregated into a dictionary.

- **Saving to File:**  
  The aggregated data is saved into a pickle file (`training_data.pkl`) for further analysis.

- **Cleanup:**  
  The resource monitoring thread is stopped, and the PsychoPy window is closed before the script exits.

---

## Profiling Instructions

To profile the script and view performance statistics (such as the cumulative time spent in various function calls), run the following command in Command Prompt (CMD):

```bash
python -m cProfile -s cumtime trainingScript.py
```

This command sorts the profiling output by cumulative time, which helps in identifying the most time-consuming parts of the script.

---

## Summary

This script efficiently integrates EEG data collection, real-time latency measurement of the data acquisition calls, visual cue presentation, and system resource monitoring. By measuring only the duration of the `pull_sample` calls, the script isolates the acquisition latency from the intentional wait times built into the experimental design. All experimental data is saved for further analysis, and the script can be profiled to optimize performance if necessary.

Feel free to modify the configuration parameters (e.g., cue duration, inter-trial interval, number of trials) to suit your experimental needs.
