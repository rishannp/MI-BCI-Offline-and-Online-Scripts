## SETUP ELEECTRODE PLOTS

import pylsl
import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.signal import welch
import time
import threading
from scipy.io import loadmat

# Define the electrode positions
electrode_positions = [
    {"name": "Fp1", "x": -0.4, "y": 1.0},
    {"name": "Fp2", "x": 0.4, "y": 1.0},
    {"name": "F7", "x": -0.7, "y": 0.5},
    {"name": "F3", "x": -0.4, "y": 0.4},
    {"name": "Fz", "x": 0.0, "y": 0.4},
    {"name": "F4", "x": 0.4, "y": 0.4},
    {"name": "F8", "x": 0.7, "y": 0.5},
    {"name": "C3", "x": -0.4, "y": 0.0},
    {"name": "Cz", "x": 0.0, "y": 0.0},
    {"name": "C4", "x": 0.4, "y": 0.0},
    {"name": "T3", "x": -0.8, "y": 0.0},
    {"name": "T4", "x": 0.7, "y": 0.0},
    {"name": "T5", "x": -0.7, "y": -0.5},
    {"name": "T6", "x": 0.7, "y": -0.5},
    {"name": "P3", "x": -0.4, "y": -0.4},
    {"name": "Pz", "x": 0.0, "y": -0.4},
    {"name": "P4", "x": 0.4, "y": -0.4},
    {"name": "O1", "x": -0.3, "y": -1.0},
    {"name": "O2", "x": 0.3, "y": -1.0},
]

def get_channel_locations(ch_names, electrode_positions):
    """
    Retrieve XY coordinates for the specified channel names.
    
    Parameters:
    - ch_names: List of desired channel names (strings).
    - electrode_positions: List of dictionaries with 'name', 'x', and 'y' fields.
    
    Returns:
    - Dictionary with channel names as keys and their (x, y) coordinates as values.
    """
    # Map electrode names to their coordinates
    electrode_dict = {electrode["name"]: (electrode["x"], electrode["y"]) for electrode in electrode_positions}
    
    # Filter for the specified channels
    channel_locations = {}
    for ch in ch_names:
        channel_locations[ch] = electrode_dict.get(ch, None)  # Handle missing channels gracefully
    return channel_locations

# Example usage
ch_names = [
    "F7", "F3", "Fz", "F4", "F8", 
    "T3", "C3", "Cz", "C4", "T4", "T5", 
    "P3", "Pz", "P4", "T6"
]  # desired channel names

channel_locations = get_channel_locations(ch_names, electrode_positions)

# Extract data
channel_names = list(channel_locations.keys())
x_coords = np.array([coord[0] for coord in channel_locations.values()])
y_coords = np.array([coord[1] for coord in channel_locations.values()])

# Plot
plt.figure(figsize=(8, 8))
plt.scatter(x_coords, y_coords, color="blue", label="Electrode Positions")
for i, label in enumerate(channel_names):
    plt.text(x_coords[i], y_coords[i], label, fontsize=20, ha="center", va="bottom")

# Add a circular boundary
circle = plt.Circle((0, 0), 1, color="black", fill=False, linestyle="--", linewidth=0.8)
plt.gca().add_artist(circle)
plt.gca().set_aspect("equal", adjustable="box")

# Set plot limits and labels
plt.xlim(-1.2, 1.2)
plt.ylim(-1.2, 1.2)
plt.title("Topographic Plot")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()

#%%

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

# Get stream information
stream_info = instream.info()
channel_count = stream_info.channel_count()
sampling_frequency = int(stream_info.nominal_srate())

print(f"Stream has {channel_count} channels.")
print(f"Sampling Frequency: {sampling_frequency} Hz")

# Define frequency bands in Hz
bands = {
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30)
}

# Reduced buffer size for 0.2s window for faster updates
buffer_size = 128  # Reduced buffer size
buffer = []  # Initialize the buffer for the stream

# Simulate electrode names for 22 channels based on the 10-20 system
ch_names = [
    "Fp1", "Fp2", "F3", "F4", "F7", "F8", 
    "C3", "C4", "T7", "T8", "P3", "P4", 
    "P7", "P8", "O1", "O2", "Fz", "Cz", 
    "Pz", "Oz", "PO3", "PO4"
]


# Use the standard 10-20 montage with 22 electrodes
#montage = mne.channels.make_standard_montage('standard_1020')

# Create an MNE info object for channel layout
#info = mne.create_info(ch_names, sampling_frequency, ch_types="eeg")
#info.set_montage(montage)

# Precompute the frequency band indices
band_indices = {band_name: np.array([], dtype=int) for band_name in bands}
for band_name, (low_freq, high_freq) in bands.items():
    band_indices[band_name] = np.where((np.arange(33) >= low_freq) & (np.arange(33) <= high_freq))[0]

# Create a RawArray object for the incoming data
raw_data = np.zeros((channel_count, buffer_size))  # For storing EEG data

# Control the frequency of plotting (only every 'plot_interval' iterations)
plot_interval = 10
plot_counter = 0  # Initialize plot counter

# Function to collect data in a separate thread
def collect_data(instream, buffer):
    while True:
        sample, timestamp = instream.pull_sample(timeout=1)
        if sample:
            buffer.append(sample)

# Start a separate thread to collect data
data_thread = threading.Thread(target=collect_data, args=(instream, buffer))
data_thread.daemon = True  # Allow thread to exit when the main program exits
data_thread.start()

print("\nStreaming data... Press Ctrl+C to stop.\n")

try:
    while True:
        # Record start time for sample collection
        start_time = time.time()

        # Process the buffer once it reaches the required size
        if len(buffer) >= buffer_size:
            # Convert the buffer into a numpy array (channel x samples)
            raw_data = np.array(buffer[:buffer_size]).T  # Transpose to channel x samples

            # Calculate PSD for each channel using Welch's method
            psds = np.zeros((channel_count, 33))  # Store PSDs for each channel

            for i in range(channel_count):
                # Compute the PSD for each channel using a smaller segment length
                f, Pxx = welch(raw_data[i], fs=sampling_frequency, nperseg=64)  # Reduce nperseg for faster calculation
                psds[i,:] = Pxx

            # Plot the topoplots for each frequency band
            if plot_counter % plot_interval == 0:
                fig, axes = plt.subplots(1, 3, figsize=(15, 6))

                # Ensure we don't exceed the number of subplots
                for idx, (band_name, _) in enumerate(bands.items()):
                    if idx >= len(axes):
                        break  # Exit the loop if there are more bands than subplots
                    
                    # Extract the PSD for the frequency band
                    band_psd = np.mean(psds[:, band_indices[band_name]], axis=1)

                    # Create topoplot for the given band
                    #mne.viz.plot_topomap(band_psd, info, axes=axes[idx], show=False)
                    axes[idx].set_title(f"{band_name} band")

                plt.tight_layout()
                plt.show()

            # Record end time for processing and calculate delay
            end_time = time.time()
            processing_delay = end_time - start_time
            print(f"Processing delay: {processing_delay:.4f} seconds")

            # Clear the buffer for the next batch
            buffer[:] = buffer[buffer_size:]

            plot_counter += 1  # Increment plot counter

except KeyboardInterrupt:
    print("\nStreaming stopped by user.")
