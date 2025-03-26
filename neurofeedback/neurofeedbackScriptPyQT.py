### NOT WORKING... Need a way to update plots
import pylsl
import numpy as np
import time
import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
from scipy.signal import welch

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
    'Theta': (4, 8),    # Theta: 4-8 Hz
    'Alpha': (8, 13),   # Alpha: 8-13 Hz
    'Beta': (13, 30),   # Beta: 13-30 Hz
}

# Reduced buffer size for 0.2s window for faster updates
buffer_size = 256
buffer = []  # Initialize the buffer for the stream

# Simulate electrode names for 22 channels based on the 10-20 system
ch_names = [
    "Fp1", "Fp2", "F3", "F4", "F7", "F8", 
    "C3", "C4", "T7", "T8", "P3", "P4", 
    "P7", "P8", "O1", "O2", "Fz", "Cz", 
    "Pz", "Oz", "PO3", "PO4"
]

# Create an MNE info object for channel layout
info = {
    "ch_names": ch_names,
    "sfreq": sampling_frequency,
    "ch_types": "eeg"
}

# Create a PyQt application
app = QApplication([])  # This now works since QApplication is imported from QtWidgets

# Create a plot window using GraphicsLayoutWidget
win = pg.GraphicsLayoutWidget()  # Use GraphicsLayoutWidget instead of GraphicsWindow
win.setWindowTitle("Real-Time EEG Visualization")  # Set window title

# Create a plot for EEG data
plot = win.addPlot(title="EEG Channel Data")
plot.setYRange(-100, 100)

# Initialize the curve to plot data for one channel
curve = plot.plot(pen='y')

# Update function for live plotting
def update_plot():
    global buffer

    if len(buffer) >= buffer_size:
        # Convert the buffer into a numpy array (channel x samples)
        raw_data = np.array(buffer).T  # Transpose to channel x samples

        # Calculate PSD for each channel using Welch's method
        psds = np.zeros((channel_count, 65))  # Store PSDs for each channel
        f, Pxx = [], []

        for i in range(channel_count):
            # Compute the PSD for each channel using Welch's method
            f, Pxx = welch(raw_data[i], fs=sampling_frequency, nperseg=128)
            psds[i, :] = Pxx

        # For now, display PSD of the first channel (you can extend this to other channels)
        band_psd = np.mean(psds[:, f >= 8], axis=1)  # Example: Display Alpha band (8-13 Hz)
        curve.setData(band_psd)  # Update the plot

        buffer = []  # Clear buffer for next batch

# Create a timer for real-time updates
timer = QTimer()  # QTimer is now imported from QtCore
timer.timeout.connect(update_plot)
timer.start(50)  # Update every 50 ms (20 Hz)

# Streaming data and handling real-time EEG input
print("\nStreaming data... Press Ctrl+C to stop.\n")
try:
    while True:
        # Record start time for sample collection
        start_time = time.time()

        # Pull a sample from the stream
        sample, timestamp = instream.pull_sample(timeout=1)
        if sample:
            buffer.append(sample)

except KeyboardInterrupt:
    print("\nStreaming stopped by user.")

# Start the Qt event loop for the PyQt application
win.show()  # Show the window before running the event loop
app.exec()
