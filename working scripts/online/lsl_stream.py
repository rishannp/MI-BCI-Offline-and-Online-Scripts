import pylsl

class LSLStreamHandler:
    def __init__(self, stream_index=None, sampling_rate=None):
        """
        Resolve available LSL streams and connect to the one the user selects.
        If sampling_rate isn’t given, grab the stream’s nominal rate.
        """
        streams = pylsl.resolve_streams()
        if not streams:
            raise RuntimeError("No LSL streams found. Ensure your EEG device is online.")
        if stream_index is None:
            for i, s in enumerate(streams):
                print(f"{i}: {s.name()} ({s.type()})")
            stream_index = int(input("Select LSL stream index: "))
        self.inlet = pylsl.StreamInlet(streams[stream_index])
        # record time offset for accurate timestamps
        self.time_offset = self.inlet.time_correction()
        # if user passed sampling_rate, use it; otherwise query the stream
        if sampling_rate is None:
            info = self.inlet.info()
            self.sampling_rate = info.nominal_srate()
            print(f"Using stream nominal sampling rate: {self.sampling_rate} Hz")
        else:
            self.sampling_rate = sampling_rate

    def pull_sample(self, timeout=None):
        """
        Pull one sample (list of floats) and correct its timestamp.
        Default timeout = 1 / sampling_rate.
        """
        if timeout is None:
            timeout = 1.0 / self.sampling_rate
        sample, ts = self.inlet.pull_sample(timeout=timeout)
        if sample:
            return sample, ts + self.time_offset
        else:
            return None, None
