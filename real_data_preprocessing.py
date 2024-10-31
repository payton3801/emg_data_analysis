# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.io import loadmat
import math

# %%
#plotting the real emg data, looking at keys and info
emgdata = loadmat('J10_s10_i0_pref.mat')
rawdata = emgdata['emg_full_raw'] 
filterdata = emgdata['emg_full_fil']
# for key in emgdata:
#     if not key.startswith('__'):
#         print(f"\nKey: {key}")
#         print(f"Type: {type(emgdata[key])}")
#         print(f"Shape: {emgdata[key].shape}")

plt.rcParams['agg.path.chunksize'] = 10000

# %%
#emg data pramaters

#calculate sampling rate
time_vector = emgdata['t_emg']
intervals = np.diff(time_vector)
SAMPLE_RATE = 1/np.mean(intervals) 

#general parameters
NFFT = 50000 #Use sampling rate/NFFT = 0.1
NPERSEG = 50000
NOVERLAP = 10000 #Use .2 (NFFT) = NOVERLAP
CHAN_IX = 0
num_channels = rawdata.shape[1]

#notch filter parameters
bandwidth = 2
notch_frequencies= [60, 120, 240, 300, 420]

channel_names = emgdata['emg_names'].flatten()
channel_names = emgdata['emg_names'][0]
channel_names = [str(name[0]) for name in channel_names]

# %%
#plotting the full raw data
plt.figure(figsize=(10,4), dpi = 200)
plt.plot(rawdata, alpha=0.4, color='k', label='ENG raw data')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Raw Data")
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.xlim([0,120])

# %%
#processing pipeline for one channel
#pre-processing step 1: notch filters at 60 Hz, 120 Hz, 240 Hz, 300 Hz, 420 Hz
def apply_notch_filter(x, notch_frequencies, bandwidth, sample_rate):
    """ 
        apply notch filter(s) to a designated channel
        notch_frequencies = desired frequencies to be filtered out (Hz)
        x = signal data being filtered
        bandwidth = bandwidth of notch filter (Hz)
        sample_rate = sampling rate of the signal (Hz)
    """

    Q = [freq / bandwidth for freq in notch_frequencies]  # Compute Q for each frequency
    for freq, q in zip(notch_frequencies, Q):
        b, a = signal.iirnotch(freq, q, SAMPLE_RATE)
        x = signal.filtfilt(b, a, x)

    return x

def apply_butterworth_filter(x):
    """
        apply 4th order butterworth high pass filter with cutoff at 65 Hz
        x = signal data being filtered
    """

    b, a = signal.butter(4, 65.0, btype='high', analog=False, fs=SAMPLE_RATE)
    butterworth = signal.filtfilt(b, a, x)
    return butterworth

# Rectify the EMG signal
applied_notch = apply_notch_filter(rawdata[:, CHAN_IX], notch_frequencies, bandwidth, SAMPLE_RATE)
applied_butter = apply_butterworth_filter(applied_notch)
rectifieddata = np.abs(applied_butter)

notchdata = np.zeros_like(rawdata)
for i in range(rawdata.shape[1]):
    notchdata[:,i] = apply_notch_filter(rawdata[:,i], notch_frequencies, bandwidth, SAMPLE_RATE)

# %%
# Time vector for plotting
duration = len(rectifieddata) / SAMPLE_RATE
t = np.linspace(0, duration, len(rectifieddata), endpoint=False)

# Plot rectified EMG data
plt.figure(figsize=(10, 4), dpi=200)
plt.plot(t, rectifieddata, '-o', alpha=0.4, color='k', markersize=2)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.ylabel(f"{channel_names[CHAN_IX]}")
plt.xlabel("Time (s)")
plt.title(f"Time Domain Rectified EMG for Channel {channel_names[CHAN_IX]}")
plt.xlim([0,30])

f, pxx = signal.welch(rectifieddata, nperseg=NPERSEG, nfft=NFFT, noverlap=NOVERLAP, fs=SAMPLE_RATE)
plt.figure(figsize=(10,4), dpi = 200)
plt.loglog(f, pxx, '-o', alpha=0.4, color='k', markersize=2)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.ylabel("Power")
plt.xlabel("Frequency (Hz)")
plt.title(f"Frequency Domain Rectified EMG for Channel {channel_names[CHAN_IX]}")
plt.tight_layout()
plt.show()

# %%
#overlaying all spectras on top of each other
plt.figure(figsize=(10,4), dpi = 200)
plt.title(f"Overlaid plots for muscle {channel_names[CHAN_IX]}")
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.ylabel("Power")
plt.xlabel("Frequency (Hz)")

f, pxx = signal.welch(rawdata[:, CHAN_IX], nperseg=NPERSEG, nfft=NFFT, noverlap=NOVERLAP, fs=SAMPLE_RATE)
plt.loglog(f, pxx, '-o', alpha=0.4, color='r', markersize=2, label = "Raw data")

f, pxx = signal.welch(applied_notch, nperseg=NPERSEG, nfft=NFFT, noverlap=NOVERLAP, fs=SAMPLE_RATE)
plt.loglog(f, pxx, '-o', alpha=0.4, color='b', markersize=2, label = "Notch-filtered data")

f, pxx = signal.welch(applied_butter, nperseg=NPERSEG, nfft=NFFT, noverlap=NOVERLAP, fs=SAMPLE_RATE)
plt.loglog(f, pxx, '-o', alpha=0.4, color='k', markersize=2, label = "Butterworth-filtered data")

f, pxx = signal.welch(rectifieddata, nperseg=NPERSEG, nfft=NFFT, noverlap=NOVERLAP, fs=SAMPLE_RATE)
plt.loglog(f, pxx, '-o', alpha=0.4, color='g', markersize=2, label = "Rectified data")

plt.legend()

# %%
#rectified emg for all signals -- able to run independently after loading rawdata
#proceessing pipeline for grid of all channels

fig, axes = plt.subplots(3, 4, figsize=(20, 15), dpi=200)
axes=axes.flatten()

for CHAN_IX in range(num_channels):
    #why do i have to redefine these variables here
    x = rawdata[:, CHAN_IX]
    applied_notch = apply_notch_filter(x, notch_frequencies, bandwidth, SAMPLE_RATE)
    applied_butter = apply_butterworth_filter(applied_notch)
    rectifieddata = np.abs(applied_butter)

    ax = axes[CHAN_IX]
    f_raw, pxx_raw = signal.welch(rawdata[:, CHAN_IX], nperseg=NPERSEG, nfft=NFFT, noverlap=NOVERLAP, fs=SAMPLE_RATE)
    f_notch, pxx_notch = signal.welch(applied_notch, nperseg=NPERSEG, nfft=NFFT, noverlap=NOVERLAP, fs=SAMPLE_RATE)
    f_butter, pxx_butter = signal.welch(applied_butter, nperseg=NPERSEG, nfft=NFFT, noverlap=NOVERLAP, fs=SAMPLE_RATE)
    f_rectified, pxx_rectified = signal.welch(rectifieddata, nperseg=NPERSEG, nfft=NFFT, noverlap=NOVERLAP, fs=SAMPLE_RATE)

    ax.loglog(f_raw, pxx_raw, '-o', alpha=0.4, color='r', markersize=2, label='Raw data')
    ax.loglog(f_notch, pxx_notch, '-o', alpha=0.4, color='b', markersize=2, label='Notch-filtered data')
    ax.loglog(f_butter, pxx_butter, '-o', alpha=0.4, color='k', markersize=2, label='Butterworth-filtered data')
    ax.loglog(f_rectified, pxx_rectified, '-o', alpha=0.4, color='g', markersize=2, label='Rectified data')

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_title(f"Spectra for {channel_names[CHAN_IX]}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].legend()
plt.tight_layout()
plt.show()

# %%
#rectifying emg signal across a single channel
rectifieddata = np.abs(applied_butter)

#timeplot
duration = len(rectifieddata) / SAMPLE_RATE
t = np.linspace(0, duration, len(rectifieddata), endpoint=False)

plt.figure(figsize=(10,4), dpi=200)
plt.plot(t, rectifieddata)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.ylabel("Amplitude")
plt.xlabel("Time(s)")
plt.title(f"Rectified EMG for Muscle {channel_names[CHAN_IX]}")
plt.xlim([0,30])

#frequency plot
f, pxx = signal.welch(rectifieddata, nperseg=NPERSEG, nfft=NFFT, noverlap=NOVERLAP, fs=SAMPLE_RATE)
plt.figure(figsize=(10, 4), dpi=200)
plt.loglog(f, pxx, '-o', alpha=0.4, color='k', markersize=2)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.ylabel("Power")
plt.xlabel("Frequency(Hz)")
plt.title(f"Rectified EMG for Muscle {channel_names[CHAN_IX]}")
plt.xlim([0,30])

# %%
#resampling EMG to 500 Hz
SAMPLE_RATE = round(SAMPLE_RATE)
TARGET_SAMPLE_RATE = 500
DOWNSAMPLING = SAMPLE_RATE // TARGET_SAMPLE_RATE
#resampled_data1 = signal.decimate(rectifieddata, q = DOWNSAMPLING, n=None, ftype='iir', axis=-1, zero_phase=True)
    #check: axis = 1 and zero_phase = True?
resampled_data = signal.resample_poly(rectifieddata, down=DOWNSAMPLING, up=1)
duration = len(resampled_data) / TARGET_SAMPLE_RATE
t2 = np.linspace(0, duration, len(resampled_data), endpoint=False)

# Plotting the resampled data over time
plt.figure(figsize=(10,4), dpi=200)
plt.plot(t, rectifieddata, 'r--', color='k', alpha=0.7, marker = 'o', markersize=3, label='Original Data')
#plt.plot(t2, resampled_data1, 'r--', color = 'b', alpha=0.7, marker='o', markersize=3, label='Decimate')
plt.plot(t2, resampled_data, 'r--', color = 'g', alpha=0.7, marker='o', markersize=3, label='Resample-poly')

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.ylabel("Amplitude")
plt.xlabel("Time (s)")
plt.title(f"Resampled Rectified EMG for Muscle {channel_names[CHAN_IX]}")
plt.tight_layout()
plt.xlim([0, 5])
plt.ylim([0, .33]) #for zooming in/ filling the whitespace

#plt.xlim([0.3, 0.4])
#plt.ylim([0, .33]) #for looking super close

plt.legend()
plt.show()

# %%
#quartile clipping the data by the 99.99th percentile
quantiled_data = np.quantile(resampled_data, .9999)
quantiled_data1 = np.quantile(resampled_data, .99)
quantiled_data2 = np.quantile(resampled_data, .95)
quantiled_data3 = np.quantile(resampled_data, .90)
clipped_data = np.clip(resampled_data, a_max = quantiled_data, a_min=None)
clipped_data1 = np.clip(resampled_data, a_max = quantiled_data1, a_min=None)
clipped_data2 = np.clip(resampled_data, a_max = quantiled_data2, a_min=None)
clipped_data3 = np.clip(resampled_data, a_max = quantiled_data3, a_min=None)
duration = len(clipped_data) / TARGET_SAMPLE_RATE
t3 = np.linspace(0, duration, len(clipped_data), endpoint=False)


#plotting
plt.figure(figsize=(10,4), dpi=200)
plt.plot(t2, resampled_data, 'r--', color='r', alpha=0.7, marker = 'o', markersize=3, label='Original Data')
plt.plot(t3, clipped_data, 'r--', color='g', alpha=0.7, marker='o', markersize=3, label='99.99')
plt.plot(t3, clipped_data1, 'r--', color = 'b', alpha=0.7, marker='o', markersize=3, label='99')
plt.plot(t3, clipped_data2, 'r--', color = 'm', alpha=0.7, marker='o', markersize=3, label='95')
plt.plot(t3, clipped_data3, 'r--', color = 'k', alpha=0.7, marker='o', markersize=3, label='90')
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.ylabel("Amplitude")
plt.xlabel("Time (s)")
plt.title(f"Quantile Clipping for Muscle {channel_names[CHAN_IX]}")
plt.tight_layout()
plt.legend()

 # %%
