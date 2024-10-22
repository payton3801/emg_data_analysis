# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.io import loadmat

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

#parameters
NFFT = 50000 #Use sampling rate/NFFT = 0.1
NPERSEG = 50000
NOVERLAP = 10000 #Use .2 (NFFT) = NOVERLAP
CHAN_IX = 1 #12 channels

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
f, pxx = signal.welch(rawdata[:,CHAN_IX], nperseg=NPERSEG, nfft=NFFT, noverlap=NOVERLAP, fs=SAMPLE_RATE)
plt.figure(figsize=(10,4), dpi=200)
plt.loglog(f, pxx, '-o', alpha=0.4, color='k', markersize=2, label='ENG raw data')
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.ylabel("Power")
plt.xlabel("Frequency (Hz)")
plt.title(f"Frequency Domain of Channel {[CHAN_IX]}")
plt.tight_layout()
#plt.xlim([0,500])

bandwidth = 2
notch_frequencies= [60, 120, 240, 300, 420]
Q = [freq / bandwidth for freq in notch_frequencies]  # Compute Q for each frequency


notchdata = rawdata[:, CHAN_IX]
for freq, q in zip(notch_frequencies, Q):
    b, a = signal.iirnotch(freq, q, SAMPLE_RATE)
    notchdata = signal.filtfilt(b, a, notchdata)

f, pxx = signal.welch(notchdata, nperseg=NPERSEG, nfft=NFFT, noverlap=NOVERLAP, fs=SAMPLE_RATE)
plt.figure(figsize=(10,4), dpi = 200)
plt.loglog(f, pxx, '-o', alpha=0.4, color='k', markersize=2, label='EMG data with notch filters')
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.ylabel("Power")
plt.xlabel("Frequency (Hz)")
plt.title(f"Frequency Domain of Channel {[CHAN_IX]} after Notch Filter")
plt.tight_layout()

#pre-processing step 2: 4th order butterworth high pass filter with cutoff at 65 Hz
b, a = signal.butter(4, 65.0, btype='high', analog=False, fs=SAMPLE_RATE)
butterworthdata = signal.filtfilt(b, a, notchdata)

#plotting the frequency spectrum
duration = len(notchdata) / SAMPLE_RATE
t = np.linspace(0, duration, len(notchdata), endpoint=False)
plt.title(f"Frequency Spectra of Channel {[CHAN_IX]} after Notch Filter")


# plt.figure(plt.figure(figsize=(10,4), dpi=200))
# plt.plot(t, butterworthdata)
# plt.gca().spines["top"].set_visible(False)
# plt.gca().spines["right"].set_visible(False)
# plt.tight_layout()

#rectified emg spectra
f, pxx = signal.welch(butterworthdata, nperseg=NPERSEG, nfft=NFFT, noverlap=NOVERLAP, fs=SAMPLE_RATE)
plt.figure(figsize=(10,4), dpi = 200)
plt.loglog(f, pxx, '-o', alpha=0.4, color='k', markersize=2)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.ylabel("Power")
plt.xlabel("Frequency (Hz)")
plt.title("Frequency Domain of Butterworth Filtered Data")

channel_names = emgdata['emg_names'].flatten()
channel_names = emgdata['emg_names'][0]
channel_names = [str(name[0]) for name in channel_names]

notchdata = rawdata[:, CHAN_IX]
for freq, q in zip(notch_frequencies, Q):
    b, a = signal.iirnotch(freq, q, SAMPLE_RATE)
    notchdata = signal.filtfilt(b, a, notchdata)

# Apply Butterworth filter
b, a = signal.butter(4, 65.0, btype='high', analog=False, fs=SAMPLE_RATE)
butterworthdata = signal.filtfilt(b, a, notchdata)

# Rectify the EMG signal
rectifieddata = np.abs(butterworthdata)

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


f, pxx = signal.welch(rawdata[:, CHAN_IX], nperseg=NPERSEG, nfft=NFFT, noverlap=NOVERLAP, fs=SAMPLE_RATE)
plt.loglog(f, pxx, '-o', alpha=0.4, color='r', markersize=2, label = "Raw data")

f, pxx = signal.welch(notchdata, nperseg=NPERSEG, nfft=NFFT, noverlap=NOVERLAP, fs=SAMPLE_RATE)
plt.loglog(f, pxx, '-o', alpha=0.4, color='b', markersize=2, label = "Notch-filtered data")

f, pxx = signal.welch(butterworthdata, nperseg=NPERSEG, nfft=NFFT, noverlap=NOVERLAP, fs=SAMPLE_RATE)
plt.loglog(f, pxx, '-o', alpha=0.4, color='k', markersize=2, label = "Butterworth-filtered data")

f, pxx = signal.welch(rectifieddata, nperseg=NPERSEG, nfft=NFFT, noverlap=NOVERLAP, fs=SAMPLE_RATE)
plt.loglog(f, pxx, '-o', alpha=0.4, color='g', markersize=2, label = "Rectified data")

plt.legend()
# %%
#rectified emg for all signals -- able to run independently after loading rawdata
#proceessing pipeline for grid of all channels
num_channels = 12
bandwidth = 2
notch_frequencies= [60, 120, 240, 300, 420]
Q = [freq / bandwidth for freq in notch_frequencies]  # Compute Q for each frequency


fig, axes = plt.subplots(3, 4, figsize=(20, 15), dpi=200)
axes=axes.flatten()

for CHAN_IX in range(num_channels):
    # Plotting the raw data for the specific channel
    ax = axes[CHAN_IX]
    f, pxx = signal.welch(rawdata[:,CHAN_IX], nperseg=NPERSEG, nfft=NFFT, noverlap=NOVERLAP, fs=SAMPLE_RATE)
    ax.loglog(f, pxx, '-o', alpha=0.4, color='k', markersize=2, label='ENG raw data')
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_title(f"Spectra for Muscle {CHAN_IX+1}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(3, 4, figsize=(20, 15))
axes = axes.flatten()

channel_names = emgdata['emg_names'].flatten()
channel_names = emgdata['emg_names'][0]
channel_names = [str(name[0]) for name in channel_names]

for CHAN_IX in range(num_channels):
    notchdata = rawdata[:, CHAN_IX]
    for freq, q in zip(notch_frequencies, Q):
        b, a = signal.iirnotch(freq, q, SAMPLE_RATE)
        notchdata = signal.filtfilt(b, a, notchdata)
    
    # Apply Butterworth filter
    b, a = signal.butter(4, 65.0, btype='high', analog=False, fs=SAMPLE_RATE)
    butterworthdata = signal.filtfilt(b, a, notchdata)

    # Rectify the EMG signal
    rectifieddata = np.abs(butterworthdata)

    # Time vector for plotting
    duration = len(rectifieddata) / SAMPLE_RATE
    t = np.linspace(0, duration, len(rectifieddata), endpoint=False)

    # Plot rectified EMG data
    ax = axes[CHAN_IX]
    ax.plot(t, rectifieddata, '-o', alpha=0.4, color='k', markersize=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel(f"{channel_names[CHAN_IX]}")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Rectified EMG for Muscle {channel_names[CHAN_IX]}")
    ax.set_xlim([0,30])


plt.tight_layout()
plt.show()

# %%
#rectifying emg signal across a single channel
rectifieddata = np.abs(butterworthdata)

#timeplot
duration = len(rectifieddata) / SAMPLE_RATE
t = np.linspace(0, duration, len(rectifieddata), endpoint=False)

plt.figure(figsize=(10,4), dpi=200)
plt.plot(t, rectifieddata)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.ylabel("Amplitude")
plt.xlabel("Time(s)")
plt.title(f"Rectified EMG for {channel_names[CHAN_IX]}")
plt.xlim([0,30])

#frequency plot
f, pxx = signal.welch(rectifieddata, nperseg=NPERSEG, nfft=NFFT, noverlap=NOVERLAP, fs=SAMPLE_RATE)
plt.figure(figsize=(10, 4), dpi=200)
plt.loglog(f, pxx, '-o', alpha=0.4, color='k', markersize=2)
# %%
