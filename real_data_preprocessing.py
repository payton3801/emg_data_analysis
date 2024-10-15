# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.io import loadmat# %%
#plotting the real emg data, looking at keys and info
emgdata = loadmat('J10_s10_i0_pref.mat')
rawdata = emgdata['emg_full_raw'].flatten() #needs to be  a 1D array for welch function
filterdata = emgdata['emg_full_fil'].flatten()
for key in emgdata:
    if not key.startswith('__'):
        print(f"\nKey: {key}")
        print(f"Type: {type(emgdata[key])}")
        print(f"Shape: {emgdata[key].shape}")

plt.rcParams['agg.path.chunksize'] = 10000


# %%
#emg data pramaters

#calculate sampling rate
time_vector = emgdata['t_emg'].flatten()
intervals = np.diff(time_vector)
SAMPLE_RATE = 1/np.mean(intervals) 

#parameters
NFFT = 50000 #Use sampling rate/NFFT = 0.1
NPERSEG = 50000
NOVERLAP = 10000 #Use .2 (NFFT) = NOVERLAP

# %%
#plotting the raw data
plt.figure(figsize=(10,4), dpi = 200)
plt.plot(rawdata, alpha=0.4, color='k', label='ENG raw data')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Raw Data")
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.xlim([0,30])

# %%
#plotting the spectra
f, pxx = signal.welch(rawdata, nperseg=NPERSEG, nfft=NFFT, noverlap=NOVERLAP, fs=SAMPLE_RATE)
plt.figure(figsize=(10,4), dpi=200)
plt.loglog(f, pxx, '-o', alpha=0.4, color='k', markersize=2, label='ENG raw data')
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.ylabel("Power")
plt.xlabel("Frequency (Hz)")
plt.title("Frequency Domain")
plt.tight_layout()
#plt.xlim([0,500])

# %%
#pre-processing step 1: notch filters at 60 Hz, 120 Hz, 240 Hz, 300 Hz, 420 Hz
notch_frequencies= [60, 120, 240, 300, 420]
Q = 30.0

notchdata = rawdata
for frequencies in notch_frequencies:
    b, a = signal.iirnotch(frequencies, Q, SAMPLE_RATE)
    notchdata = signal.filtfilt(b, a, notchdata)

f, pxx = signal.welch(notchdata, nperseg=NPERSEG, nfft=NFFT, noverlap=NOVERLAP, fs=SAMPLE_RATE)
plt.figure(figsize=(10,4), dpi = 200)
plt.loglog(f, pxx, '-o', alpha=0.4, color='k', markersize=2, label='EMG data with notch filters')
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.ylabel("Power")
plt.xlabel("Frequency (Hz)")
plt.title("Frequency Domain")
plt.tight_layout()

# %%
#pre-processing step 2: 4th order butterworth high pass filter with cutoff at 65 Hz
b, a = signal.butter(4, 65.0, btype='high', analog=False, fs=SAMPLE_RATE)
output = signal.filtfilt(b, a, notchdata)

#plotting the frequency spectrum
duration = len(notchdata) / SAMPLE_RATE
t = np.linspace(0, duration, len(notchdata), endpoint=False)

plt.figure(plt.figure(figsize=(10,4), dpi=200))
plt.plot(t, output)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.tight_layout()

f, pxx = signal.welch(output, nperseg=NPERSEG, nfft=NFFT, noverlap=NOVERLAP, fs=SAMPLE_RATE)
plt.figure(figsize=(10,4), dpi = 200)
plt.loglog(f, pxx, '-o', alpha=0.4, color='k', markersize=2)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.ylabel("Power")
plt.xlabel("Frequency (Hz)")
plt.title("Frequency Domain of Butterworth Filtered Data")
# %%
#scatterplot of the above
duration = len(notchdata) / SAMPLE_RATE
t = np.linspace(0, duration, len(notchdata), endpoint=False)
plt.scatter(t, output)
# %%
