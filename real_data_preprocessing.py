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
