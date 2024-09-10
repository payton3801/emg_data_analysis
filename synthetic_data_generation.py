# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal

# %%
#creating and printing the sine wave
SAMPLE_RATE = 2000
AMPLITUDE = 1
FREQUENCY = 20 # Hz
PHASE = 0
DURATION_S = 10 # seconds
N_HAR = 3 # number of harmonics

def sine_wave_generator(duration_s, sample_rate, freq, amp=1, phase=0, n_harmonics=0):
    """ generates a sine wave with the specified parameters

        duration: (seconds) how long to generate the sine wave for
        sample_rate: (Hz) sampling frequency
        freq: (Hz) fundamental frequency of sine wave
        phase: (radians) phase shift of the sine wave
    """
    # generate the time vector
    t = np.arange(0, duration_s, 1/sample_rate)  #gives us the sample rate number of points in duration_s seconds
    x = (amp/(n_harmonics+1)) * np.sin((2*np.pi * freq * t) + phase) #sine formula
    if n_harmonics > 0:
        for i in range(n_harmonics):
            # create harmonic of fundamental frequency
            x_har = (amp/(n_harmonics+1)) * np.sin((2*np.pi * freq*(i+2) * t) + phase) #i+2 because we want the first harmonic to be 2*freq
            # add harmonic to signal
            x = x + x_har
    return t, x

def spike_generator(x):
    pass

def noise_generator(x):
    pass

t, sinewave = sine_wave_generator(DURATION_S, SAMPLE_RATE, FREQUENCY, amp=AMPLITUDE, phase=PHASE, n_harmonics=N_HAR) #storing time vector and sinewave signal  

true_frequencies = []

# add the fundamental frequency
true_frequencies.append(FREQUENCY) #lists where the harmonic frequencies are
if N_HAR > 0:
    for i in range(N_HAR):
        true_frequencies.append(FREQUENCY*(i+2))

fig = plt.figure(figsize=(10,4), dpi=200)
ax = fig.add_subplot(111)
ax.plot(t, sinewave, 'k') #plots time vs sinewave

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlim([0,1])

# %%
# transferring this to the frequency subspace

# compute the power spectra using the welch medthod
NFFT = 5000 #fs/nfft is the frequency resolution, FFT computed to this number of points
NPERSEG = NFFT #each segment has the same length as number of points in FFT
NOVERLAP = 500 #points of overlap, must be less than NPERSEG
NOVERLAPS = [0, 500, 1000, 2000, 4999] #different overlap values to test
fontsize = 8
fig, axs = plt.subplots(len(NOVERLAPS), 1, figsize=(5,7), dpi=300)
#fig = plt.figure(figsize=(5,5), dpi=300) #dpi is the resolution
#ax = fig.add_subplot(111) # grid with 1 row, 1 column, and this subplot being the only

for i, current_NOVERLAP in enumerate(NOVERLAPS):    
    #signal.welch computes the power spectral density of the input signal
    f, pxx = signal.welch(sinewave, nfft=NFFT, noverlap=current_NOVERLAP, nperseg=NPERSEG, fs=SAMPLE_RATE) #f=array of sample frequencies, pxx=power spectral density of signal
    # plot frequency spectra in log-log
    axs[i].semilogy(f, pxx, '-o', alpha=0.4, color='k', markersize=2) #log scale for both x and y axes, alpha is the transparency of the line
    # make plot prettier
    axs[i].spines["top"].set_visible(False)
    axs[i].spines["right"].set_visible(False)    
    axs[i].set_ylabel("Power", fontsize=fontsize)
    axs[i].set_xlim([0, 125])
    axs[i].set_title(f"NFFT: {NFFT}, NOLAP: {current_NOVERLAP}", fontsize=fontsize)
    for true_freq in true_frequencies:
        axs[i].axvline(x=true_freq, color='k', linestyle='--', alpha=0.4) #plotting vertical lines at the true frequencies    
    if i == len(NOVERLAPS)-1:
        axs[i].set_xlabel("Frequency (Hz)", fontsize=fontsize)
fig.tight_layout()


# %% 
#Just like we set up a code cell that looped through different values of N overlap and kept constant NFFT, 
#setup another cell that instead fixes NOVERLAP and loops over different values of NFFT
NOVERLAP = 500 
#whyy does the first value have to be over 1000? <- LW: I don't think it has to be see below. It has to satisfy NFFT > NOVERLAP
NFFTS = [900, 1000, 2000, 10000] 

fig, axs = plt.subplots(len(NFFTS), 1, figsize=(5,5), dpi=300)
#ax = fig.add_subplot(111)

for i, current_NFFT in enumerate(NFFTS):
    current_NPERSEG = current_NFFT
    f, pxx = signal.welch(sinewave, nperseg= current_NPERSEG, nfft= current_NFFT, noverlap = NOVERLAP, fs = SAMPLE_RATE)
    axs[i].semilogy(f, pxx, '-o', color='k', markersize=2, alpha=0.4)
    # make plot prettier
    axs[i].spines["top"].set_visible(False)
    axs[i].spines["right"].set_visible(False)
    axs[i].set_xlabel("Frequency (Hz)")
    axs[i].set_ylabel("Power")
    axs[i].set_xlim([0, 125])
    axs[i].set_title(f"NFFT: {current_NFFT}, NOLAP: {NOVERLAP}")
    for true_freq in true_frequencies:
        axs[i].axvline(x=true_freq, color='k', linestyle='--', alpha=0.4) #plotting vertical lines at the true frequencies            

fig.tight_layout()


# %%
# now that we better understand how to use frequency estimation tools

# lets try to simulate a signal with powerline noise.

# create a signal that has powerline noise at 60Hz and at 6 harmonics (120, 180, 240, ...)
# plot 1: plot the signal
FREQUENCY = 60
SAMPLE_RATE = 2000
DURATION_S = 10
N_HAR = 6
t, sinewave = sine_wave_generator(DURATION_S, SAMPLE_RATE, FREQUENCY, n_harmonics=N_HAR) 

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,4), dpi=200)
ax1.plot(t, sinewave, 'k')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xlim([0,1])
ax1.set_ylabel("Power")
ax1.set_xlabel("Time (s)")
ax1.set_title("Time Domain")
# plot 2: compute the FFT using appropriate parameters for the power spectral density estimate (using signal.welch) and plot the spectra
#Use sampling rate/NFFT = frequency resolution (0.1) to find ideal NFFT value
#Use .2 (NFFT) = NOVERLAP

NFFT = 20000
NPERSEG = 20000
NOVERLAP = 4000

f, pxx = signal.welch(sinewave, nperseg = NPERSEG, nfft = NFFT, noverlap = NOVERLAP, fs = SAMPLE_RATE)
ax2.semilogy(f, pxx, '-o',color = 'k', markersize = 2, alpha = 0.4)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.set_ylabel("Power")
ax2.set_xlabel("Frequency (Hz)")
ax2.set_xlim([0,370])
ax2.set_title("Frequency Domain")
fig.tight_layout()
# %%
