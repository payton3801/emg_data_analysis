# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.io import loadmat

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
        n_harmonics: number of harmonics to add to the sine wave
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

def noise_generator(x=sinewave, noise_mean=0, noise_std=1):
    """ generates randomized Gaussian noise with specified mean and standard deviation
    
        noise_mean = mean of Gaussian distribution
        noise_std = standard deviation of Gaussian distribution
    """
    # TODO: create random Gaussian noise and add it to the signal x
    # TODO: noise is parameterized by `noise_mean` and `noise_std`
    # HINT: use numpy's `random.randn` to generate a sequence of random samples from a normal distribution of length `x`
    # HINT: to change the mean of the distribution, what operation (e.g., addition, subtraction, division, multiplication) should be use to incorporate `noise_mean`
    # HINT: to change the std of the distribution, what operation should we use to incorporate `noise_std`
    noise = noise_mean + noise_std * np.random.randn(len(x))
    return noise 

t, sinewave = sine_wave_generator(DURATION_S, SAMPLE_RATE, FREQUENCY, amp=AMPLITUDE, phase=PHASE, n_harmonics=N_HAR) #storing time vector and sinewave signal  
scaled_sinewave = sinewave * 10
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

# compute the power spectra using the welch method
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
# create a signal that has powerline noise at 60Hz and at 6 harmonics (120, 180, 240, ...)

# plot 1: plot the signal
FREQUENCY = 60
SAMPLE_RATE = 2000
DURATION_S = 10
N_HAR = 6
t, sinewave = sine_wave_generator(DURATION_S, SAMPLE_RATE, FREQUENCY, n_harmonics=N_HAR) 
scaled_sinewave = sinewave *  1
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

f, pxx = signal.welch(x, nperseg = NPERSEG, nfft = NFFT, noverlap = NOVERLAP, fs = SAMPLE_RATE)
ax2.semilogy(f, pxx, '-o',color = 'k', markersize = 2, alpha = 0.4)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.set_ylabel("Power")
ax2.set_xlabel("Frequency (Hz)")
ax2.set_xlim([0,370])
ax2.set_title("Frequency Domain")
fig.tight_layout()

# %% - plot noise distributions with changing parameters

# noise_mean = 0, noise_std = 1
noise_0 = noise_generator()
plt.hist(noise_0, bins=100, color='m', alpha=0.4)
# -- plot the distribution, HINT: use plt.hist. Adjust resolution of histogram by modifying kwarg 'bins'

# noise_mean = 2, noise_std = 1
NOISE_MEAN = 10
NOISE_STD = 30
noise_1 = noise_generator(noise_mean = NOISE_MEAN, noise_std = NOISE_STD)
plt.hist(noise_1, bins=100, color='c', alpha=0.4)
# noise_mean = 0, noise_std = 4

# noise_mean = -2, noise_std = 0.25
NOISE_MEAN = -5
NOISE_STD = 0.1
noise_2 = noise_generator(noise_mean = NOISE_MEAN, noise_std = NOISE_STD)
plt.hist(noise_2, bins=100, color='g', alpha=0.4)

# %% -- PLotting power vs frequency spectra of clean and noisy sinewaves

#adding noise to sinewave
noisy_sinewave_0 = scaled_sinewave + noise_0 #*0.00000000000001 #added scaling factor to bring noise to same scale as sinewave
noisy_sinewave_1 = scaled_sinewave + noise_1 #*0.00000000000001
noisy_sinewave_2 = scaled_sinewave + noise_2 #*0.00000000000001

#defining parameters
NFFT = 20000
NPERSEG = 20000
NOVERLAP = 4000
SAMPLE_RATE = 2000

#computing power spectral density of the clean and noisy sinewaves
f, pxx = signal.welch(scaled_sinewave, nperseg=NPERSEG, nfft=NFFT, noverlap=NOVERLAP, fs=SAMPLE_RATE)
f, pxx_noisy_sinewave_0 = signal.welch(noisy_sinewave_0, nperseg=NPERSEG, nfft=NFFT, noverlap=NOVERLAP, fs=SAMPLE_RATE)
f, pxx_noisy_sinewave_1 = signal.welch(noisy_sinewave_1, nperseg=NPERSEG, nfft=NFFT, noverlap=NOVERLAP, fs=SAMPLE_RATE)
f, pxx_noisy_sinewave_2 = signal.welch(noisy_sinewave_2, nperseg=NPERSEG, nfft=NFFT, noverlap=NOVERLAP, fs=SAMPLE_RATE)

f, pxx_noise_0 = signal.welch(noise_0, nperseg=NPERSEG, nfft=NFFT, noverlap=NOVERLAP, fs=SAMPLE_RATE)
f, pxx_noise_1 = signal.welch(noise_1, nperseg=NPERSEG, nfft=NFFT, noverlap=NOVERLAP, fs=SAMPLE_RATE)
f, pxx_noise_2 = signal.welch(noise_2, nperseg=NPERSEG, nfft=NFFT, noverlap=NOVERLAP, fs=SAMPLE_RATE)

#plotting the original and noisy plots on top of each other
plt.figure(figsize=(10, 4), dpi=200)

plt.semilogy(f, pxx, '-o', alpha=0.4, color='k', markersize=2, label='Clean Sinewave')
plt.semilogy(f, pxx_noisy_sinewave_0, '-o', alpha=0.4, color='m', markersize=2, label='Noisy Sinewave 0')
plt.semilogy(f, pxx_noisy_sinewave_1, '-o', alpha=0.4, color='c', markersize=2, label='Noisy Sinewave 1')
plt.semilogy(f, pxx_noisy_sinewave_2, '-o', alpha=0.4, color='g', markersize=2, label='Noisy Sinewave 2')

#plot details
plt.xlim([0, 370])
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.title("Overlaid Spectra of Clean and Noisy Sinewave")
plt.legend()
plt.tight_layout()
plt.show()


# %%
#plotting just the noise
plt.figure(figsize=(10,4), dpi=200)
plt.semilogy(f, pxx_noise_0, '-o', alpha=0.4, color='m', markersize=2, label='Noisy Sinewave 0')
plt.semilogy(f, pxx_noise_1, '-o', alpha=0.4, color='c', markersize=2, label='Noisy Sinewave 1')
plt.semilogy(f, pxx_noise_2, '-o', alpha=0.4, color='g', markersize=2, label='Noisy Sinewave 2')

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
#also, change figure size to 10, 4

plt.xlim([0,370])
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.title("Noise Spectra")
plt.legend
plt.tight_layout()
plt.show()

