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

        duration: (seconds) how long to generate the sine wave
        sample_rate: (Hz) sampling frequency
        freq: (Hz) fundamental frequency of sine wave
        phase: (radians) phase shift of the sine wave
    """
    # generate the time vector
    t = np.arange(0, duration_s, 1/sample_rate)
    x = amp * np.sin((2*np.pi * freq * t) + phase)
    if n_harmonics > 0:
        for i in range(n_harmonics):
            # create harmonic of fundamental frequency
            x_har = amp * np.sin((2*np.pi * freq*(i+2) * t) + phase)
            # add harmonic to signal
            x = x + x_har
    

    return t, x

def spike_generator(x):
    pass

def noise_generator(x):
    pass

t, sinewave = sine_wave_generator(DURATION_S, SAMPLE_RATE, FREQUENCY, amp=AMPLITUDE, phase=PHASE, n_harmonics=N_HAR)    

true_frequencies = []

# add the fundamental frequency
true_frequencies.append(FREQUENCY)
if N_HAR > 0:
    for i in range(N_HAR):
        true_frequencies.append(FREQUENCY*(i+2))

plt.plot(t, sinewave, 'k')

plt.xlim([0,1])

# %%
# what does this look like in frequency space?

# compute the power spectra
NFFT = 5000
NPERSEG = NFFT
NOVERLAP = 500
N_OVERLAPS = [0, 500, 1000]

fig = plt.figure(figsize=(5,5), dpi=300)
ax = fig.add_subplot(111)

for N_OVERLAP in N_OVERLAPS:
    f, pxx = signal.welch(sinewave, nfft=NFFT, noverlap=N_OVERLAP, nperseg=NPERSEG, fs=SAMPLE_RATE)
    # plot frequency spectra in log-log
    ax.loglog(f, pxx, '-o', alpha=0.4)
    

for true_freq in true_frequencies:
    ax.axvline(x=true_freq, color='r', linestyle='--')

# make plot prettier
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power")
ax.set_xlim([0, 200])


# %% 
# now test changing the number of points in the FFT in the same way 


# %% 
#time = np.arange(start_time, end_time, 1/sample_rate)
#sinewave = amp * np.sin(2 * np.pi * freq * time + phase)
#abs_sinewave = np.abs(sinewave)  # Compute the absolute values


# %%
#Adding spikes to signify powerline harmonics
num_harmonics = int(input("Enter the number of harmonics: "))

spiketimes = [2 * i for i in range(1, num_harmonics+1)]
spike_index = np.isin(time, spiketimes) 
abs_sinewave[spike_index] += np.random.uniform(3,5)

# %%
#the final plot
plt.plot(time, abs_sinewave)
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
# %%
