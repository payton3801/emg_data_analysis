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
    x = amp * np.sin((2*np.pi * freq * t) + phase) #sine formula
    if n_harmonics > 0:
        for i in range(n_harmonics):
            # create harmonic of fundamental frequency
            x_har = amp * np.sin((2*np.pi * freq*(i+2) * t) + phase) #i+2 because we want the first harmonic to be 2*freq
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

plt.plot(t, sinewave, 'k') #plots time vs sinewave

plt.xlim([0,1])

# %%
# what does this look like in frequency space?

# compute the power spectra using the welch medthod
NFFT = 5000 #fs/nfft is the frequency resolution, FFT computed to this number of points
NPERSEG = NFFT #each segment has the same length as number of points in FFT
NOVERLAP = 500 #points of overlap, must be less than NPERSEG
N_OVERLAPS = [0, 500, 1000] #different overlap values to test

fig = plt.figure(figsize=(5,5), dpi=300) #dpi is the resolution
ax = fig.add_subplot(111) # grid with 1 row, 1 column, and this subplot being the only

for N_OVERLAP in N_OVERLAPS:
    #signal.welch computes the power spectral density of the input signal
    f, pxx = signal.welch(sinewave, nfft=NFFT, noverlap=N_OVERLAP, nperseg=NPERSEG, fs=SAMPLE_RATE) #f=array of sample frequencies, pxx=power spectral density of signal
    # plot frequency spectra in log-log
    ax.loglog(f, pxx, '-o', alpha=0.4) #log scale for both x and y axes, alpha is the transparency of the line
    

for true_freq in true_frequencies:
    ax.axvline(x=true_freq, color='r', linestyle='--') #plotting vertical lines at the true frequencies

# make plot prettier
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power")
ax.set_xlim([0, 200])


# %% 
# now test changing the number of points in the FFT in the same way 


# %%
#Adding spikes to signify powerline harmonics
num_harmonics = int(input("Enter the number of harmonics: "))

#spiketimes = [2 * i for i in range(1, num_harmonics+1)]
#spike_index = np.isin(time, spiketimes) 
#abs_sinewave[spike_index] += np.random.uniform(3,5)

