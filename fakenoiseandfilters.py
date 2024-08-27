# %%
import numpy as np
import pandas as pd
7import matplotlib.pyplot as plt

# %%
#creating and printing the sine wave
sample_rate = 2000
amp = 1
freq = .2
phase = 2
start_time = 0
end_time = 10
time = np.arange(start_time, end_time, 1/sample_rate)
sinewave = amp * np.sin(2 * np.pi * freq * time + phase)
abs_sinewave = np.abs(sinewave)  # Compute the absolute values


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
