# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
spike_index = np.isin(time, [2,4,6,8])

abs_sinewave[spike_index] += np.random.uniform(3,5)

# %%
#the final plot
plt.plot(time, abs_sinewave)
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
# %%
