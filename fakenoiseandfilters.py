# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
#creating and printing the sine wave
sample_rate = 2000
amp = 1
freq = .5
phase = 2
start_time = 0
end_time = 10
time = np.arange(start_time, end_time, 1/sample_rate)
sinewave = amp * np.sin(2 * np.pi * freq * time + phase)
plt.plot(time, sinewave)

# %%

# %%
