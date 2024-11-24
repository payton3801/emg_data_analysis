# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.io import loadmat
import math

# %%
kindata = loadmat('J10_s20_i0_pref.mat')
for key in kindata:
    if not key.startswith('__'):
        print(f"\nKey: {key}")
        print(f"Type: {type(kindata[key])}")
        print(f"Shape: {kindata[key].shape}")

joints = kindata['joints_raw_nonSeg']
num_joints = joints.shape[0]
cmap = plt.get_cmap('viridis')
colors = cmap(np.linspace(0, 1, num_joints))

# %%
#calculate sampling rate
time_vector = kindata['t_kin'].flatten()
intervals = np.diff(time_vector)
SAMPLE_RATE = 1 / np.mean(intervals)
SAMPLE_RATE = round(SAMPLE_RATE)
TARGET_SAMPLE_RATE = 500

joint_names_1 = kindata['joints_names'].flatten()
joint_names = [name[0] for name in joint_names_1]
JOINT_IX = 1
BLIP_DURATION = 0.03
BLIP_SAMPLES = int(BLIP_DURATION * TARGET_SAMPLE_RATE) 

# %%
#low pass filtering the data 
def apply_butterworth_filter(x, sample_rate=SAMPLE_RATE, cutoff=40.0, order=4, padlen = 150):
    """
        apply 4th order butterworth low pass filter with cutoff at 40 Hz
        x = signal data being filtered
        sample_rate = samples per second of the data
        cutoff = cutoff frequency
        order = filter order
    """
    #padded_x = np.pad(x, padlen, mode = 'edge')

    b, a = signal.butter(4, 40.0, btype='low', analog=False, fs=SAMPLE_RATE)
    butterworth = signal.filtfilt(b, a, x)

    #butterworth = butterworth[padlen:-padlen]

    return butterworth


for JOINT_IX in range(num_joints):
    joint_data = joints[JOINT_IX][0]
    if joint_data.shape[1] == 2:
        for col in range(2):
            
            applied_butter = apply_butterworth_filter(joint_data[:, col], SAMPLE_RATE)

            plt.plot(time_vector, applied_butter, alpha=0.7, marker= 'o', color = colors[JOINT_IX], markersize=3, label=joint_names[JOINT_IX])

    else:

        applied_butter = apply_butterworth_filter(joint_data[:, 0])

        plt.plot(time_vector, applied_butter, alpha=0.7, marker= 'o', color = colors[JOINT_IX], markersize=3, label=joint_names[JOINT_IX])

plt.xlabel('Time (s)') 
plt.ylabel('Kinematic Angle (degrees)')  
plt.title('Kinematic Angle per Joint Over Time')
plt.xlim([0, 5])

plt.legend()
plt.show()

# %%
#extracting the 2d arrays
for i, element in enumerate(joints):
    inner_array = element[0]
    print(inner_array.shape)


# %%
#resampling and filtering the data
plt.figure(figsize=(7, 4), dpi=200)


UPSAMPLING = 5
DOWNSAMPLING = 2

for JOINT_IX in range(num_joints):
    joint_data = joints[JOINT_IX][0]
    if joint_data.shape[1] == 2:
        for col in range(2):


            filtered_data = apply_butterworth_filter(joint_data[:, col])
            resampled_data = signal.resample_poly(filtered_data, up=UPSAMPLING, down= DOWNSAMPLING) 


            duration = len(resampled_data) / TARGET_SAMPLE_RATE
            t2 = np.linspace(0, duration, len(resampled_data), endpoint=False)

            # Plotting the resampled data over time
            plt.plot(t2, resampled_data, alpha=0.7, marker= 'o', color = colors[JOINT_IX], markersize=3, label=f' {joint_names[JOINT_IX] } Col {col+1}')

    else:

        filtered_data = apply_butterworth_filter(joint_data[:, 0])
        resampled_data = signal.resample_poly(filtered_data, up = UPSAMPLING, down = DOWNSAMPLING)


        duration = len(resampled_data) / TARGET_SAMPLE_RATE
        t2 = np.linspace(0, duration, len(resampled_data), endpoint=False)

        # Plotting the resampled data over time
        plt.plot(t2, resampled_data, alpha=0.7, marker= 'o', color = colors[JOINT_IX], markersize=3, label=joint_names[JOINT_IX])

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.ylabel("Amplitude")
plt.xlabel("Time (s)")
plt.xlim([0, 5])
plt.title(f"Resampled and EMG")
plt.tight_layout()
plt.legend()
plt.show()







# %% -
#plotting both overlaid

plt.figure(figsize=(10, 6))
joint_names_1 = kindata['joints_names'].flatten()
joint_names = [name[0] for name in joint_names_1]

for JOINT_IX in range(num_joints):
    # Access the inner 2D array
    joint_data = joints[JOINT_IX][0]
    min_length_original = min(len(time_vector), len(joint_data))
    plt.plot(time_vector[:min_length_original], joint_data[:min_length_original], linestyle='--', marker='o', color=colors[JOINT_IX], alpha=0.9, label=f'Original {joint_names[JOINT_IX]}')

    if joint_data.shape[1] == 2:
        for col in range(2):
            # Resample the data
            filtered_data = apply_butterworth_filter(joint_data[:, col])
            resampled_data = signal.resample_poly(filtered_data, up=UPSAMPLING, down=DOWNSAMPLING)

            duration = len(resampled_data) / TARGET_SAMPLE_RATE
            t2 = np.linspace(0, duration, len(resampled_data), endpoint=False)

            # Ensure the lengths match for plotting
            min_length_resampled = min(len(t2), len(resampled_data))

            # Plotting the resampled data over time
            plt.plot(t2[:min_length_resampled], resampled_data[:min_length_resampled], linestyle='-', color='k', alpha=0.2, marker='o', label=f'Resampled {joint_names[JOINT_IX]} Col {col+1}')
    else:
        # Resample the data
        filtered_data = apply_butterworth_filter(joint_data[:, 0])
        resampled_data = signal.resample_poly(filtered_data, up=UPSAMPLING, down=DOWNSAMPLING)

        duration = len(resampled_data) / TARGET_SAMPLE_RATE
        t2 = np.linspace(0, duration, len(resampled_data), endpoint=False)

        # Ensure the lengths match for plotting
        min_length_resampled = min(len(t2), len(resampled_data))

        # Plotting the resampled data over time
        plt.plot(t2[:min_length_resampled], resampled_data[:min_length_resampled], linestyle='-', color='k', alpha=0.2, marker='o', label=f'Resampled {joint_names[JOINT_IX]}')

# Set plot labels and title
plt.xlabel('Time (s)')
plt.ylabel('Kinematic Angle (degrees)')  # Specific label for kinematic angle
plt.title('Original and Resampled Kinematic Angles for All Joints')
plt.legend()
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.xlim([0, .5])
plt.show()

# %%
