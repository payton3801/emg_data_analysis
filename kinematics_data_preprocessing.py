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


# %%
#calculate sampling rate
time_vector = kindata['t_kin'].flatten()
intervals = np.diff(time_vector)
SAMPLE_RATE = 1 / np.mean(intervals)
joint_names_1 = kindata['joints_names'].flatten()
joint_names = [name[0] for name in joint_names_1]
JOINT_IX = 1

# %%
#extracting the 2d arrays
for i, element in enumerate(joints):
    inner_array = element[0]
    print(inner_array.shape)

# %%
plt.figure(figsize=(10, 6))

for i, joint in enumerate(joints):
    # Flatten the inner array to 1D if necessary
    joint_flat = np.concatenate(joint) #ravel #combines into a single array, flattened the array
    min_length = min(len(time_vector), len(joint_flat)) #allows time array and combined joint array to have the same length for plotting
    plt.plot(time_vector[:min_length], joint_flat[:min_length], marker= 'o',alpha=0.4, label=joint_names[i])

plt.xlabel('Time (s)') 
plt.ylabel('Kinematic Angle (degrees)')  
plt.title('Kinematic Angle per Joint Over Time')
plt.xlim([0, 1])

plt.legend()
plt.show()


# %%
#resampling the data
num_joints = joints.shape[0]
plt.figure(figsize=(7, 4), dpi=200)
cmap = plt.get_cmap('viridis')
colors = cmap(np.linspace(0, 1, num_joints))

SAMPLE_RATE = round(SAMPLE_RATE)
TARGET_SAMPLE_RATE = 500
UPSAMPLING = 5
DOWNSAMPLING = 2


for JOINT_IX in range(num_joints):
    joint_data = joints[JOINT_IX][0]
    if joint_data.shape[1] == 2:
        for col in range(2):
            
            resampled_data = signal.resample_poly(joint_data[:, col], up=UPSAMPLING, down= DOWNSAMPLING)
            duration = len(resampled_data) / TARGET_SAMPLE_RATE
            t2 = np.linspace(0, duration, len(resampled_data), endpoint=False)

            # Plotting the resampled data over time
            plt.plot(t2, resampled_data, alpha=0.7, marker= 'o', color = colors[JOINT_IX], markersize=3, label=joint_names[JOINT_IX])

    else:

        resampled_data = signal.resample_poly(joint_data[:,0], up=UPSAMPLING, down= DOWNSAMPLING)
        duration = len(resampled_data) / TARGET_SAMPLE_RATE
        t2 = np.linspace(0, duration, len(resampled_data), endpoint=False)

        # Plotting the resampled data over time
        plt.plot(t2, resampled_data, alpha=0.7, marker= 'o', color = colors[JOINT_IX], markersize=3, label=joint_names[JOINT_IX])

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.ylabel("Amplitude")
plt.xlabel("Time (s)")
plt.xlim([0, 1])
plt.title(f"Resampled EMG")
plt.tight_layout()
plt.legend()
plt.show()

# %%
#plotting both overlaid
plt.figure(figsize=(10, 6))
joint_names_1 = kindata['joints_names'].flatten()
joint_names = [name[0] for name in joint_names_1]

for JOINT_IX in range(num_joints):
    # Access the inner 2D array
    joint_data = joints[JOINT_IX][0]
    if joint_data.shape[1] == 2:

        min_length_original = min(len(time_vector), len(joint_data))
        plt.plot(time_vector[:min_length_original], joint_data[:min_length_original], linestyle='--', marker = 'o', color=colors[JOINT_IX], alpha=0.9, label=f'Original {joint_names[JOINT_IX]}')


        for col in range(2):

            # Resample the data
            resampled_data = signal.resample_poly(joint_data[:, col], up=UPSAMPLING, down=DOWNSAMPLING)
            duration = len(resampled_data) / TARGET_SAMPLE_RATE
            t2 = np.linspace(0, duration, len(resampled_data), endpoint=False)

            # Ensure the lengths match for plotting
            min_length_resampled = min(len(t2), len(resampled_data))

            # Plotting the original and resampled data over time
            plt.plot(t2, resampled_data, linestyle='-', color = 'k', alpha=0.2, marker='o', label=f'Resampled {joint_names[JOINT_IX]} Col {col+1}')
    else:

        resampled_data = signal.resample_poly(joint_data[:, 0], up=UPSAMPLING, down=DOWNSAMPLING)
        duration = len(resampled_data) / TARGET_SAMPLE_RATE
        t2 = np.linspace(0, duration, len(resampled_data), endpoint=False)

            # Ensure the lengths match for plotting
        min_length_original = min(len(time_vector), len(joint_data))
        min_length_resampled = min(len(t2), len(resampled_data))

            # Plotting the original and resampled data over time
        plt.plot(time_vector[:min_length_original], joint_data[:min_length_original], linestyle='--', marker = 'o', color=colors[JOINT_IX], alpha=0.9, label=f'Original {joint_names[JOINT_IX]}')
        plt.plot(t2, resampled_data, linestyle='-', color = 'k', alpha=0.2, marker='o', label=f'Resampled {joint_names[JOINT_IX]}')


# Set plot labels and title
plt.xlabel('Time (s)')
plt.ylabel('Kinematic Angle (degrees)')  # Specific label for kinematic angle
plt.title('Original and Resampled Kinematic Angles for All Joints')
plt.legend()
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.xlim([0,.5])
plt.show()

# %%
