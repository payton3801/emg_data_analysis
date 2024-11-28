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

            plt.plot(t2, resampled_data, alpha=0.7, marker= 'o', color = colors[JOINT_IX], markersize=3, label=f' {joint_names[JOINT_IX] } Col {col+1}')

    else:

        filtered_data = apply_butterworth_filter(joint_data[:, 0])
        resampled_data = signal.resample_poly(filtered_data, up = UPSAMPLING, down = DOWNSAMPLING)


        duration = len(resampled_data) / TARGET_SAMPLE_RATE
        t2 = np.linspace(0, duration, len(resampled_data), endpoint=False)

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

            filtered_data = apply_butterworth_filter(joint_data[:, col])
            resampled_data = signal.resample_poly(filtered_data, up=UPSAMPLING, down=DOWNSAMPLING)

            duration = len(resampled_data) / TARGET_SAMPLE_RATE
            t2 = np.linspace(0, duration, len(resampled_data), endpoint=False)

            min_length_resampled = min(len(t2), len(resampled_data))

            plt.plot(t2[:min_length_resampled], resampled_data[:min_length_resampled], linestyle='-', color='k', alpha=0.2, marker='o', label=f'Resampled {joint_names[JOINT_IX]} Col {col+1}')
    else:
        # Resample the data
        filtered_data = apply_butterworth_filter(joint_data[:, 0])
        resampled_data = signal.resample_poly(filtered_data, up=UPSAMPLING, down=DOWNSAMPLING)

        duration = len(resampled_data) / TARGET_SAMPLE_RATE
        t2 = np.linspace(0, duration, len(resampled_data), endpoint=False)

        min_length_resampled = min(len(t2), len(resampled_data))

        plt.plot(t2[:min_length_resampled], resampled_data[:min_length_resampled], linestyle='-', color='k', alpha=0.2, marker='o', label=f'Resampled {joint_names[JOINT_IX]}')

# Set plot labels and title
plt.xlabel('Time (s)')
plt.ylabel('Kinematic Angle (degrees)') 
plt.title('Original and Resampled Kinematic Angles for All Joints')
plt.legend()
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.xlim([0, .5])
plt.show()

# %%
#savistsky-golay differentiation on one plot

fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
joint_names_1 = kindata['joints_names'].flatten()
joint_names = [name[0] for name in joint_names_1]

POLYORDER = 5
WINDOW_LENGTH = 27

for JOINT_IX in range(num_joints):
    # Access the inner 2D array
    joint_data = joints[JOINT_IX][0]
    min_length_original = min(len(time_vector), len(joint_data))

    if joint_data.shape[1] == 2:
        for col in range(2):

            filtered_data = apply_butterworth_filter(joint_data[:, col])
            resampled_data = signal.resample_poly(filtered_data, up=UPSAMPLING, down=DOWNSAMPLING)

            duration = len(resampled_data) / TARGET_SAMPLE_RATE
            t2 = np.linspace(0, duration, len(resampled_data), endpoint=False)

            min_length_resampled = min(len(t2), len(resampled_data))

            if resampled_data.ndim == 1:
                resampled_data = resampled_data.reshape(-1, 1)


            sg = signal.savgol_filter(resampled_data[:,0], window_length=WINDOW_LENGTH, polyorder= POLYORDER)
            angular_velocity = signal.savgol_filter(resampled_data[:, 0],window_length=WINDOW_LENGTH, polyorder= POLYORDER, deriv=1, delta=t2[1] - t2[0])
            angular_acceleration = signal.savgol_filter(resampled_data[:, 0], window_length=WINDOW_LENGTH, polyorder= POLYORDER, deriv=2, delta=t2[1] - t2[0])
            
            axs[0].plot(t2[:min_length_resampled], sg[:min_length_resampled], alpha=0.2, marker='o', label=f'Resampled {joint_names[JOINT_IX]} Col {col+1}')
            axs[1].plot(t2[:min_length_resampled], angular_velocity[:min_length_resampled], alpha=0.2, marker='o', label=f'Resampled {joint_names[JOINT_IX]} Col {col+1}')
            axs[2].plot(t2[:min_length_resampled], angular_acceleration[:min_length_resampled], alpha=0.2, marker='o', label=f'Resampled {joint_names[JOINT_IX]} Col {col+1}')

   
    else:

        filtered_data = apply_butterworth_filter(joint_data[:, 0])
        resampled_data = signal.resample_poly(filtered_data, up=UPSAMPLING, down=DOWNSAMPLING)

        duration = len(resampled_data) / TARGET_SAMPLE_RATE
        t2 = np.linspace(0, duration, len(resampled_data), endpoint=False)

        min_length_resampled = min(len(t2), len(resampled_data))

        if resampled_data.ndim == 1:
            resampled_data = resampled_data.reshape(-1, 1)


        sg = signal.savgol_filter(resampled_data[:,0], window_length=WINDOW_LENGTH, polyorder= POLYORDER)
        angular_velocity = signal.savgol_filter(resampled_data[:, 0], window_length=WINDOW_LENGTH, polyorder= POLYORDER, deriv=1, delta=t2[1] - t2[0])
        angular_acceleration = signal.savgol_filter(resampled_data[:, 0], window_length=WINDOW_LENGTH, polyorder= POLYORDER, deriv=2, delta=t2[1] - t2[0])


        axs[0].plot(t2[:min_length_resampled], sg[:min_length_resampled], alpha=0.2, marker='o', label=f'{joint_names[JOINT_IX]} ')
        axs[1].plot(t2[:min_length_resampled], angular_velocity[:min_length_resampled], alpha=0.2, marker='o', label=f'{joint_names[JOINT_IX]} ')
        axs[2].plot(t2[:min_length_resampled], angular_acceleration[:min_length_resampled], alpha=0.2, marker='o', label=f'{joint_names[JOINT_IX]} ')

# Set plot labels and title
axs[0].set_ylabel('Degrees')  
axs[1].set_ylabel('Degrees/s')  
axs[2].set_ylabel('Degrees/s^2')  
axs[2].set_xlabel('Time(s)')

axs[0].set_title('Angular Position')
axs[1].set_title('Angular Velocity')
axs[2].set_title('Angular Acceleration')

axs[1].set_ylim(-2000, 2000)
axs[2].set_ylim(-100000,100000)

plt.legend()

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.xlim([0, 5])
plt.show()

# %%
#plotting savistsky-golay differentiation for each joint on indivisual plots
joint_names_1 = kindata['joints_names'].flatten()
joint_names = [name[0] for name in joint_names_1]

POLYORDER = 5
WINDOW_LENGTH = 27

for JOINT_IX in range(num_joints):
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    # Access the inner 2D array
    joint_data = joints[JOINT_IX][0]
    min_length_original = min(len(time_vector), len(joint_data))

    if joint_data.shape[1] == 2:
        for col in range(2):

            filtered_data = apply_butterworth_filter(joint_data[:, col])
            resampled_data = signal.resample_poly(filtered_data, up=UPSAMPLING, down=DOWNSAMPLING)

            duration = len(resampled_data) / TARGET_SAMPLE_RATE
            t2 = np.linspace(0, duration, len(resampled_data), endpoint=False)

            min_length_resampled = min(len(t2), len(resampled_data))

            if resampled_data.ndim == 1:
                resampled_data = resampled_data.reshape(-1, 1)


            sg = signal.savgol_filter(resampled_data[:,0], window_length=WINDOW_LENGTH, polyorder= POLYORDER)
            angular_velocity = signal.savgol_filter(resampled_data[:, 0],window_length=WINDOW_LENGTH, polyorder= POLYORDER, deriv=1, delta=t2[1] - t2[0])
            angular_acceleration = signal.savgol_filter(resampled_data[:, 0], window_length=WINDOW_LENGTH, polyorder= POLYORDER, deriv=2, delta=t2[1] - t2[0])
            
            axs[0].plot(t2[:min_length_resampled], sg[:min_length_resampled], alpha=0.2, marker='o', label=f' {joint_names[JOINT_IX]} ')
            axs[1].plot(t2[:min_length_resampled], angular_velocity[:min_length_resampled], alpha=0.2, marker='o', label=f' {joint_names[JOINT_IX]} ')
            axs[2].plot(t2[:min_length_resampled], angular_acceleration[:min_length_resampled], alpha=0.2, marker='o', label=f' {joint_names[JOINT_IX]} ')

   
    else:

        filtered_data = apply_butterworth_filter(joint_data[:, 0])
        resampled_data = signal.resample_poly(filtered_data, up=UPSAMPLING, down=DOWNSAMPLING)

        duration = len(resampled_data) / TARGET_SAMPLE_RATE
        t2 = np.linspace(0, duration, len(resampled_data), endpoint=False)

        min_length_resampled = min(len(t2), len(resampled_data))

        if resampled_data.ndim == 1:
            resampled_data = resampled_data.reshape(-1, 1)


        sg = signal.savgol_filter(resampled_data[:,0], window_length=WINDOW_LENGTH, polyorder= POLYORDER)
        angular_velocity = signal.savgol_filter(resampled_data[:, 0], window_length=WINDOW_LENGTH, polyorder= POLYORDER, deriv=1, delta=t2[1] - t2[0])
        angular_acceleration = signal.savgol_filter(resampled_data[:, 0], window_length=WINDOW_LENGTH, polyorder= POLYORDER, deriv=2, delta=t2[1] - t2[0])


        axs[0].plot(t2[:min_length_resampled], sg[:min_length_resampled], alpha=0.2, marker='o', label=f'Resampled {joint_names[JOINT_IX]}')
        axs[1].plot(t2[:min_length_resampled], angular_velocity[:min_length_resampled], alpha=0.2, marker='o', label=f' {joint_names[JOINT_IX]} ')
        axs[2].plot(t2[:min_length_resampled], angular_acceleration[:min_length_resampled], alpha=0.2, marker='o', label=f' {joint_names[JOINT_IX]} ')

# Set plot labels and title
    axs[0].set_ylabel('Degrees')  
    axs[1].set_ylabel('Degrees/s')  
    axs[2].set_ylabel('Degrees/s^2')  

    axs[0].set_title('Angular Position')
    axs[1].set_title('Angular Velocity')
    axs[2].set_title('Angular Acceleration')

    axs[1].set_ylim(-2000, 2000)
    axs[2].set_ylim(-100000,100000)

    plt.legend()

    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.xlim([0, 5])
    plt.show()

# %%
#making a pandas dataframe
dataframe = pd.DataFrame(columns = ['Joint', 'Time', 'Position', 'Velocity', 'Acceleration'])
rows = []

for JOINT_IX in range(num_joints):
    joint_data= joints[JOINT_IX][0]
    min_length_original = min(len(time_vector), len(joint_data))

    if joint_data.shape[1] == 2:
        for col in range(2):
            filtered_data = apply_butterworth_filter(joint_data[:, col])
            resampled_data = signal.resample_poly(filtered_data, up=UPSAMPLING, down=DOWNSAMPLING)
            duration = len(resampled_data) / TARGET_SAMPLE_RATE
            t2 = np.linspace(0, duration, len(resampled_data), endpoint=False)

            min_length_resampled = min(len(t2), len(resampled_data))

            sg = signal.savgol_filter(resampled_data, window_length=WINDOW_LENGTH, polyorder=POLYORDER)
            angular_velocity = signal.savgol_filter(resampled_data, window_length=WINDOW_LENGTH, polyorder=POLYORDER, deriv=1, delta=t2[1] - t2[0])
            angular_acceleration = signal.savgol_filter(resampled_data, window_length=WINDOW_LENGTH, polyorder=POLYORDER, deriv=2, delta=t2[1] - t2[0])

            for i in range(len(t2)):
                joint_name = joint_names_1[JOINT_IX][0]
                row = ({'Time': t2[i],'Joint': joint_name, 'Position': sg[i], 'Velocity': angular_velocity[i], 'Acceleration': angular_acceleration[i]})
                rows.append(row)

    else:
        filtered_data = apply_butterworth_filter(joint_data[:, 0])
        resampled_data = signal.resample_poly(filtered_data, up=UPSAMPLING, down=DOWNSAMPLING)
        duration = len(resampled_data) / TARGET_SAMPLE_RATE
        t2 = np.linspace(0, duration, len(resampled_data), endpoint=False)

        min_length_resampled = min(len(t2), len(resampled_data))
        sg = signal.savgol_filter(resampled_data, window_length=WINDOW_LENGTH, polyorder=POLYORDER)
        angular_velocity = signal.savgol_filter(resampled_data, window_length=WINDOW_LENGTH, polyorder=POLYORDER, deriv=1, delta=t2[1] - t2[0])
        angular_acceleration = signal.savgol_filter(resampled_data, window_length=WINDOW_LENGTH, polyorder=POLYORDER, deriv=2, delta=t2[1] - t2[0])

        for i in range(len(t2)):
            joint_name = joint_names_1[JOINT_IX][0]
            row = ({'Time': t2[i],'Joint': joint_name, 'Position': sg[i], 'Velocity': angular_velocity[i], 'Acceleration': angular_acceleration[i]})
            rows.append(row)

dataframe = pd.concat([dataframe, pd.DataFrame(rows)])

dataframe_multi = dataframe.pivot_table( index='Time', columns='Joint', values=['Position', 'Velocity', 'Acceleration'])
dataframe_multi.columns = pd.MultiIndex.from_tuples(dataframe_multi.columns)

dataframe_multi.reset_index(inplace=True)

#dataframe = dataframe.sort_values(by='Time')
#dataframe.set_index('Time', inplace=True)

# %%
#Adding stance/swing column
velocity_threshold = 10
acceleration_threshold = 10

stance_conditions = []
for joint in dataframe_multi['Velocity'].columns:
    # Calculate stance condition for each joint separately
    stance_condition_joint = (np.abs(dataframe_multi['Velocity'][joint]) < velocity_threshold) & (np.abs(dataframe_multi['Acceleration'][joint]) < acceleration_threshold)
    stance_conditions.append(stance_condition_joint)

# Combine the stance conditions correctly
combined_stance = np.all(stance_conditions, axis=0)

# Assign the combined stance condition to the 'Phase' column as a 1D array
dataframe_multi['Phase'] = np.where(combined_stance, 'Stance', 'Swing')

# Show the filtered DataFrame
print(dataframe_multi)


# %%
#toe kinematics position trace over time, with stance and swing roughly colored
toeposition= dataframe_multi['Position']['mtp']
ankleposition = dataframe_multi['Position']['ankle']
time = dataframe_multi['Time']

ratio = np.abs(np.diff(toeposition)/ np.diff(time))
threshold = np.percentile(ratio, 90)
swingposition = np.where(ratio > threshold)[0]

plt.plot(time, toeposition, color='black')

for i in range(1, len(swingposition)):
    plt.plot(time[swingposition[i-1]:swingposition[i]], toeposition[swingposition[i-1]:swingposition[i]], color='red')

plt.xlabel('Time (s)')
plt.ylabel('Toe positions (degrees)') 
plt.title('Toe positions over time')
plt.legend()
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.xlim([0, 5])
plt.ylim([130, 210])
plt.show()

# %%
#toe kinematics over time with peaks and troughs identified

peaks, _ = signal.find_peaks(toeposition, prominence=50)
troughs, _ = signal.find_peaks(-toeposition, prominence=50)


plt.plot (time, toeposition)

plt.plot (time[peaks], toeposition[peaks], 'ro')
plt.plot (time[troughs], toeposition[troughs], 'ro')


plt.xlabel('Time (s)')
plt.ylabel('Toe positions (degrees)') 
plt.title('Toe positions over time')
plt.legend()
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.xlim([0, 5])
plt.ylim([130, 210])
plt.show()
# %%
