# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.io import loadmat
import math

# %%
emgdata = loadmat('J10_s10_i0_pref.mat')
rawdata = emgdata['emg_full_raw'] 
filterdata = emgdata['emg_full_fil']
# for key in emgdata:
#     if not key.startswith('__'):
#         print(f"\nKey: {key}")
#         print(f"Type: {type(emgdata[key])}")
#         print(f"Shape: {emgdata[key].shape}")

plt.rcParams['agg.path.chunksize'] = 10000
# %%
kindata = loadmat('J10_s20_i0_pref.mat')

joints = kindata['joints_raw_nonSeg']
num_joints = joints.shape[0]
cmap = plt.get_cmap('viridis')
colors = cmap(np.linspace(0, 1, num_joints))
markers = kindata['mk_raw_nonSeg']
markers = markers.reshape(7,1)
num_markers = markers.shape[0]

# %%
#calculate sampling rate
time_vector = kindata['t_kin'].flatten()
intervals = np.diff(time_vector)
SAMPLE_RATE = 1 / np.mean(intervals)
SAMPLE_RATE = round(SAMPLE_RATE)
TARGET_SAMPLE_RATE = 500

joint_names_1 = kindata['joints_names'].flatten()
joint_names = [name[0] for name in joint_names_1]
marker_names_1 = kindata['mk_names'].flatten()
marker_names = [name[0] for name in marker_names_1]
JOINT_IX = 1
MARKER_IX = 1

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
#low pass filtering the data -> markers
def apply_butterworth_filter(x, sample_rate=SAMPLE_RATE, cutoff=40.0, order=4):
    """
        apply 4th order butterworth low pass filter with cutoff at 40 Hz
        x = signal data being filtered
        sample_rate = samples per second of the data
        cutoff = cutoff frequency
        order = filter order
    """
    b, a = signal.butter(4, 40.0, btype='low', analog=False, fs=SAMPLE_RATE)
    butterworth = signal.filtfilt(b, a, x)
    return butterworth

fig, axs = plt.subplots(2, 1, figsize=(10, 12))
for MARKER_IX in range(num_markers):
    marker_data = markers[MARKER_IX][0]
    marker_x = marker_data[:,0]
    marker_y = marker_data[:,1]

            
    applied_butter_marker_x = apply_butterworth_filter(marker_x, SAMPLE_RATE)
    applied_butter_marker_y = apply_butterworth_filter(marker_y, SAMPLE_RATE)

    axs[0].plot(time_vector, applied_butter_marker_x-marker_x, alpha=0.7, marker= 'o', color = colors[MARKER_IX], markersize=3, label=marker_names[MARKER_IX])
    axs[1].plot(time_vector, applied_butter_marker_y-marker_y, alpha=0.7, marker= 'o', color = colors[MARKER_IX], markersize=3, label=marker_names[MARKER_IX])


axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Position X')
axs[0].set_title('Filtered Marker Position X Over Time')
axs[0].legend()
axs[0].set_xlim(0,5)
axs[0].set_ylim(-1,1)

axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Position Y')
axs[1].set_title('Filtered Marker Position Y Over Time')
axs[1].legend()
axs[1].set_xlim(0,5)
axs[0].set_ylim(-1,1)


plt.tight_layout()
plt.show()

# %%
#extracting the 2d arrays
for i, element in enumerate(joints):
    inner_array = element[0]
    print(inner_array.shape)


# %%
#resampling and filtering the data
fig, axs = plt.subplots(2, 1, figsize=(10, 12))


UPSAMPLING = 5
DOWNSAMPLING = 2

for MARKER_IX in range(num_markers):
    marker_data = markers[MARKER_IX][0]
    marker_x = marker_data[:, 0]
    marker_y = marker_data[:, 1]


    filtered_data_x = apply_butterworth_filter(marker_x, SAMPLE_RATE)
    resampled_data_x = signal.resample_poly(filtered_data_x, up=UPSAMPLING, down= DOWNSAMPLING) 
    filtered_data_y = apply_butterworth_filter(marker_y, SAMPLE_RATE)
    resampled_data_y = signal.resample_poly(filtered_data_y, up=UPSAMPLING, down= DOWNSAMPLING) 


    duration_x = len(resampled_data_x) / TARGET_SAMPLE_RATE
    duration_y = len(resampled_data_y) / TARGET_SAMPLE_RATE

    t2_x = np.linspace(0, duration_x, len(resampled_data_x), endpoint=False)
    t2_y = np.linspace(0, duration_y, len(resampled_data_y), endpoint=False)

    axs[0].plot(t2_x, resampled_data_x, alpha=0.7, marker= 'o', color = colors[MARKER_IX], markersize=3, label=f' {marker_names[MARKER_IX] } ')
    #axs[0].plot(time_vector, marker_x, alpha=0.7, marker= 'o', color = colors[JOINT_IX], markersize=3, label=f' {marker_names[MARKER_IX] } ')
    #uncomment to plot overlaid before/after filtering and resampling

    axs[1].plot(t2_y, resampled_data_y, alpha=0.7, marker= 'o', color = colors[MARKER_IX], markersize=3, label=f' {marker_names[MARKER_IX] } ')
    #axs[1].plot(time_vector, marker_y, alpha=0.7, marker= 'o', color = colors[JOINT_IX], markersize=3, label=f' {marker_names[MARKER_IX] } ')


axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Amplitude')
axs[0].set_title('Filtered and Resampled Marker Position X Over Time')
axs[0].legend()
axs[0].set_xlim(0, 5)

axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Position Y')
axs[1].set_title('Filtered Marker and Resampled Position Y Over Time')
axs[1].legend()
axs[1].set_xlim(0, 5)

plt.tight_layout()
plt.show()


# %%
#savistsky-golay differentiation on one plot

fig, axs = plt.subplots(6, 1, figsize=(10, 8), sharex=True)
marker_names_1 = kindata['mk_names'].flatten()
marker_names = [name[0] for name in marker_names_1]

POLYORDER = 5
WINDOW_LENGTH = 27

for MARKER_IX in range(num_markers):
    marker_data = markers[MARKER_IX][0]
    marker_x = marker_data[:, 0]
    marker_y = marker_data[:, 1]

    filtered_data_x = apply_butterworth_filter(marker_x, SAMPLE_RATE)
    resampled_data_x = signal.resample_poly(filtered_data_x, up=UPSAMPLING, down= DOWNSAMPLING) 
    filtered_data_y = apply_butterworth_filter(marker_y, SAMPLE_RATE)
    resampled_data_y = signal.resample_poly(filtered_data_y, up=UPSAMPLING, down= DOWNSAMPLING) 

    duration_x = len(resampled_data_x) / TARGET_SAMPLE_RATE
    duration_y = len(resampled_data_y) / TARGET_SAMPLE_RATE

    t2_x = np.linspace(0, duration_x, len(resampled_data_x), endpoint=False)
    t2_y = np.linspace(0, duration_y, len(resampled_data_y), endpoint=False)

    min_length_resampled_x = min(len(t2_x), len(resampled_data_x))
    min_length_resampled_y = min(len(t2_y), len(resampled_data_y))


    if resampled_data_x.ndim == 1:
        resampled_data_x = resampled_data_x.reshape(-1, 1)
    if resampled_data_y.ndim == 1:
        resampled_data_y = resampled_data_y.reshape(-1, 1)


    sg_x = signal.savgol_filter(resampled_data_x[:,0], window_length=WINDOW_LENGTH, polyorder= POLYORDER)
    sg_y = signal.savgol_filter(resampled_data_y[:,0], window_length=WINDOW_LENGTH, polyorder= POLYORDER)

    angular_velocity_x = signal.savgol_filter(resampled_data_x[:, 0], window_length=WINDOW_LENGTH, polyorder= POLYORDER, deriv=1, delta=t2_x[1] - t2_x[0])
    angular_velocity_y = signal.savgol_filter(resampled_data_y[:, 0], window_length=WINDOW_LENGTH, polyorder= POLYORDER, deriv=1, delta=t2_y[1] - t2_y[0])

    angular_acceleration_x = signal.savgol_filter(resampled_data_x[:, 0], window_length=WINDOW_LENGTH, polyorder= POLYORDER, deriv=2, delta=t2_x[1] - t2_x[0])
    angular_acceleration_y = signal.savgol_filter(resampled_data_y[:, 0], window_length=WINDOW_LENGTH, polyorder= POLYORDER, deriv=2, delta=t2_y[1] - t2_y[0])


    axs[0].plot(t2_x[:min_length_resampled_x], sg_x[:min_length_resampled_x], alpha=0.2, marker='o', markersize=1, label=f'{marker_names[MARKER_IX]} ')
    axs[1].plot(t2_x[:min_length_resampled_x], angular_velocity_x[:min_length_resampled_x], alpha=0.2, marker='o', markersize = 1, label=f'{marker_names[MARKER_IX]} ')
    axs[2].plot(t2_x[:min_length_resampled_x], angular_acceleration_x[:min_length_resampled_x], alpha=0.2, marker='o', markersize=1, label=f'{marker_names[MARKER_IX]} ')
    axs[3].plot(t2_y[:min_length_resampled_y], sg_y[:min_length_resampled_y], alpha=0.2, marker='o', markersize=1, label=f'{marker_names[MARKER_IX]} ')
    axs[4].plot(t2_y[:min_length_resampled_y], angular_velocity_y[:min_length_resampled_y], alpha=0.2, marker='o', markersize=1, label=f'{marker_names[MARKER_IX]} ')
    axs[5].plot(t2_y[:min_length_resampled_y], angular_acceleration_y[:min_length_resampled_y], alpha=0.2, marker='o', markersize=1, label=f'{marker_names[MARKER_IX]} ')

# Set plot labels and title
    axs[0].set_ylabel('Degrees')  
    axs[1].set_ylabel('Degrees/s')  
    axs[2].set_ylabel('Degrees/s^2')  
    axs[3].set_ylabel('Degrees')  
    axs[4].set_ylabel('Degrees/s')  
    axs[5].set_ylabel('Degrees/s^2')  

    axs[0].set_title('Angular Position X')
    axs[1].set_title('Angular Velocity Y')
    axs[2].set_title('Angular Acceleration Z')
    axs[3].set_title('Angular Position X')
    axs[4].set_title('Angular Velocity Y')
    axs[5].set_title('Angular Acceleration Z')

axs[1].set_ylim(-2000, 2000)
axs[2].set_ylim(-100000,100000)

plt.legend()

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.xlim([0, 5])
plt.tight_layout()
plt.show()


# %%
marker_names_1 = kindata['mk_names'].flatten()
marker_names = [name[0] for name in marker_names_1]

POLYORDER = 5
WINDOW_LENGTH = 27

for MARKER_IX in range(num_markers):
    fig, axs = plt.subplots(6, 1, figsize=(10, 8), sharex=True)
    marker_data = markers[MARKER_IX][0]
    marker_x = marker_data[:, 0]
    marker_y = marker_data[:, 1]

    filtered_data_x = apply_butterworth_filter(marker_x, SAMPLE_RATE)
    resampled_data_x = signal.resample_poly(filtered_data_x, up=UPSAMPLING, down= DOWNSAMPLING) 
    filtered_data_y = apply_butterworth_filter(marker_y, SAMPLE_RATE)
    resampled_data_y = signal.resample_poly(filtered_data_y, up=UPSAMPLING, down= DOWNSAMPLING) 

    duration_x = len(resampled_data_x) / TARGET_SAMPLE_RATE
    duration_y = len(resampled_data_y) / TARGET_SAMPLE_RATE

    t2_x = np.linspace(0, duration_x, len(resampled_data_x), endpoint=False)
    t2_y = np.linspace(0, duration_y, len(resampled_data_y), endpoint=False)

    min_length_resampled_x = min(len(t2_x), len(resampled_data_x))
    min_length_resampled_y = min(len(t2_y), len(resampled_data_y))


    if resampled_data_x.ndim == 1:
        resampled_data_x = resampled_data_x.reshape(-1, 1)
    if resampled_data_y.ndim == 1:
        resampled_data_y = resampled_data_y.reshape(-1, 1)


    sg_x = signal.savgol_filter(resampled_data_x[:,0], window_length=WINDOW_LENGTH, polyorder= POLYORDER)
    sg_y = signal.savgol_filter(resampled_data_y[:,0], window_length=WINDOW_LENGTH, polyorder= POLYORDER)

    angular_velocity_x = signal.savgol_filter(resampled_data_x[:, 0], window_length=WINDOW_LENGTH, polyorder= POLYORDER, deriv=1, delta=t2_x[1] - t2_x[0])
    angular_velocity_y = signal.savgol_filter(resampled_data_y[:, 0], window_length=WINDOW_LENGTH, polyorder= POLYORDER, deriv=1, delta=t2_y[1] - t2_y[0])

    angular_acceleration_x = signal.savgol_filter(resampled_data_x[:, 0], window_length=WINDOW_LENGTH, polyorder= POLYORDER, deriv=2, delta=t2_x[1] - t2_x[0])
    angular_acceleration_y = signal.savgol_filter(resampled_data_y[:, 0], window_length=WINDOW_LENGTH, polyorder= POLYORDER, deriv=2, delta=t2_y[1] - t2_y[0])


    axs[0].plot(t2_x[:min_length_resampled_x], sg_x[:min_length_resampled_x], alpha=0.2, marker='o', markersize=1, label=f'{marker_names[MARKER_IX]} ')
    axs[1].plot(t2_x[:min_length_resampled_x], angular_velocity_x[:min_length_resampled_x], alpha=0.2, marker='o', markersize = 1, label=f'{marker_names[MARKER_IX]} ')
    axs[2].plot(t2_x[:min_length_resampled_x], angular_acceleration_x[:min_length_resampled_x], alpha=0.2, marker='o', markersize=1, label=f'{marker_names[MARKER_IX]} ')
    axs[3].plot(t2_y[:min_length_resampled_y], sg_y[:min_length_resampled_y], alpha=0.2, marker='o', markersize=1, label=f'{marker_names[MARKER_IX]} ')
    axs[4].plot(t2_y[:min_length_resampled_y], angular_velocity_y[:min_length_resampled_y], alpha=0.2, marker='o', markersize=1, label=f'{marker_names[MARKER_IX]} ')
    axs[5].plot(t2_y[:min_length_resampled_y], angular_acceleration_y[:min_length_resampled_y], alpha=0.2, marker='o', markersize=1, label=f'{marker_names[MARKER_IX]} ')

# Set plot labels and title
    axs[0].set_ylabel('Degrees')  
    axs[1].set_ylabel('Degrees/s')  
    axs[2].set_ylabel('Degrees/s^2')  
    axs[3].set_ylabel('Degrees')  
    axs[4].set_ylabel('Degrees/s')  
    axs[5].set_ylabel('Degrees/s^2')  

    axs[0].set_title(f'Angular Position X for {marker_names[MARKER_IX]}')    
    axs[1].set_title('Angular Velocity Y')
    axs[2].set_title('Angular Acceleration Z')
    axs[3].set_title('Angular Position X')
    axs[4].set_title('Angular Velocity Y')
    axs[5].set_title('Angular Acceleration Z')

axs[1].set_ylim(-2000, 2000)
axs[2].set_ylim(-100000,100000)

fig.tight_layout()
for MARKER_IX in range(num_markers):

    for ax in axs:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
plt.show()




# %%
#plotting savistsky-golay differentiation for each joint on indivisual plots
joint_angle_names_1 = kindata['joints_names'].flatten()
joint_angle_names = [name[0] for name in joint_angle_names_1]
marker_names_1 = kindata['mk_names'].flatten()
marker_names = [name[0] for name in marker_names_1]

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

    axs[0].set_title('Angular Position X')
    axs[1].set_title('Angular Velocity Y')
    axs[2].set_title('Angular Acceleration Z')
    

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
print(dataframe_multi)

#dataframe = dataframe.sort_values(by='Time')
#dataframe.set_index('Time', inplace=True)

# 
# %%
#making a pandas dataframe for markers
dataframe_markers = pd.DataFrame(columns=['Marker', 'Time', 'Position', 'Velocity', 'Acceleration'])
rows = []

for MARKER_IX in range(num_markers):
    marker_data = markers[MARKER_IX][0]
    min_length_original = min(len(time_vector), len(marker_data))

    # Assuming marker_x and marker_y are defined as parts of marker_data
    marker_x = marker_data[:, 0]
    marker_y = marker_data[:, 1]

    filtered_data_x = apply_butterworth_filter(marker_x, SAMPLE_RATE)
    resampled_data_x = signal.resample_poly(filtered_data_x, up=UPSAMPLING, down=DOWNSAMPLING)
    filtered_data_y = apply_butterworth_filter(marker_y, SAMPLE_RATE)
    resampled_data_y = signal.resample_poly(filtered_data_y, up=UPSAMPLING, down=DOWNSAMPLING)

    duration_x = len(resampled_data_x) / TARGET_SAMPLE_RATE
    duration_y = len(resampled_data_y) / TARGET_SAMPLE_RATE

    t2_x = np.linspace(0, duration_x, len(resampled_data_x), endpoint=False)
    t2_y = np.linspace(0, duration_y, len(resampled_data_y), endpoint=False)

    min_length_resampled_x = min(len(t2_x), len(resampled_data_x))
    min_length_resampled_y = min(len(t2_y), len(resampled_data_y))

    if resampled_data_x.ndim == 1:
        resampled_data_x = resampled_data_x.reshape(-1, 1)
    if resampled_data_y.ndim == 1:
        resampled_data_y = resampled_data_y.reshape(-1, 1)

    sg_x = signal.savgol_filter(resampled_data_x[:, 0], window_length=WINDOW_LENGTH, polyorder=POLYORDER)
    sg_y = signal.savgol_filter(resampled_data_y[:, 0], window_length=WINDOW_LENGTH, polyorder=POLYORDER)

    angular_velocity_x = signal.savgol_filter(resampled_data_x[:, 0], window_length=WINDOW_LENGTH, polyorder=POLYORDER, deriv=1, delta=t2_x[1] - t2_x[0])
    angular_velocity_y = signal.savgol_filter(resampled_data_y[:, 0], window_length=WINDOW_LENGTH, polyorder=POLYORDER, deriv=1, delta=t2_y[1] - t2_y[0])

    angular_acceleration_x = signal.savgol_filter(resampled_data_x[:, 0], window_length=WINDOW_LENGTH, polyorder=POLYORDER, deriv=2, delta=t2_x[1] - t2_x[0])
    angular_acceleration_y = signal.savgol_filter(resampled_data_y[:, 0], window_length=WINDOW_LENGTH, polyorder=POLYORDER, deriv=2, delta=t2_y[1] - t2_y[0])


for MARKER_IX in range(num_markers):
    marker_name = marker_names[MARKER_IX]

    for i in range(min_length_resampled_x):
        joint_name = joint_names_1[JOINT_IX][0]
        row = {
            'Time': t2_x[i],
            'Marker': f'{marker_name}_x',
            'Position': sg_x[i],
            'Velocity': angular_velocity_x[i],
            'Acceleration': angular_acceleration_x[i]
        }
        rows.append(row)

    for i in range(min_length_resampled_y):
        joint_name = joint_names_1[JOINT_IX][0]
        row = {
            'Time': t2_y[i],
            'Marker': f'{marker_name}_y',
            'Position': sg_y[i],
            'Velocity': angular_velocity_y[i],
            'Acceleration': angular_acceleration_y[i]
        }
        rows.append(row)

dataframe_markers = pd.DataFrame(rows)

# Pivot the DataFrame to create a multi-index DataFrame
dataframe_multi_markers = dataframe_markers.pivot_table(index='Time', columns='Marker', values=['Position', 'Velocity', 'Acceleration'])

dataframe_multi_markers.reset_index(inplace=True)
print(dataframe_multi_markers)

# %%
#toe kinematics position trace over time, using  change in slope as a metric
#wrong
toevelocity = dataframe_multi_markers['Velocity']['toe_y']
time = dataframe_multi['Time']

# %%
#toe kinematics over time with peaks and troughs identified
plt.figure(figsize=(12,6))
peaks, _ = signal.find_peaks(toevelocity, prominence=300)
troughs, _ = signal.find_peaks(-toevelocity, prominence=300)


plt.plot (time, toevelocity, color='black')

#plt.plot (time[peaks], toeposition[peaks], 'ko')
#plt.plot (time[troughs], toeposition[troughs], 'ko')

for i in range(len(troughs)): #less than the length of peaks
    plt.plot(time[peaks[i]:troughs[i]+1], toevelocity[peaks[i]:troughs[i]+1], color='red')


plt.xlabel('Time (s)')
plt.ylabel('Velocities') 
plt.title('Toe velocities over time')
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.xlim([0, 20])
plt.axhline(y=50, color='blue', linestyle='--', linewidth=1)
plt.axhline(y=-20, color='blue', linestyle='--', linewidth=1)

#plt.ylim([-100,100])
plt.xlim([0,10])
plt.show()

# %%
#finding the swing/stance phase by looking within thresholds

peaks, _ = signal.find_peaks(toevelocity, prominence=300)
troughs, _ = signal.find_peaks(-toevelocity, prominence=300)

pos_crossing_thresh = 80
neg_crossing_thresh = -50
        
def thresh_crossings(data, time, threshold, start_indices, end_indices):
    zero_crossings = []
    for start, end in  zip(start_indices, end_indices):
        for i in range(start, end):
            if (data[i] < threshold and data[i + 1] > threshold) or (data[i] > threshold and data[i + 1] < threshold):
                crossings = time[i] + (time[i + 1] - time[i]) * ((threshold - data[i]) / (data[i + 1] - data[i]))
                zero_crossings.append(crossings)
    return zero_crossings


start_crossings = thresh_crossings(toevelocity, time, neg_crossing_thresh, troughs[:-1], peaks[1:])
end_crossings = thresh_crossings(toevelocity, time, pos_crossing_thresh, troughs[:-1], peaks[1:])

plt.plot(time, toevelocity)
plt.scatter(start_crossings, [neg_crossing_thresh] * len(start_crossings), color='black', label='Start Crossings')
plt.scatter(end_crossings, [pos_crossing_thresh] * len(end_crossings), color='black', label='End Crossings')
plt.axhline(y=-50, color='red', linestyle='--', linewidth=1, label='Threshold -70')
plt.axhline(y=80, color='red', linestyle='--', linewidth=1, label='Threshold -70')

plt.xlim([0,10])
plt.ylim([-500,500])

print(f'Start Crossings: {start_crossings}')
print(f'End Crossings: {end_crossings}')

# %%
#creating second dataframe: we want to store where foot lands on the ground, use step number as index for this dataframe (row is step_id, stance_starttime, stance_endtime, swing_endtime) can do swing_startime but it's the same as stance_endtime
step_data = {'step_id': [], 'stance_starttime': [], 'stance_endtime': [], 'swing_starttime': [], 'swing_endtime': []}

for i in range(len(troughs)-1):
    step_data['step_id'].append(i)
    step_data['stance_starttime'].append([start_crossings[i]])
    step_data['stance_endtime'].append([end_crossings[i]])
    step_data['swing_starttime'].append([end_crossings[i]])
    step_data['swing_endtime'].append(start_crossings[i + 1])

dataframe_step = pd.DataFrame(step_data)
dataframe_step.set_index('step_id', inplace=True)

print(dataframe_step)

# %%
