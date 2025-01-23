# %% -- importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.io import loadmat
import math

# %% -- loading in data/ parameters/ names 
emgdata = loadmat('J10_s10_i0_pref.mat')
rawdata = emgdata['emg_full_raw'] 
filterdata = emgdata['emg_full_fil']
time_vector_emg = emgdata['t_emg']
intervals_emg = np.diff(time_vector_emg)
EMG_SAMPLE_RATE = 1/np.mean(intervals_emg) 
bandwidth = 2
notch_frequencies= [60, 120, 240, 300, 420]
channel_names = emgdata['emg_names'].flatten()
channel_names = emgdata['emg_names'][0]
channel_names = [str(name[0]) for name in channel_names]

kindata = loadmat('J10_s20_i0_pref.mat')

joints = kindata['joints_raw_nonSeg']
num_joints = joints.shape[0]
cmap = plt.get_cmap('viridis')
colors = cmap(np.linspace(0, 1, num_joints))
markers = kindata['mk_raw_nonSeg']
markers = markers.reshape(7,1)
num_markers = markers.shape[0]

time_vector_kin = kindata['t_kin'].flatten()
intervals_kin = np.diff(time_vector_kin)
SAMPLE_RATE = 1 / np.mean(intervals_kin)
SAMPLE_RATE = round(SAMPLE_RATE)
TARGET_SAMPLE_RATE = 500

joint_names_1 = kindata['joints_names'].flatten()
joint_names = [name[0] for name in joint_names_1]
marker_names_1 = kindata['mk_names'].flatten()
marker_names = [name[0] for name in marker_names_1]
JOINT_IX = 1
MARKER_IX = 1

NFFT = 50000 #Use sampling rate/NFFT = 0.1
NPERSEG = 50000
NOVERLAP = 10000 #Use .2 (NFFT) = NOVERLAP
CHAN_IX = 5
num_channels = rawdata.shape[1]

plt.rcParams['agg.path.chunksize'] = 10000

# %% -- defining various filtering functions to use later
def apply_notch_filter(x, notch_frequencies, bandwidth, sample_rate):
    """ 
        apply notch filter(s) to a designated channel
        notch_frequencies = desired frequencies to be filtered out (Hz)
        x = signal data being filtered
        bandwidth = bandwidth of notch filter (Hz)
        sample_rate = sampling rate of the signal (Hz)
    """

    Q = [freq / bandwidth for freq in notch_frequencies]  # Compute Q for each frequency
    for freq, q in zip(notch_frequencies, Q):
        w0 = freq/ (sample_rate/2) #to normalize the frequency, convert to a value btwn 0 and 1
        b, a = signal.iirnotch(w0, q, SAMPLE_RATE)
        x = signal.filtfilt(b, a, x)

    return x

def apply_butterworth_filter_emg(x):
    """
        apply 4th order butterworth high pass filter with cutoff at 65 Hz
        x = signal data being filtered
    """

    b, a = signal.butter(4, 65.0, btype='high', analog=False, fs=SAMPLE_RATE)
    butterworth = signal.filtfilt(b, a, x)
    return butterworth

def apply_butterworth_filter_kin(x, sample_rate=SAMPLE_RATE, cutoff=40.0, order=4):
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

# %% -- plotting joint angles over time
for JOINT_IX in range(num_joints):
    joint_data = joints[JOINT_IX][0]
    if joint_data.shape[1] == 2:
        for col in range(2):
            
            applied_butter = apply_butterworth_filter_kin(joint_data[:, col], SAMPLE_RATE)
            plt.plot(time_vector_kin, applied_butter, alpha=0.7, marker= 'o', color = colors[JOINT_IX], markersize=3, label=joint_names[JOINT_IX])

    else:

        applied_butter = apply_butterworth_filter_kin(joint_data[:, 0])
        plt.plot(time_vector_kin, applied_butter, alpha=0.7, marker= 'o', color = colors[JOINT_IX], markersize=3, label=joint_names[JOINT_IX])

plt.xlabel('Time (s)') 
plt.ylabel('Kinematic Angle (degrees)')  
plt.title('Kinematic Angle per Joint Over Time')
plt.xlim([0, 5])

plt.legend()
plt.show()

# %% -- plotting one plot of filtered marker position on x and y overtime
fig, axs = plt.subplots(2, 1, figsize=(10, 12))
for MARKER_IX in range(num_markers):
    marker_data = markers[MARKER_IX][0]
    marker_x = marker_data[:,0]
    marker_y = marker_data[:,1]
            
    applied_butter_marker_x = apply_butterworth_filter_kin(marker_x, SAMPLE_RATE)
    applied_butter_marker_y = apply_butterworth_filter_kin(marker_y, SAMPLE_RATE)

    axs[0].plot(time_vector_kin, applied_butter_marker_x-marker_x, alpha=0.7, marker= 'o', color = colors[MARKER_IX], markersize=3, label=marker_names[MARKER_IX])
    axs[1].plot(time_vector_kin, applied_butter_marker_y-marker_y, alpha=0.7, marker= 'o', color = colors[MARKER_IX], markersize=3, label=marker_names[MARKER_IX])

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

# %% -- plotting one plot of filtered and resampled marker positions
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

UPSAMPLING = 5
DOWNSAMPLING = 2

for MARKER_IX in range(num_markers):
    marker_data = markers[MARKER_IX][0]
    marker_x = marker_data[:, 0]
    marker_y = marker_data[:, 1]

    filtered_data_x = apply_butterworth_filter_kin(marker_x, SAMPLE_RATE)
    resampled_data_x = signal.resample_poly(filtered_data_x, up=UPSAMPLING, down= DOWNSAMPLING) 
    filtered_data_y = apply_butterworth_filter_kin(marker_y, SAMPLE_RATE)
    resampled_data_y = signal.resample_poly(filtered_data_y, up=UPSAMPLING, down= DOWNSAMPLING) 

    duration_x = len(resampled_data_x) / TARGET_SAMPLE_RATE
    duration_y = len(resampled_data_y) / TARGET_SAMPLE_RATE

    t2_marker_x = np.linspace(0, duration_x, len(resampled_data_x), endpoint=False)
    t2_marker_y = np.linspace(0, duration_y, len(resampled_data_y), endpoint=False)

    axs[0].plot(t2_marker_x, resampled_data_x, alpha=0.7, marker= 'o', color = colors[MARKER_IX], markersize=3, label=f' {marker_names[MARKER_IX] } ')
    #axs[0].plot(time_vector, marker_x, alpha=0.7, marker= 'o', color = colors[JOINT_IX], markersize=3, label=f' {marker_names[MARKER_IX] } ')
    #uncomment to plot overlaid before/after filtering and resampling

    axs[1].plot(t2_marker_y, resampled_data_y, alpha=0.7, marker= 'o', color = colors[MARKER_IX], markersize=3, label=f' {marker_names[MARKER_IX] } ')
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

# %% -- plotting savistsky-golay differentiation for each marker on individual plots
POLYORDER = 5
WINDOW_LENGTH = 27

for MARKER_IX in range(num_markers):
    fig, axs = plt.subplots(6, 1, figsize=(10, 8), sharex=True)
    marker_data = markers[MARKER_IX][0]
    marker_x = marker_data[:, 0]
    marker_y = marker_data[:, 1]

    filtered_data_x = apply_butterworth_filter_kin(marker_x, SAMPLE_RATE)
    resampled_data_x = signal.resample_poly(filtered_data_x, up=UPSAMPLING, down= DOWNSAMPLING) 
    filtered_data_y = apply_butterworth_filter_kin(marker_y, SAMPLE_RATE)
    resampled_data_y = signal.resample_poly(filtered_data_y, up=UPSAMPLING, down= DOWNSAMPLING) 

    duration_x = len(resampled_data_x) / TARGET_SAMPLE_RATE
    duration_y = len(resampled_data_y) / TARGET_SAMPLE_RATE

    t2_marker_x = np.linspace(0, duration_x, len(resampled_data_x), endpoint=False)
    t2_marker_y = np.linspace(0, duration_y, len(resampled_data_y), endpoint=False)

    min_length_resampled_x = min(len(t2_marker_x), len(resampled_data_x))
    min_length_resampled_y = min(len(t2_marker_y), len(resampled_data_y))

    if resampled_data_x.ndim == 1:
        resampled_data_x = resampled_data_x.reshape(-1, 1)
    if resampled_data_y.ndim == 1:
        resampled_data_y = resampled_data_y.reshape(-1, 1)


    sg_x = signal.savgol_filter(resampled_data_x[:,0], window_length=WINDOW_LENGTH, polyorder= POLYORDER)
    sg_y = signal.savgol_filter(resampled_data_y[:,0], window_length=WINDOW_LENGTH, polyorder= POLYORDER)

    angular_velocity_x = signal.savgol_filter(resampled_data_x[:, 0], window_length=WINDOW_LENGTH, polyorder= POLYORDER, deriv=1, delta=t2_marker_x[1] - t2_marker_x[0])
    angular_velocity_y = signal.savgol_filter(resampled_data_y[:, 0], window_length=WINDOW_LENGTH, polyorder= POLYORDER, deriv=1, delta=t2_marker_y[1] - t2_marker_y[0])

    angular_acceleration_x = signal.savgol_filter(resampled_data_x[:, 0], window_length=WINDOW_LENGTH, polyorder= POLYORDER, deriv=2, delta=t2_marker_x[1] - t2_marker_x[0])
    angular_acceleration_y = signal.savgol_filter(resampled_data_y[:, 0], window_length=WINDOW_LENGTH, polyorder= POLYORDER, deriv=2, delta=t2_marker_y[1] - t2_marker_y[0])


    axs[0].plot(t2_marker_x[:min_length_resampled_x], sg_x[:min_length_resampled_x], alpha=0.2, marker='o', markersize=1, label=f'{marker_names[MARKER_IX]} ')
    axs[1].plot(t2_marker_x[:min_length_resampled_x], angular_velocity_x[:min_length_resampled_x], alpha=0.2, marker='o', markersize = 1, label=f'{marker_names[MARKER_IX]} ')
    axs[2].plot(t2_marker_x[:min_length_resampled_x], angular_acceleration_x[:min_length_resampled_x], alpha=0.2, marker='o', markersize=1, label=f'{marker_names[MARKER_IX]} ')
    axs[3].plot(t2_marker_y[:min_length_resampled_y], sg_y[:min_length_resampled_y], alpha=0.2, marker='o', markersize=1, label=f'{marker_names[MARKER_IX]} ')
    axs[4].plot(t2_marker_y[:min_length_resampled_y], angular_velocity_y[:min_length_resampled_y], alpha=0.2, marker='o', markersize=1, label=f'{marker_names[MARKER_IX]} ')
    axs[5].plot(t2_marker_y[:min_length_resampled_y], angular_acceleration_y[:min_length_resampled_y], alpha=0.2, marker='o', markersize=1, label=f'{marker_names[MARKER_IX]} ')

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

# %% -- plotting savistsky-golay differentiation for each joint on individual plots
POLYORDER = 5
WINDOW_LENGTH = 27

for JOINT_IX in range(num_joints):
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    # Access the inner 2D array
    joint_data = joints[JOINT_IX][0]
    min_length_original = min(len(time_vector_kin), len(joint_data))

    if joint_data.shape[1] == 2:
        for col in range(2):

            filtered_data = apply_butterworth_filter_kin(joint_data[:, col])
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

        filtered_data = apply_butterworth_filter_kin(joint_data[:, 0])
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

# %% -- making a pandas dataframe for joint position, velocity, and acceleration
dataframe = pd.DataFrame(columns = ['Joint', 'Time', 'Position', 'Velocity', 'Acceleration'])
rows = []

for JOINT_IX in range(num_joints):
    joint_data= joints[JOINT_IX][0]
    min_length_original = min(len(time_vector_kin), len(joint_data))

    if joint_data.shape[1] == 2:
        for col in range(2):
            filtered_data = apply_butterworth_filter_kin(joint_data[:, col])
            resampled_data = signal.resample_poly(filtered_data, up=UPSAMPLING, down=DOWNSAMPLING)
            duration = len(resampled_data) / TARGET_SAMPLE_RATE
            t_joint = np.linspace(0, duration, len(resampled_data), endpoint=False)

            min_length_resampled = min(len(t_joint), len(resampled_data))

            sg = signal.savgol_filter(resampled_data, window_length=WINDOW_LENGTH, polyorder=POLYORDER)
            angular_velocity = signal.savgol_filter(resampled_data, window_length=WINDOW_LENGTH, polyorder=POLYORDER, deriv=1, delta=t_joint[1] - t_joint[0])
            angular_acceleration = signal.savgol_filter(resampled_data, window_length=WINDOW_LENGTH, polyorder=POLYORDER, deriv=2, delta=t_joint[1] - t_joint[0])

            for i in range(len(t_joint)):
                joint_name = joint_names_1[JOINT_IX][0]
                row = ({'Time': t_joint[i],'Joint': joint_name, 'Position': sg[i], 'Velocity': angular_velocity[i], 'Acceleration': angular_acceleration[i]})
                rows.append(row)

    else:
        filtered_data = apply_butterworth_filter_kin(joint_data[:, 0])
        resampled_data = signal.resample_poly(filtered_data, up=UPSAMPLING, down=DOWNSAMPLING)
        duration = len(resampled_data) / TARGET_SAMPLE_RATE
        t_joint = np.linspace(0, duration, len(resampled_data), endpoint=False)

        min_length_resampled = min(len(t_joint), len(resampled_data))
        sg = signal.savgol_filter(resampled_data, window_length=WINDOW_LENGTH, polyorder=POLYORDER)
        angular_velocity = signal.savgol_filter(resampled_data, window_length=WINDOW_LENGTH, polyorder=POLYORDER, deriv=1, delta=t_joint[1] - t_joint[0])
        angular_acceleration = signal.savgol_filter(resampled_data, window_length=WINDOW_LENGTH, polyorder=POLYORDER, deriv=2, delta=t_joint[1] - t_joint[0])

        for i in range(len(t2)):
            joint_name = joint_names_1[JOINT_IX][0]
            row = ({'Time': t_joint[i],'Joint': joint_name, 'Position': sg[i], 'Velocity': angular_velocity[i], 'Acceleration': angular_acceleration[i]})
            rows.append(row)

dataframe = pd.concat([dataframe, pd.DataFrame(rows)])

dataframe_joint = dataframe.pivot_table( index='Time', columns='Joint', values=['Position', 'Velocity', 'Acceleration'])
dataframe_joint.columns = pd.MultiIndex.from_tuples(dataframe_joint.columns)

dataframe_joint.reset_index(inplace=True)
print(dataframe_joint)

#dataframe = dataframe.sort_values(by='Time')
#dataframe.set_index('Time', inplace=True)

# %% -- making a pandas dataframe for markers velocity, position, and acceleration
dataframe_markers = pd.DataFrame(columns=['Marker', 'Time', 'Position', 'Velocity', 'Acceleration'])
rows = []

for MARKER_IX in range(num_markers):
    marker_name = marker_names[MARKER_IX]

    for i in range(min_length_resampled_x):
        joint_name = joint_names_1[JOINT_IX][0]
        row = {
            'Time': t2_marker_x[i],
            'Marker': f'{marker_name}_x',
            'Position': sg_x[i],
            'Velocity': angular_velocity_x[i],
            'Acceleration': angular_acceleration_x[i]
        }
        rows.append(row)

    for i in range(min_length_resampled_y):
        joint_name = joint_names_1[JOINT_IX][0]
        row = {
            'Time': t2_marker_y[i],
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

# %% -- defining toe velocity and time from dataframe
toevelocity = dataframe_multi_markers['Velocity']['toe_y']
time = dataframe_joint['Time']

# %% -- full scale plot of toevelocities, area around stance identified
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

# %% -- plotting the toevelocities to map out threshold crossing around each step
mindistance = .3
mindistance_samples = int(mindistance * SAMPLE_RATE)

peaks, _ = signal.find_peaks(toevelocity, prominence=300, distance=mindistance_samples)
troughs, _ = signal.find_peaks(-toevelocity, prominence=300, distance=mindistance_samples)

pos_crossing_thresh = 80
neg_crossing_thresh = -50

def thresh_crossings(data, time, threshold, start_indices, end_indices, min_distance, after_trough=True):
    zero_crossings = []
    last_crossing_time = -np.inf  # Initialize to negative infinity
    for start, end in zip(start_indices, end_indices):
        after_trough = False
        for i in range(start, end):
            if data[i] == min(data[start:end+1]): 
                after_trough = True # Check for trough
            if after_trough and ((data[i] < threshold and data[i + 1] > threshold) or (data[i] > threshold and data[i + 1] < threshold)):
                crossing_time = time[i] + (time[i + 1] - time[i]) * ((threshold - data[i]) / (data[i + 1] - data[i]))
                if crossing_time - last_crossing_time >= min_distance:
                    zero_crossings.append(crossing_time)
                    last_crossing_time = crossing_time
    return zero_crossings

start_crossings = thresh_crossings(toevelocity, time, neg_crossing_thresh, troughs[:-1], peaks[1:], mindistance, after_trough=True)
end_crossings = thresh_crossings(toevelocity, time, pos_crossing_thresh, troughs[:-1], peaks[1:], mindistance, after_trough=False)

plt.plot(time, toevelocity)
plt.scatter(start_crossings, [neg_crossing_thresh] * len(start_crossings), color='black', label='Start Crossings')
plt.scatter(end_crossings, [pos_crossing_thresh] * len(end_crossings), color='black', label='End Crossings')
plt.axhline(y=-50, color='red', linestyle='--', linewidth=1, label='Threshold -70')
plt.axhline(y=80, color='red', linestyle='--', linewidth=1, label='Threshold -70')

plt.xlim([70, 72])
plt.ylim([-500, 500])

print(f'Start Crossings: {start_crossings}')
print(f'End Crossings: {end_crossings}')

# %% -- #making the stance duration histogram
stance_duration= [end-start for end, start in zip(end_crossings, start_crossings)]
plt.hist(stance_duration, bins=100)
plt.xlabel('Stance Times (in s)')
plt.ylabel('Step count')
plt.title('Stance Times for Each Step')
plt.show()

# %% -- making the swing duration histogram
swing_duration= [start-end for start, end in zip(start_crossings[1:], end_crossings[:-1])]
plt.hist(swing_duration, bins=100)
plt.xlabel('Swing times (in s)')
plt.ylabel('Step Count')
plt.title('Swing Times for Each Step')
plt.show()

# %% -- creating swing/stance dataframe_step
step_data = {'step_id': [], 'stance_starttime': [], 'stance_endtime': [], 'swing_starttime': [], 'swing_endtime': [], 'stance_duration': []}

for i in range(len(troughs) - 1):
    step_data['step_id'].append(i)
    step_data['stance_starttime'].append(start_crossings[i])
    step_data['stance_endtime'].append(end_crossings[i])
    step_data['swing_starttime'].append(end_crossings[i])
    stance_duration_for_dataframe = end_crossings[i] - start_crossings[i]
    step_data['stance_duration'].append(stance_duration_for_dataframe)
    if i + 1 < len(start_crossings):
        step_data['swing_endtime'].append(start_crossings[i + 1])
    else:
        step_data['swing_endtime'].append(None)  # Handle the last element case

dataframe_step = pd.DataFrame(step_data)

dataframe_step['stance_starttime'] = pd.to_timedelta(dataframe_step['stance_starttime'], unit='s')
dataframe_step['stance_endtime'] = pd.to_timedelta(dataframe_step['stance_endtime'], unit='s')
dataframe_step['swing_starttime'] = pd.to_timedelta(dataframe_step['swing_starttime'], unit='s')
dataframe_step['swing_endtime'] = pd.to_timedelta(dataframe_step['swing_endtime'], unit='s')
dataframe_step['stance_duration'] = pd.to_timedelta(dataframe_step['stance_duration'], unit='s')

print(dataframe_step)

# %% -- preprocessing the data again and making dataframe_emg
finalized_data = []
for CHAN_IX in range(num_channels):
    #if CHAN_IX == 6:
        #continue
    applied_notch_emg = apply_notch_filter(rawdata[:, CHAN_IX], notch_frequencies, bandwidth, SAMPLE_RATE)
    applied_butter_emg = apply_butterworth_filter_emg(applied_notch_emg)
    rectifieddata_emg = np.abs(applied_butter)
    
    # Resample the data
    EMG_SAMPLE_RATE = round(EMG_SAMPLE_RATE)
    TARGET_SAMPLE_RATE = 500
    DOWNSAMPLING_EMG = EMG_SAMPLE_RATE // TARGET_SAMPLE_RATE
    resampled_data_emg = signal.resample_poly(rectifieddata_emg, down=DOWNSAMPLING, up=1)
    quartiled_data_999 = np.quantile(resampled_data, .999)
    clipped_data_emg = np.clip(resampled_data, a_max=quartiled_data_999, a_min=None)
    clipped_data_emg = np.abs(clipped_data_emg)

    # Quartile clipping the data
    quartiled_data_95_emg = np.quantile(clipped_data_emg, .95)
    duration_emg = len(resampled_data) / TARGET_SAMPLE_RATE
    t_emg = np.linspace(0, duration, len(resampled_data), endpoint=False)

    normalized_data = clipped_data_emg / quartiled_data_95_emg
    normalized_data = np.abs(normalized_data)

    for i in range(len(t_emg)):
        finalized_data.append({
            'Time': t_emg[i],
            'Channel': channel_names[CHAN_IX],
            'Finalized Data': normalized_data[i]
        })

dataframe_emg = pd.DataFrame(finalized_data)
dataframe_emg = dataframe_emg.pivot_table(index='Time', columns='Channel', values='Finalized Data')
dataframe_emg.columns = pd.MultiIndex.from_product([['EMG'], dataframe_emg.columns.tolist()])
print(dataframe_emg)

# %% -- concatenating the dataframes
if 'Time' not in dataframe_multi_markers.columns:
    dataframe_multi_markers.reset_index(inplace=True)
if 'Time' not in dataframe_emg.columns:
    dataframe_emg.reset_index(inplace=True)
if 'Time' not in dataframe_joint.columns:
    dataframe_joint.reset_index(inplace=True)

dataframe_multi_markers['Time'] = pd.to_timedelta(dataframe_multi_markers['Time'], unit='s')
dataframe_joint['Time'] = pd.to_timedelta(dataframe_joint['Time'], unit='s')
dataframe_emg['Time'] = pd.to_timedelta(dataframe_emg['Time'], unit='s')

# Set 'Time' as the index for each dataframe
dataframe_multi_markers.set_index('Time', inplace=True)
dataframe_emg.set_index('Time', inplace=True)
dataframe_joint.set_index('Time', inplace=True)

# Concatenate the dataframes
dataframe_all = pd.concat([dataframe_multi_markers, dataframe_emg, dataframe_joint], axis=1)
print(dataframe_all)
# %% -- making the time locked averaging

window_size = 2
bins = np.arange(-window_size, window_size, .01)

fig, axs = plt.subplots(3, 4, figsize=(15, 10))
axs = axs.flatten()

for idx, channel in enumerate(channel_names):
    psth_counts = np.zeros(len(bins) - 1)
    num_trials = len(dataframe_step)

    for i, row in dataframe_step.iterrows():
        swing_onset = row['swing_starttime']
        stance_duration = row['stance_duration']

        start_window = swing_onset - pd.Timedelta(seconds=window_size)
        end_window = swing_onset + pd.Timedelta(seconds=window_size)

        dataframe_emg.index = pd.to_timedelta(dataframe_emg.index, unit='s')
        emg_window = dataframe_emg[(dataframe_emg.index >= start_window) & (dataframe_emg.index <= end_window)]

        #relative_times = (emg_window.index - swing_onset).total_seconds()
        #counts, _ = np.histogram(relative_times, bins=bins, weights=emg_window['EMG Marker Position', channel]) no histogram needed
        #axs[idx].plot(bins, counts (should be actual data), color='lightgray', alpha=0.5)  # Plot individual trials in lighter lines

        #psth_counts += counts

    avg_psth_counts = psth_counts / num_trials

    axs[idx].plot(bins[:-1], avg_psth_counts, color='black')  # Plot average counts vs. time bins
    axs[idx].set_xlabel("Time (s)")
    axs[idx].set_ylabel("Spike Count")
    axs[idx].set_title(f'PSTH of Muscle Activation for {channel}')

plt.tight_layout()
plt.show()



# %%

# concatanate this dataframe: [ emg | mk_pos | mk_vel | mk_acc | joint_ang_pos | joint_ang_vel | joint_ang_acc ]
# use arrays to make dataframes instead of dictionaries
# was indexing time by time, dont use histograms for "psths" and also they arent psths