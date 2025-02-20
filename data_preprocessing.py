# %% -- importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.io import loadmat
import seaborn as sns
import math
import plotly.graph_objs as go
from copy import deepcopy
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


# %% -- load mat file
mat_data = loadmat('J10_s10_i0_pref.mat')

# %% -- load raw emg

# -- load data, sampling rate, time vector
emg_raw = mat_data['emg_full_raw'] 
t_emg = mat_data['t_emg'].squeeze()
dt_emg = np.mean(np.diff(t_emg))
EMG_SAMPLE_RATE = np.round(1/dt_emg)

# -- load emg channel names, number of channels
emg_chan_names = mat_data['emg_names'].flatten()
emg_chan_names = mat_data['emg_names'][0]
emg_chan_names = [str(name[0]) for name in emg_chan_names]
num_channels = emg_raw.shape[1]
print(num_channels)

# %% -- load raw kinematics

# -- load joint angle data (multi-dimensional np array)
joints = mat_data['joints_raw_nonSeg']
# remove limbV_ln from joint angle data
joints = joints[:-1]

# compute number of joints (after removing limbV_ln)
n_joints = joints.shape[0]

# -- load kinematics marker position data
markers = mat_data['mk_raw_nonSeg']

# -- load time vector, compute sampling rate
t_kin = mat_data['t_kin'].flatten()
dt_kin = np.mean(np.diff(t_kin))
KIN_SAMPLE_RATE = np.round(1 / dt_kin)

# -- load joint angle names
joint_names = mat_data['joints_names'].flatten()
# trim limbV_ln from joint angle names
joint_names = joint_names[:-1]
joint_names = [name[0] for name in joint_names]

# -- load marker position names
marker_names = mat_data['mk_names'].flatten()
marker_names = [name[0] for name in marker_names]

# -- adjust marker names to break out x/y components
xy_marker_names = []
for marker_name in marker_names:
    xy_marker_names.append(f"{marker_name}_x")
    xy_marker_names.append(f"{marker_name}_y")

# -- reformat joint angle data
joint_pos_raw = np.zeros((t_kin.size, len(joint_names)))

for i, joint in enumerate(joints.squeeze()):    
    joint_pos_raw[:, i] = joint.squeeze()

# -- reformat marker position data
mk_pos_raw = np.zeros((t_kin.size, len(xy_marker_names)))
#mk_raw2 = []
for i, marker in enumerate(markers.squeeze()):
    #print(f"{2*i}:{2*i+2}")
    # have to index two rows at a time    
    mk_pos_raw[:,2*i:(2*i)+2] = marker
    # -- alternative approach to filling array
    #mk_raw2.append(marker)
#mk_raw2 = np.concatenate(mk_raw2, axis=1)

# %% -- define filtering functions

def _filter(b, a, x):
    """zero-phase digital filtering that handles nans"""
    nan_mask = np.isnan(x)
    is_nans = nan_mask.sum() > 0
    # temporarily replace nans with mean for filtering
    if is_nans:
        x[nan_mask] = np.nanmean(x)

    # apply filtering
    x = signal.filtfilt(b, a, x, axis=0)

    # put nans back where they were
    if is_nans:
        x[nan_mask] = np.nan

    return x

def apply_notch_filter(x, fs, notch_frequencies, bandwidth):
    """ 
        apply notch filter(s) to a designated channel
        notch_frequencies = desired frequencies to be filtered out (Hz)
        x = signal data being filtered
        bandwidth = bandwidth of notch filter (Hz)
        sample_rate = sampling rate of the signal (Hz)
    """

    Q = [freq / bandwidth for freq in notch_frequencies]  # Compute Q for each frequency
    for freq, q in zip(notch_frequencies, Q):
        b, a = signal.iirnotch(freq, q, fs)
        x = _filter(b, a, x)
    return x

def apply_butter_filter(x, fs, cutoff_freq=65, btype="high", filt_order=4):
    """
        apply 4th order butterworth high pass filter with cutoff at 65 Hz
        x = signal data being filtered
    """

    b, a = signal.butter(filt_order, cutoff_freq, btype=btype, analog=False, fs=fs)
    x = _filter(b, a, x)
    return x


def apply_savgol_filter(x, fs, delta, window_length=27, polyorder=5, deriv=1):
    x = signal.savgol_filter(x, window_length=window_length, polyorder=polyorder, deriv=deriv, delta=delta)
    return x


def compare_spectra(signal_1, signal_2, fs, dim, nfft=4000, nperseg=4000, noverlap=500):
    assert nfft == nperseg, 'nfft must match nperseg'

    def check_nans(x):
        nan_mask = np.isnan(x)
        is_nans = nan_mask.sum() > 0
        # temporarily replace nans with mean for computation
        if is_nans:
            print(f'nans found: {nan_mask.sum()}. replacing nans for computation')
            x[nan_mask] = np.nanmean(x)
        return x
    f, pxx_sig1 = signal.welch(check_nans(signal_1[:,dim]), fs=fs, nfft=nfft, nperseg=nperseg, noverlap=noverlap)
    f, pxx_sig2 = signal.welch(check_nans(signal_2[:,dim]), fs=fs, nfft=nfft, nperseg=nperseg, noverlap=noverlap)

    fig = plt.figure(figsize=(5,3), dpi=150)
    ax = fig.add_subplot(111)

    ax.loglog(f, pxx_sig1, color='k')
    ax.loglog(f, pxx_sig2, color='r')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def apply_data_resampling(x, upsampling=5, downsampling=2):
    x = signal.resample_poly(x, up=upsampling, down=downsampling)
    return x

# %% -- preprocess EMG 

# -- set filtering parameters
EMG_NOTCH_FREQUENCIES = [60, 120, 240, 300, 420] # Hz
BW_FREQ = 2 # Hz 
EMG_HP_CUTOFF = 65 # Hz
emg_filt = deepcopy(emg_raw)

# --- notch filter and highpass filter
for i in range(emg_filt.shape[1]):
    emg_filt[:,i] = apply_notch_filter(emg_filt[:,i], EMG_SAMPLE_RATE, EMG_NOTCH_FREQUENCIES, BW_FREQ)
    emg_filt[:,i] = apply_butter_filter(emg_filt[:,i], EMG_SAMPLE_RATE, EMG_HP_CUTOFF, btype="high")

# --- rectifying data
emg_rectify = np.abs(emg_filt)







############################
#working on smoothing emg here
############################




smooth_emg_rectify = apply_butter_filter(emg_rectify, fs = EMG_SAMPLE_RATE, cutoff_freq=10, btype="low", filt_order=4)


# --- resampling data
emg_resamp = apply_data_resampling(smooth_emg_rectify, upsampling=1, downsampling=10)
non_smooth_emg_resamp = apply_data_resampling(emg_rectify, upsampling=1, downsampling=10)
#emg_resamp = apply_data_resampling(smooth_emg_rectify, upsampling=1, downsampling=10)

# --- quartile clipping (channel 6 gets special attention)
for i in range(num_channels):
    if i == 5:
        emg_quar_chan_6 = np.quantile(emg_resamp[:, 5], 0.99)
        emg_clip_chan_6 = np.clip(emg_resamp, a_max=emg_quar_chan_6, a_min=None)
        plt.plot(emg_resamp[:, 5], label='Original Data Channel 6')
        plt.plot(emg_clip_chan_6[:, 5], label='Clipped Data Channel 6')
    else:
        emg_quar = np.quantile(emg_resamp[:, i], 0.999)
        emg_clip = np.clip(emg_resamp[:, i], a_max=emg_quar, a_min=None)
        plt.plot(emg_resamp[:, i], label= f'Original Data Channel {i+1}')
        plt.plot(emg_clip, label= f'Clipped Data Channel {i+1}')

    plt.legend()
    plt.show()

# %%
# --- normalizing by 95th percentile

all_clipped_data = []
all_normalized_data = []

# --- commented out plots plot the plot the overlaid normalized clipped and cliped emg as a sanity check
for i in range(num_channels):
    if i == 5:
        emg_quar_chan_6 = np.quantile(emg_resamp[:, 5], 0.99)
        emg_clip_chan_6 = np.clip(emg_resamp[:, 5], a_max=emg_quar_chan_6, a_min=None)
        emg_quar_chan_6_95 = np.quantile(emg_clip_chan_6, 0.95)
        emg_normal_6 = emg_clip_chan_6 / emg_quar_chan_6_95
        print(f'Channel {i+1} 99th percentile: {emg_quar_chan_6}')
        print(f'Channel {i+1} 95th percentile of clipped data: {emg_quar_chan_6_95}')
        print(f'Channel {i+1} max clipped value: {np.max(emg_clip_chan_6)}')
        print(f'Channel {i+1} max normalized value: {np.max(emg_normal_6)}')
        #plt.plot(emg_normal_6, label=f'Normalized Data Channel {i+1}')
        #plt.plot(emg_clip_chan_6, label=f'Clipped Data Channel {i+1}')
        all_clipped_data.append((i+1, emg_clip))
        all_normalized_data.append((i+1, emg_normal))
    else:
        emg_quar = np.quantile(emg_resamp[:, i], 0.999)
        emg_clip = np.clip(emg_resamp[:, i], a_max=emg_quar, a_min=None)
        emg_quar_95 = np.quantile(emg_clip, 0.95)
        emg_normal = emg_clip / emg_quar_95
        print(f'Channel {i+1} 99th percentile: {emg_quar}')
        print(f'Channel {i+1} 95th percentile of clipped data: {emg_quar_95}')
        print(f'Channel {i+1} max clipped value: {np.max(emg_clip)}')
        print(f'Channel {i+1} max normalized value: {np.max(emg_normal)}')
        #plt.plot(emg_normal, label=f'Normalized Data Channel {i+1}')
        #plt.plot(emg_clip, label=f'Clipped Data Channel {i+1}')
        all_clipped_data.append((i+1, emg_clip))
        all_normalized_data.append((i+1, emg_normal))

    #plt.legend()
    #plt.show()

# --- overlaying all the plots of clipped and normalized data - two plots with y=1
# Plot all clipped data on the same plot
plt.figure(figsize=(12, 6))
for channel, data in all_clipped_data:
    plt.plot(data, label=f'Clipped Data Channel {channel}', alpha=0.5)
    plt.axhline(y=1, color='k', linestyle='--', label='95th Quantile')
plt.legend()
plt.title('Clipped Data for All Channels')
plt.show()

# Plot all normalized data on the same plot
plt.figure(figsize=(12, 6))
for channel, data in all_normalized_data:
    plt.plot(data, label=f'Normalized Data Channel {channel}', alpha=0.5)
    plt.axhline(y=1, color='k', linestyle='--', label='95th Quantile')
plt.legend()
plt.title('Normalized Data for All Channels')
plt.show()

# %% -- sanity check: spectra differ between raw and filtered EMG

EMG_IX = 11
compare_spectra(emg_raw, emg_filt, fs=EMG_SAMPLE_RATE, dim=EMG_IX)

# %% -- preprocess kinematic data

# -- set filtering parameters
KIN_LP_CUTOFF = 40 # Hz
WINDOW_LENGTH = 27
POLYORDER = 5

joint_pos_filt = deepcopy(joint_pos_raw)
joint_vel_filt = deepcopy(joint_pos_raw)
joint_acc_filt = deepcopy(joint_pos_raw)

mk_pos_filt = deepcopy(mk_pos_raw)
mk_vel_filt = deepcopy(mk_pos_raw)
mk_acc_filt = deepcopy(mk_pos_raw)

# -- low pass filter joint angles and compute differentiation of the kinematics
for i in range(joint_pos_filt.shape[1]):
    joint_pos_filt[:,i] = apply_butter_filter(joint_pos_filt[:,i], KIN_SAMPLE_RATE, cutoff_freq=KIN_LP_CUTOFF, btype="low")
    joint_vel_filt[:,i] = apply_savgol_filter(joint_pos_filt[:,i], KIN_SAMPLE_RATE, dt_kin, window_length=WINDOW_LENGTH, polyorder=POLYORDER, deriv=1)
    joint_acc_filt[:,i] = apply_savgol_filter(joint_pos_filt[:,i], KIN_SAMPLE_RATE, dt_kin, window_length=WINDOW_LENGTH, polyorder=POLYORDER, deriv=2)

# -- low pass filter marker positions and compute differentiation of the kinematics
for i in range(mk_pos_filt.shape[1]):
    mk_pos_filt[:,i] = apply_butter_filter(mk_pos_filt[:,i], KIN_SAMPLE_RATE, cutoff_freq=KIN_LP_CUTOFF, btype="low")
    mk_vel_filt[:,i] = apply_savgol_filter(mk_pos_filt[:,i], KIN_SAMPLE_RATE, dt_kin, window_length=WINDOW_LENGTH, polyorder=POLYORDER, deriv=1)
    mk_acc_filt[:,i] = apply_savgol_filter(mk_pos_filt[:,i], KIN_SAMPLE_RATE, dt_kin, window_length=WINDOW_LENGTH, polyorder=POLYORDER, deriv=2)

#  %% -- resample data

mk_pos_resamp = apply_data_resampling(mk_pos_filt, upsampling=5, downsampling=2)
mk_vel_resamp = apply_data_resampling(mk_vel_filt, upsampling=5, downsampling=2)
mk_acc_resamp = apply_data_resampling(mk_acc_filt, upsampling=5, downsampling=2)

joint_pos_resamp = apply_data_resampling(joint_pos_filt, upsampling=5, downsampling=2)
joint_vel_resamp = apply_data_resampling(joint_vel_filt, upsampling=5, downsampling=2)
joint_acc_resamp = apply_data_resampling(joint_acc_filt, upsampling=5, downsampling=2)

#plotting to ensure resampling looks right
plt.figure(figsize=(12, 6))
plt.scatter(np.linspace(0, len(mk_pos_filt[:, 0]) / KIN_SAMPLE_RATE, len(mk_pos_resamp[:,0])), mk_pos_resamp[:,0], label='Resampled MK Data', alpha=0.7, s=1)
plt.scatter(np.linspace(0, len(mk_pos_filt[:, 0]) / KIN_SAMPLE_RATE, len(mk_pos_filt[:, 0])), mk_pos_filt[:, 0], label='Original MK Data', alpha=0.7, s=1)
plt.legend()

plt.figure(figsize=(12, 6))
plt.scatter(np.linspace(0, len(joint_pos_filt[:, 0]) / KIN_SAMPLE_RATE, len(joint_pos_resamp[:,0])), joint_pos_resamp[:,0], label='Resampled Joint Data', alpha=0.7, s=1)
plt.scatter(np.linspace(0, len(joint_pos_filt[:, 0]) / KIN_SAMPLE_RATE, len(joint_pos_filt[:, 0])), joint_pos_filt[:, 0], label='Original Joint Data', alpha=0.7, s=1)
plt.legend()

plt.figure(figsize=(12, 6))
plt.scatter(np.linspace(0, len(emg_filt[:, 0]) / EMG_SAMPLE_RATE, len(emg_filt[:, 0])), emg_filt[:, 0], label='Original EMG Data', alpha=0.7, s=1)
plt.scatter(np.linspace(0, len(emg_filt[:, 0]) / EMG_SAMPLE_RATE, len(emg_resamp[:,0])), emg_resamp[:,0], label='Resampled EMG Data', alpha=0.7, s=1)
plt.legend()

# %% -- sanity check: compare spectra
KIN_IX = 4
EMG_IX = 4
compare_spectra(joint_pos_raw, joint_pos_filt, fs=KIN_SAMPLE_RATE, dim=KIN_IX, nfft=400, nperseg=400, noverlap=50)
compare_spectra(emg_raw, emg_filt, fs=EMG_SAMPLE_RATE, dim=EMG_IX, nfft=400, nperseg=400, noverlap=50)

# %% --- making a pandas dataframe for joint/mk position, velocity, acceleration and emg
#adding each column to the dataframe
df_emg = pd.DataFrame(emg_resamp, columns = emg_chan_names)
df_mk_pos = pd.DataFrame(mk_pos_resamp, columns = xy_marker_names)
df_mk_vel = pd.DataFrame(mk_vel_resamp, columns=[f"{name}_vel" for name in xy_marker_names])
df_mk_acc = pd.DataFrame(mk_acc_resamp, columns=[f"{name}_acc" for name in xy_marker_names])
df_joint_pos = pd.DataFrame(joint_pos_resamp, columns = joint_names)
df_joint_vel = pd.DataFrame(joint_vel_resamp, columns=[f"{name}_vel" for name in joint_names])
df_joint_acc = pd.DataFrame(joint_acc_resamp, columns=[f"{name}_acc" for name in joint_names])

#adding time vector and indexing for each dataframe
dataframes = [df_emg, df_mk_pos, df_mk_vel, df_mk_acc, df_joint_pos, df_joint_vel, df_joint_acc]
time_vector = np.linspace(0, len(df_emg) / EMG_SAMPLE_RATE, len(df_emg))

for df in dataframes:
    df['Time'] = time_vector
    df.set_index('Time', inplace=True)

#concatenating dataframes
df_all = pd.concat({
    'EMG': pd.concat({
        'EMG': df_emg #repetitive because needed all dataframes on the same level
    }, axis=1),
    'Marker': pd.concat({
        'Position': df_mk_pos,
        'Velocity': df_mk_vel,
        'Acceleration': df_mk_acc
    }, axis = 1),
    'Joint': pd.concat({
        'Position': df_joint_pos,
        'Velocity': df_joint_vel,
        'Acceleration': df_joint_acc
    }, axis = 1)
}, axis = 1)
print(df_all)

# note: mtp_acc has 120 nan values

# %% 
# --- defining toevelocity as a marker to differentiate between stance and swing
toevelocity = df_all['Marker']['Velocity']['toe_y_vel']
time = df_all.index

# --- graphing toevelocity and mapping out stance and swing visually
plt.figure(figsize=(12,6))
peaks, _ = signal.find_peaks(toevelocity, prominence=300)
troughs, _ = signal.find_peaks(-toevelocity, prominence=300)

plt.plot (time, toevelocity, color='black')

plt.scatter(time[peaks], toevelocity.iloc[peaks], label ='Peaks')
plt.scatter(time[troughs], toevelocity.iloc[troughs], label ='Peaks')


plt.xlabel('Time (s)')
plt.ylabel('Velocities') 
plt.title('Toe velocities over time')
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.axhline(y=50, color='blue', linestyle='--', linewidth=1)
plt.axhline(y=-20, color='blue', linestyle='--', linewidth=1)

#plt.ylim([-100,100])
plt.xlim([2,3.25])
plt.show()


# %%
# --- plotting stance/swing graph
plt.figure(figsize=(24,6))
mindistance = .015
mindistance_samples = int(mindistance * EMG_SAMPLE_RATE)
lower_mindistance = .02
upper_mindistance = .03

peaks, _ = signal.find_peaks(toevelocity, prominence=300, distance=mindistance_samples)
troughs, _ = signal.find_peaks(-toevelocity, prominence=300, distance=mindistance_samples)

pos_crossing_thresh = 80
neg_crossing_thresh = -45

def thresh_crossings(data, time, lower_threshold, upper_threshold, start_indices, end_indices, min_distance, lower_min_distance, upper_min_distance, new_points=None, indices_to_delete=None):
    crossings = []
    last_crossing_time = -np.inf 
    last_lower_crossing_time = -np.inf #initialized to negative infinity
    for start, end in zip(start_indices, end_indices):
        after_trough = False #flags troughs 
        for i in range(start, end):
            if data.iloc[i] == min(data.iloc[start:end+1]): #detcts minimum point to flag as a trough
                after_trough = True #trough has been detected, now looking for the exact point
            if after_trough:
                if data.iloc[i] < lower_threshold and data.iloc[i + 1] > lower_threshold:
                    lower_crossing_time = time[i] + (time[i + 1] - time[i]) * ((lower_threshold - data.iloc[i]) / (data.iloc[i + 1] - data.iloc[i])) #calculate crossing based on linear interpolation
                    if lower_crossing_time - last_lower_crossing_time >= lower_min_distance: #ensure separation by lower min distance
                        crossings.append((lower_crossing_time, lower_threshold))
                        last_lower_crossing_time = lower_crossing_time
                if data.iloc[i] < upper_threshold and data.iloc[i + 1] > upper_threshold:
                    upper_crossing_time = time[i] + (time[i + 1] - time[i]) * ((upper_threshold - data.iloc[i]) / (data.iloc[i + 1] - data.iloc[i])) #calculate crossing based on linear interpolation
                    if upper_crossing_time - last_crossing_time >= upper_min_distance: #ensure separation by upper min distance
                        crossings.append((upper_crossing_time, upper_threshold))
                        last_crossing_time = upper_crossing_time

    if new_points:
        crossings.extend(new_points)
        crossings.sort() #add in new points sorted by time

    if indices_to_delete:
        crossings = [crossing for i, crossing in enumerate(crossings) if i not in indices_to_delete] #delete redundant points

    return crossings

new_points = [(3.461, -45), (6.024, -45), (11.302, -45)] #adding new points where algorithm got messed up
indices_to_delete = [86, 97, 98, 176, 347] #removing points where algorithm messed up
crossings = thresh_crossings(toevelocity, time, neg_crossing_thresh, pos_crossing_thresh, troughs[:-1], peaks[1:], mindistance, lower_mindistance, upper_mindistance, new_points=new_points, indices_to_delete=indices_to_delete)

#count upper and lower crossings
upper_crossings = [crossing for crossing in crossings if crossing[1] == pos_crossing_thresh]
lower_crossings = [crossing for crossing in crossings if crossing[1] == neg_crossing_thresh]
upper_crossings_count = len(upper_crossings)
lower_crossings_count = len(lower_crossings)

#print(f'Upper crossings count: {upper_crossings_count}')
#print(f'Lower crossings count: {lower_crossings_count}')
#print('Upper crossings:', upper_crossings)
#print('Lower crossings:', lower_crossings)

# --- plotting with the indices labeled
plt.plot(time, toevelocity)
for i, (crossing_time, threshold) in enumerate(crossings):
    plt.scatter(crossing_time, threshold, color='black', label='Crossing' if i == 0 else "")
    plt.annotate(str(i), (crossing_time, threshold), textcoords="offset points", xytext=(0,10), ha='center')

plt.axhline(y=neg_crossing_thresh, color='red', linestyle='--', linewidth=1, label='Lower Threshold')
plt.axhline(y=pos_crossing_thresh, color='blue', linestyle='--', linewidth=1, label='Upper Threshold')

plt.xlim([10.8,11.5])

plt.show()

# %% --- making swing/stance dataframe
upper_crossing_times = [crossing_time for crossing_time, _ in upper_crossings]
lower_crossing_times = [crossing_time for crossing_time, _ in lower_crossings]


# creating dataframe
step_df = pd.DataFrame({
    'Start Stance': lower_crossing_times,
    'End Stance': upper_crossing_times,
    'Start Swing': upper_crossing_times,
    'End Swing': lower_crossing_times[1:] + [np.nan]
})

#adding stance and swing durations columns
step_df['Stance Duration'] = step_df['End Stance'] - step_df['Start Stance']
step_df['Swing Duration'] = step_df['End Swing'] - step_df['Start Swing']
step_df['stance_duration_seconds'] = step_df['Stance Duration']

#naning where swing is weird
step_df.loc[42, 'Start Swing'] = np.nan
step_df.loc[42, 'End Swing'] = np.nan
step_df.loc[47, 'Start Swing'] = np.nan
step_df.loc[47, 'End Swing'] = np.nan
step_df.loc[85, 'Start Swing'] = np.nan
step_df.loc[85, 'End Swing'] = np.nan
step_df.loc[170, 'Start Swing'] = np.nan
step_df.loc[170, 'End Swing'] = np.nan

#removing the 4 outliers on both sides
smallest_indices = step_df['Stance Duration'].nsmallest(4).index
largest_indices = step_df['Stance Duration'].nlargest(4).index

# Set these values to NaN
step_df.loc[smallest_indices, 'Stance Duration'] = np.nan
step_df.loc[largest_indices, 'Stance Duration'] = np.nan


# making the last start swing nan since the data ends in a stance
step_df.at[len(step_df) - 1, 'Start Swing'] = np.nan
print(step_df.to_string()) #prints scrollable step df

# %% --- making stance and swing duration histograms
stance_duration= step_df['Stance Duration']
plt.hist(stance_duration, bins=100)
plt.xlabel('Stance Times (in s)')
plt.ylabel('Step count')
plt.title('Stance Times for Each Step')
plt.show()

swing_duration= step_df['Swing Duration']
plt.hist(swing_duration, bins=100)
plt.xlabel('Swing times (in s)')
plt.ylabel('Step Count')
plt.title('Swing Times for Each Step')
plt.show()


# %% --- plotting psths
WINDOW_SIZE = .025
#note: 180 bins (or step number) is the individual trial plots

def plotting_trials(step_df, df_all, num_bins=10, window_size=0.025):
    # Normalize stance duration for color mapping
    norm = Normalize(vmin=.035, vmax=.065)  # selecting range of values with majority of points
    cmap = plt.get_cmap('plasma')
    step_df['bin'] = pd.qcut(step_df['Stance Duration'], num_bins, labels=False)

    fig, ax = plt.subplots(figsize=(10, 6))

    for bin_num in range(num_bins):
        bin_df = step_df[step_df['bin'] == bin_num]  # collects rows corresponding to select bin
        mean_data = None  # stores sum of EMG data
        count = 0

        for idx, row in bin_df.iterrows():
            swing_onset = row['Start Swing']

            start_window = swing_onset - window_size
            end_window = swing_onset + window_size

            data_window = df_all[(df_all.index >= start_window) & (df_all.index <= end_window)]  # gets data within time window
            if data_window.empty:  # skips if window is empty
                continue

            time_relative = (data_window.index - swing_onset) * 1000  # convert to milliseconds

            if mean_data is None:
                mean_data = data_window['EMG']['EMG']['VL'].values  # first iteration through dataframe
            else:
                mean_data += data_window['EMG']['EMG']['VL'].values  # continues to add data and increases count by 1
            count += 1

        if mean_data is not None:  # calculates the average within the current bin
            mean_data /= count  # dividing data by count within bin
            color = cmap(norm(bin_df['Stance Duration'].mean()))  # determines color of plotted line based on mean of values within bin
            ax.plot(time_relative, mean_data, color=color, label=f'Bin {bin_num + 1}')
            ax.axvline(color='red', linestyle='--')  # vertical line at onset time

    ax.set_xlim(-window_size * 1000, window_size * 1000)  # convert to milliseconds
    ax.set_title("VL Activity Across Multiple Step Cycles")
    ax.set_xlabel('Time Relative to Swing Onset (ms)')
    ax.set_ylabel('Rectified EMG Signal')

    # Drop the spines on the top and right
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # sets array for colorbar
    plt.colorbar(sm, ax=ax, label='Stance Duration (s)')

    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage
plotting_trials(step_df, df_all, num_bins=20, window_size=0.025)  # edit bin sizes here

# %% --- pca plots

import pandas as pd
import plotly.graph_objs as go
from sklearn.decomposition import PCA
import numpy as np
import time

index = df_all.index

# Function to check if each datapoint is stance or swing using vectorized operations
def check_phase(index, step_df):
    phase = np.full(len(index), 'Unknown', dtype=object)

    stance_start = step_df['Start Stance'].values
    stance_end = step_df['End Stance'].values
    swing_start = step_df['Start Swing'].values
    swing_end = step_df['End Swing'].values

    index_arr = np.array(index)

    # Vectorized operations to determine stance and swing phases
    is_stance = np.any((stance_start[:, None] <= index_arr) & (index_arr <= stance_end[:, None]), axis=0)
    is_swing = np.any((swing_start[:, None] <= index_arr) & (index_arr <= swing_end[:, None]), axis=0)

    phase[is_stance] = 'Stance'
    phase[is_swing] = 'Swing'

    return phase

# Timing the phase checking
start_time = time.time()
phases = check_phase(index, step_df)
end_time = time.time()
print(f"Time taken for phase checking: {end_time - start_time} seconds")

# Create a DataFrame to display the results
start_time = time.time()
result_df = pd.DataFrame({'Time': index, 'Phase': phases})
print(result_df.to_string())
end_time = time.time()
print(f"Time taken to create result_df: {end_time - start_time} seconds")

# Convert Phase column to a multi-level column before merging
result_df.columns = pd.MultiIndex.from_tuples([('Time', ''), ('Phase', '')])
result_df.set_index(('Time', ''), inplace=True)
result_multi = pd.concat({('Phase', ''): result_df['Phase']}, axis=1)

start_time = time.time()
from sklearn.preprocessing import StandardScaler
emg_data = StandardScaler().fit_transform(df_all['EMG']['EMG'].values)
pca = PCA(n_components=3)
emg_pca = pca.fit_transform(df_all['EMG']['EMG'].values)
end_time = time.time()
print(f"Time taken for PCA: {end_time - start_time} seconds")

# Merge the phase information back into df_all
start_time = time.time()
df_all = pd.concat({
    'EMG': pd.concat({
        'EMG': df_emg
    }, axis=1),
    'Marker': pd.concat({
        'Position': df_mk_pos,
        'Velocity': df_mk_vel,
        'Acceleration': df_mk_acc
    }, axis=1),
    'Joint': pd.concat({
        'Position': df_joint_pos,
        'Velocity': df_joint_vel,
        'Acceleration': df_joint_acc
    }, axis=1),
    'Phase': result_multi
}, axis=1)

end_time = time.time()
print(f"Time taken to merge phase information: {end_time - start_time} seconds")
print(df_all)

# Separate indices for stance and swing
start_time = time.time()
stance_indices = np.array(np.where(df_all[('Phase', 'Phase')] == 'Stance')[0])
swing_indices = np.array(np.where(df_all[('Phase', 'Phase')] == 'Swing')[0])
print(f"Time taken to separate indices: {end_time - start_time} seconds")

# Plotting
import plotly.graph_objs as go
import time

# Assuming emg_pca, stance_indices, and swing_indices are already defined

# Plotting with Plotly
start_time = time.time()

trace_stance = go.Scatter3d(
    x=emg_pca[stance_indices, 0],
    y=emg_pca[stance_indices, 1],
    z=emg_pca[stance_indices, 2],
    mode='lines+markers',
    marker=dict(size=1, color='blue'),
    name='Stance'
)

trace_swing = go.Scatter3d(
    x=emg_pca[swing_indices, 0],
    y=emg_pca[swing_indices, 1],
    z=emg_pca[swing_indices, 2],
    mode='lines+markers',
    marker=dict(size=1, color='red'),
    name='Swing'
)

data = [trace_stance, trace_swing]
layout = go.Layout(
    title='3D PCA Trajectory of Smoothed EMG Data',
    scene=dict(
        xaxis=dict(title='PC1'),
        yaxis=dict(title='PC2'),
        zaxis=dict(title='PC3')
    ),
    legend=dict(title='Phase')
)

fig = go.Figure(data=data, layout=layout)
fig.show()

end_time = time.time()
print(f"Time taken to plot: {end_time - start_time} seconds")















# %% -- replicated code to try stuff on
index = df_all.index

# Function to check if each datapoint is stance or swing
def check_phase(index, step_df):
    phase = np.full(len(index), 'Unknown', dtype=object)

    stance_start = step_df['Start Stance'].values
    stance_end = step_df['End Stance'].values
    swing_start = step_df['Start Swing'].values
    swing_end = step_df['End Swing'].values

    index_arr = np.array(index)

    # Vectorized operations to determine stance and swing phases
    is_stance = np.any((stance_start[:, None] <= index_arr) & (index_arr <= stance_end[:, None]), axis=0)
    is_swing = np.any((swing_start[:, None] <= index_arr) & (index_arr <= swing_end[:, None]), axis=0)

    phase[is_stance] = 'Stance'
    phase[is_swing] = 'Swing'

    return phase

phases = check_phase(index, step_df)

# Create a DataFrame to display the results
result_df = pd.DataFrame({'Time': index, 'Phase': phases})
print(result_df.to_string())


pca = PCA(n_components=3)
emg_pca = pca.fit_transform(df_all['EMG']['EMG'].values)

# Merge the phase information back into df_all
df_all = pd.concat({
    'EMG': pd.concat({
        'EMG': df_emg #repetitive because needed all dataframes on the same level
    }, axis=1),
    'Marker': pd.concat({
        'Position': df_mk_pos,
        'Velocity': df_mk_vel,
        'Acceleration': df_mk_acc
    }, axis = 1),
    'Joint': pd.concat({
        'Position': df_joint_pos,
        'Velocity': df_joint_vel,
        'Acceleration': df_joint_acc
    }, axis = 1),
    'Phase': pd.concat({
        'Phase': phases #repep to be able to concatenate dataframes
    }, axis=1)
}, axis = 1)
print(df_all)

# %%
stance_indices = df_all[df_all['Phase'] == 'Stance'].index.to_numpy().astype(int)
swing_indices = df_all[df_all['Phase'] == 'Swing'].index.to_numpy().astype(int)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot(emg_pca[stance_indices, 0], emg_pca[stance_indices, 1], emg_pca[stance_indices, 2], label='Stance', color='blue')
ax.plot(emg_pca[swing_indices, 0], emg_pca[swing_indices, 1], emg_pca[swing_indices, 2], label='Swing', color='red')

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('3D PCA Trajectory of Smoothed EMG Data')
ax.legend()

plt.show()