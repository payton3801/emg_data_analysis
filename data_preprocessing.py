# %% -- importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.io import loadmat
import seaborn as sns
import math
from copy import deepcopy
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# %% -- load mat file
mat_data = loadmat('J10_s10_i0_pref.mat')
# %% -- load raw emg

# -- load data, sampling rate, time vector
emg_raw = mat_data['emg_full_raw'] 
t_emg = mat_data['t_emg'].squeeze()
dt_emg = np.mean(np.diff(t_emg))
EMG_SAMPLE_RATE = np.round(1/dt_emg)

# -- load emg channel names
emg_chan_names = mat_data['emg_names'].flatten()
emg_chan_names = mat_data['emg_names'][0]
emg_chan_names = [str(name[0]) for name in emg_chan_names]

# %% -- load raw kinematics

# -- load joint angle data (multi-dimensional np array)
joints = mat_data['joints_raw_nonSeg']
# remove limbV_ln from joint angle data
joints = joints[:-1]

# compute number of joints (after removing limbV_ln)
n_joints = joints.shape[0]

# -- load kinematics marker position data
markers = mat_data['mk_raw_nonSeg']
#markers = markers.reshape(7,1)
#num_markers = markers.shape[0]

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

# %% -- sanity check: spectra differ between raw and filtered EMG


EMG_IX = 0
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
    # --- TODO: apply savgol filter for markers

#  %% -- resample data




# %% -- sanity check: compare spectra
KIN_IX = 4
compare_spectra(joint_pos_raw, joint_pos_filt, fs=KIN_SAMPLE_RATE, dim=KIN_IX, nfft=400, nperseg=400, noverlap=50)
# %%
#KIN_SAMPLE_RATE
# %%
