# %%
#note : resampling is in milliseconds not seconds for nwb dataset
from snel_toolkit.datasets.nwb import NWBDataset
from snel_toolkit.datasets.base import DataWrangler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from snel_toolkit.decoding import prepare_decoding_data
import scipy.signal as signal
from os import path
from tqdm import tqdm
import _pickle as pickle
import logging
import sys
import os
import yaml
import pandas as pd
import dill
from snel_toolkit.analysis import PSTH
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go


# %% -- load in dataset

#configure logging messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ds_base_dir = "/snel/share/share/data/Tresch_gaitEMG/data/NWB/"
#ds_name = "J10_s20_i25"
#ds_name = "J10_s10_i0"
ds_name = "J10_s20_i0" #looks similar to the paper
BIN_SIZE = 2  # ms
use_cached = False  #Set this to True if you want to use cached data
nwb_cache_dir = "/path/to/cache/dir"  #Where cache data is stored

if use_cached:
    ds_path = path.join(nwb_cache_dir, "nlb_" + ds_name + ".pkl")
    logger.info("Loading from pickled NLB")
    with open(ds_path, "rb") as rfile:
        dataset = pickle.load(rfile)
        logger.info("Dataset loaded from pickle.")
else: #likely using this most of the time
    ds_path = path.join(ds_base_dir, ds_name + ".nwb")
    logger.info("Loading from NWB")
    dataset = NWBDataset(ds_path)
    # if needed, resample
    #dataset.resample(BIN_SIZE)

crit_freqs = [2, 7, 10, 20, 40] #smooth out these specific frequencies

#path of merged lfads data
lfads_path = '/snel/share/share/tmp/pbechef/Tresch/nwb_lfads/runs/run_002/torch_output/best_model/lfads_J10_s20_i0_emg_2_full_merged_output.pkl'

with open(lfads_path, "rb") as f: #rb = read binary mode, required for binary data
    lfads_data = dill.load(f) #assigns file object to variable f

for crit_freq in crit_freqs:
    lfads_data.smooth_cts(
        signal_type = 'emg',
        filt_type = 'butter',
        crit_freq = crit_freq,
        btype = "low",
        order = 4,
        name = f'lf{crit_freq}',
        overwrite = False, 
        use_causal = False #zero phase filtering
    )


# %% -- defining filtering functions

# savistky golay differentiation function
def apply_savgol_diff(x, window_length, polyorder, deriv, delta):
    """apply savitsky-golay differentiation (1st order)"""
    DERIV = 1
    FRAMELEN = 27
    POLY_ORDER = 5
    y = signal.savgol_filter(
        x,
        window_length=window_length,
        polyorder=polyorder, #order of the polynomial before filtering
        deriv=deriv, #order of deriv to compute
        delta=delta, #spacing between data points
        mode="constant",
    )
    return y


def _filter(b, a, x):
    """Zero-phase digital filtering that handles NaNs. Replaces NAN values with the mean of the signal"""
    if isinstance(x, pd.Series):
        # Handle NaNs for Series
        nan_mask = x.isna()
        if nan_mask.any():
            x[nan_mask] = x.mean()  # Replace NaNs with the mean for filtering
        x_filtered = signal.filtfilt(b, a, x)
        x[nan_mask] = np.nan  # Restore NaNs
        return pd.Series(x_filtered, index=x.index)
    elif isinstance(x, np.ndarray):
        # Handle NaNs for numpy arrays
        nan_mask = np.isnan(x)
        if nan_mask.any():
            x[nan_mask] = np.nanmean(x)  # Replace NaNs with the mean for filtering
        x_filtered = signal.filtfilt(b, a, x)
        x[nan_mask] = np.nan  # Restore NaNs
        return x_filtered
    else:
        raise ValueError("Input must be a pandas Series or numpy array.")


# high-pass filter
def apply_butter_filt(x, fs, filt_type, cutoff_freq, filt_order=4):
    """Apply 4th-order Butterworth filtering at a specified frequency."""
    # Design filter
    b, a = signal.butter(filt_order, cutoff_freq, btype=filt_type, fs=fs)

    # Apply filter
    if isinstance(x, pd.DataFrame):
        return x.apply(lambda col: _filter(b, a, col), axis=0)
    elif isinstance(x, pd.Series):
        return _filter(b, a, x)
    else:
        raise ValueError("Input must be a pandas DataFrame or Series.")


# %%
dataset.data
#dataset.trial_info

# %%
# -- update joint angle differentation

# # savgol differentiation parameters
WINDOW_LENGTH = 27
POLYORDER = 5
DELTA = lfads_data.bin_width / 1000
# == smoothing cutoff
LF_CUTOFF = 75  # Hz

jnt_p = lfads_data.data.joint_ang_p
# lf filter joint angles
jnt_p_filt = jnt_p.apply(apply_butter_filt, args=(1000 / lfads_data.bin_width, "low", 75))

# joint differentiation
jnt_v = jnt_p_filt.apply(apply_savgol_diff, args=(WINDOW_LENGTH, POLYORDER, 1, DELTA))
jnt_a = jnt_p_filt.apply(apply_savgol_diff, args=(WINDOW_LENGTH, POLYORDER, 2, DELTA))

jnt_a_40 = jnt_a.apply(apply_butter_filt, args=(1000 / lfads_data.bin_width, "low", 40))


joint_names = jnt_p.columns.values.tolist()
for joint_name in joint_names:
    lfads_data.data[("joint_ang_v", joint_name)] = jnt_v[joint_name]
    lfads_data.data[("joint_ang_a", joint_name)] = jnt_a[joint_name]
    lfads_data.data[("joint_ang_a_40", joint_name)] = jnt_a_40[joint_name]

# %% -- quantile clip and normalize
# lfads_data.data.emg_rectified = lfads_data.data.emg.abs()

# num_channels = 12

# --- quartile clipping (channel 6 gets special attention)
# for i in range(num_channels):
#     if i == 5:
#         emg_quar_chan_6 = np.quantile(lfads_data.data.emg_rectified.iloc[:, 5], 0.99)
#         emg_clip_chan_6 = np.clip(lfads_data.data.emg_rectified.iloc[:, 5], a_max=emg_quar_chan_6, a_min=None)
#         #plt.plot(lfads_data.data.emg.iloc[:, 5], label='Original Data Channel 6')
#         plt.plot(emg_clip_chan_6, label='Clipped Data Channel 6')
#     else:
#         emg_quar = np.quantile(lfads_data.data.emg_rectified.iloc[:, i], 0.999)
#         emg_clip = np.clip(lfads_data.data.emg_rectified.iloc[:, i], a_max=emg_quar, a_min=None)
#         #plt.plot(lfads_data.data.emg.iloc[:, i], label= f'Original Data Channel {i+1}')
#         plt.plot(emg_clip, label= f'Clipped Data Channel {i+1}')

#     plt.legend()
#     #plt.show()

# # --- quantile clipping lfads data
# lfads_clipped = lfads_data.data.apply(lambda col: np.clip(col, a_min=None, a_max=np.quantile(col, 0.99)), axis=0)

# %%
# --- normalizing by 95th percentile
# num_channels = 12

# all_clipped_data = []
# all_normalized_data = []

# %%
# --- commented out plots plot the plot the overlaid normalized clipped and clipped emg as a sanity check
# for i in range(num_channels):
#     if i == 5:
#         emg_quar_chan_6 = np.quantile(lfads_data.data.emg_rectified.iloc[:, 5], 0.99)
#         emg_clip_chan_6 = np.clip(lfads_data.data.emg_rectified.iloc[:, 5], a_max=emg_quar_chan_6, a_min=None)
#         emg_quar_chan_6_95 = np.quantile(emg_clip_chan_6, 0.95)
#         emg_normal_6 = emg_clip_chan_6 / emg_quar_chan_6_95
#         print(f'Channel {i+1} 99th percentile: {emg_quar_chan_6}')
#         print(f'Channel {i+1} 95th percentile of clipped data: {emg_quar_chan_6_95}')
#         print(f'Channel {i+1} max clipped value: {np.max(emg_clip_chan_6)}')
#         print(f'Channel {i+1} max normalized value: {np.max(emg_normal_6)}')
#         plt.plot(emg_normal_6, label=f'Normalized Data Channel {i+1}')
#         plt.plot(emg_clip_chan_6, label=f'Clipped Data Channel {i+1}')
#         all_clipped_data.append((i+1, emg_clip_chan_6))
#         all_normalized_data.append((i+1, emg_normal_6))
#     else:
#         emg_quar = np.quantile(lfads_data.data.emg_rectified.iloc[:, i], 0.999)
#         emg_clip = np.clip(lfads_data.data.emg_rectified.iloc[:, i], a_max=emg_quar, a_min=None)
#         emg_quar_95 = np.quantile(emg_clip, 0.95)
#         emg_normal = emg_clip / emg_quar_95
#         print(f'Channel {i+1} 99th percentile: {emg_quar}')
#         print(f'Channel {i+1} 95th percentile of clipped data: {emg_quar_95}')
#         print(f'Channel {i+1} max clipped value: {np.max(emg_clip)}')
#         print(f'Channel {i+1} max normalized value: {np.max(emg_normal)}')
#         plt.plot(emg_normal, label=f'Normalized Data Channel {i+1}')
#         plt.plot(emg_clip, label=f'Clipped Data Channel {i+1}')
#         all_clipped_data.append((i+1, emg_clip))
#         all_normalized_data.append((i+1, emg_normal))

    #plt.legend()
    #plt.show()


# %% -- making dataframe of normalized data
# normalized_df = pd.DataFrame(
#     {f"Channel_{i+1}": data for i, data in all_normalized_data}
# )

# # Ensure the normalized_df has the same shape as the original lfads_data.data.emg
# assert normalized_df.shape == lfads_data.data.emg.shape, "Shape mismatch between normalized data and original EMG data."


# # %%
# # Replace the values in the original DataFrame with the normalized values
# # Align the index of normalized_df with lfads_data.data.emg
# normalized_df.index = lfads_data.data.emg.index
# normalized_df.columns = lfads_data.data.emg.columns
# #lfads_data.data.emg.loc[:, :] = normalized_df.values
# lfads_data.data.emg = normalized_df.copy()



# # %% -- resample
# lfads_data.resample(BIN_SIZE) #should resample to 500


# %% -- absolute value EMG post-resampling
# lfads_data.data.emg = lfads_data.data.emg.abs()

# # %% low pass filter at the end (abs value again?)
# lfads_data.data.emg = apply_butter_filt(lfads_data.data.emg , fs = 500, cutoff_freq=10, filt_type="low", filt_order=4)

# lfads_data.data.emg = lfads_data.data.emg.abs()

# %% -- removing bad emg trials

bad_emg_trials = [
    0,
    3,
    4,
    14,
    15,
    24,
    25,
    34,
    36,
    47,
    57,
    59,
    70,
    73,
    87,
    107,
    110,
    119,
    126,
    140,
    153,
    154,
    182,
    192,
    193,
    205,
    210,
    226,
    229,
]
# bad_emg_trials = []
ti = lfads_data.trial_info #extract trial info
bad_emg = ti.trial_id == -1  # all false
for trial_id in bad_emg_trials:
    bad_emg[ti.trial_id == trial_id] = True #marks corresponding files as true (bad emg files)

bad_cycle = ti.good_cycle == 0 #True = bad trials

ignore_trials = bad_cycle | bad_emg #ignores both bad cycles and bad emg trials

#perhaps i need this line????

# for col in lfads_data.data.columns:
#     lfads_data.data[col] = lfads_data.data[col]

# dw = DataWrangler(lfads_data)


# dw.make_trial_data(
#     name="foot_off",
#     align_field="foot_off_time",
#     align_range=(-100, 250),
#     ignored_trials=ignore_trials,
# )

# %% -- lag decoding 
# predict how well x-field can predict y-field at different time lags

def lag_decoding(trial_df, x_field, y_field, bin_ms, lag_steps=10, k_folds=10):
    logger.info("Inferring data dimensionality")
    lagged_data = prepare_decoding_data(
        trial_df, x_field, y_field, valid_ratio=0, ms_lag=0 #no lag initially, all data used for training and testing
    )
    
    #unpack return values
    X_data, y_data, clock_time = lagged_data  #adding in the third returned value
    
    x_ss = StandardScaler() #removing mean and scaling to unit variance to standardize the data
    x_ss.fit(X_data) #makes sure all features have the same scalar
    y_ss = StandardScaler()
    y_ss.fit(y_data) #makes sure same scalar
    lag_r2 = np.zeros( 
        (lag_steps, y_data.shape[1], k_folds) #3d array to store the r^2 scores of lag, target axis, and cross validation folds
    )  # (lags) x (axis) x (fold score)

    for lag in range(0, lag_steps):
        lagged_data = prepare_decoding_data(
            trial_df, x_field, y_field, valid_ratio=0, ms_lag=bin_ms * lag #applies time lag to input data
        )
        
        # unpacked all three returned values 
        X_data, y_data, clock_time = lagged_data  
        
        X_data = x_ss.transform(X_data)
        y_data = y_ss.transform(y_data) #data is standardized again
        for i in range(y_data.shape[1]):
            r2_i = cross_val_score(
                Ridge(alpha=1e0), X_data, y_data[:, i], cv=k_folds, scoring="r2"
            )
            lag_r2[lag, i, :] = r2_i

    sem = lambda x: np.std(x, axis=2) / np.sqrt(x.shape[2])

    lag_r2_sem = sem(lag_r2)  # sem score of folds
    lag_r2_mean = np.mean(lag_r2, axis=2)  # average score of folds
    opt_r2 = np.max(lag_r2_mean, axis=0)  # compute max over all lags (for each axis)

    logger.info(f"Optimal performance decoding {y_field} from {x_field} : {opt_r2}")

    return lag_r2_mean, lag_r2_sem

# %% -- lagged decoding analysis to evaluate how well smoothed emg signals can predict joint accelerations

#xfields = input signal fields
#yfield = target signal being predicted

xfields = [
    "lfads_rates",
    "emg_lf40",
    "emg_lf20",
    "emg_lf10",
    "emg_lf7",
    "emg_lf2",
]

yfield = "joint_ang_a_40" 

dw = DataWrangler(lfads_data)  #aligning data to foot off times
dw.make_trial_data(
    name="foot_off",
    align_field="foot_off_time",
    align_range=(-100, 250),
    ignored_trials=None,
    allow_overlap=True
)

#creating clock time index if needed, as error was being created by no clock time index
if 'clock_time' not in dw._t_df.columns:
    dw._t_df['clock_time'] = dw._t_df.index


#performing lag decoding for xfield
r2_mean = []
r2_sem = []
for xfield in xfields:
    lag_r2_mean, lag_r2_sem = lag_decoding(dw._t_df, xfield, yfield, 2, lag_steps=25) #inputs: dw,_t_df: trial-aligned dataframe, xfield, yfield, bin_ms: bin size, lag_steps=25: number of time lags to eval
    r2_mean.append(lag_r2_mean)
    r2_sem.append(lag_r2_sem)

#combines r2_mean and r2_sem lists into 3d arrays
r2_mean = np.stack(r2_mean)
r2_sem = np.stack(r2_sem)

r2_mean_opt = np.max(r2_mean, axis=1) #max r2 for each xfield across all lags
max_ix = np.argmax(r2_mean, axis=1) #index of the max r2 for each xfield across all lags
x, z = np.indices(max_ix.shape) #creates arrays of indices for the xfields and yfields dimensions
r2_sem_opt = r2_sem[x, max_ix, z] #extracts SEM values corresponding to optimal lags

# %%
group_ixs = [1, 3, 0]  # hip bot, knee, ankle
group_spacing = 4  # spacing between groups
n_bars = len(xfields)
n_groups = len(group_ixs)

# hacky way to figure out how many deEMG names there are
n_deemg = sum(["deEMG" in name for name in xfields])

# --- specify colors

cval_1 = 190
cval_2 = 55

lf_colors = []
lf_cutoffs = [40, 20, 10, 7, 2]
#lf_cutoffs = [20]
for i in range(len(lf_cutoffs)):
    lf_color = np.array((160 + (i * 10), cval_2, cval_1)) / 255
    lf_color = np.append(lf_color, 1.0)
    lf_colors.append(lf_color)

deemg_colors = []
for i in range(n_deemg):
    deemg_color = np.array((0 + (i * 10), 140, 255)) / 255
    deemg_colors.append(deemg_color)


bar_colors = lf_colors + deemg_colors
bar_colors = [bar_colors] * n_groups
bar_colors = np.concatenate(bar_colors, axis=0).tolist()

group_pos = []
group_r2_mean = []
group_r2_sem = []
start_pos = 0
for i, col_ix in enumerate(group_ixs):
    bar_pos = np.arange(n_bars)
    bar_pos = bar_pos + start_pos
    start_pos = bar_pos[-1]
    group_pos.append(bar_pos + (i * group_spacing))
    group_r2_mean.append(r2_mean_opt[:, col_ix])
    group_r2_sem.append(r2_sem_opt[:, col_ix])

group_pos = np.concatenate(group_pos)
group_r2_mean = np.concatenate(group_r2_mean)
group_r2_sem = np.concatenate(group_r2_sem)

#makes sure that all the bars get colored
while len(bar_colors) < len(group_pos):
    bar_colors.extend(bar_colors)  
bar_colors = bar_colors[:len(group_pos)] 

#plotting bars
plt.figure(figsize=(6, 8))
h = plt.bar(
    group_pos,
    group_r2_mean,
    yerr=group_r2_sem,
    align="center",
    ecolor="black",
    capsize=2,
    color="purple" 
)

for i, bar in enumerate(h):
    group_index = i // n_bars  #using to find the group number
    if i % n_bars == 0:  #first bar of group (LFADS)
        bar.set_facecolor("#1E90FF")  #setting to blue
    else:
        bar.set_facecolor(bar_colors[i])  #use same color for bars



group_labels = ["Hip", "Knee", "Ankle"]
group_midpoints = [np.mean(group_pos[i * n_bars:(i + 1) * n_bars]) for i in range(n_groups)] #calculating where to put labesl on graph

for i, label in enumerate(group_labels):
    plt.text(
        group_midpoints[i],  
        0.9,                 
        label,               
        ha="center",         
        va="bottom",         
        fontsize=16,         
        fontweight="bold"    
    )

plt.ylim((0, 0.85))
plt.ylabel("Decoding Performance (VAF)")
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xticklabels([])
ax.set_xticks([])

# %%
######## making psths ###################

#TO DO: figure out why SEM goes away

channel_name = "TA"

plt.figure(figsize=(10,5))

dw = DataWrangler(lfads_data) #aligning to foot off time
dw.make_trial_data(
    name="foot_off",
    align_field="foot_off_time", 
    align_range=(-100, 250),
    ignored_trials=None,
    allow_overlap = True
)

trial_data = dw._d.trial_dfs["foot_off"]
df = trial_data[[("align_time", ""), ("trial_id", ""), ("lfads_rates", channel_name)]].copy() #extracting columns of interest
df.columns = ["align_time", "trial_id", channel_name]  #flattens columns for plotting


#plotting
trial_ids = df["trial_id"].unique() #each trial has a bunch of time points, but we want one of each
for trial_id in trial_ids:
    single_trial_data = df[df["trial_id"] == trial_id] #filters the df for only the rows where the trial ID matches the current trial_id in the loop (Trial 1: plot -100 to +250 ms -> one line, etc)
    plt.plot(
        single_trial_data["align_time"], #-> x-axis
        single_trial_data[channel_name], #-> y-axis
        alpha=0.5,  #make lines semi-transparent
        label=f"Trial {trial_id}" if trial_id == trial_ids[0] else None  #labeling first trial only
    )


plt.axvline(0, color="red", linestyle="--", label="Foot Off")
plt.title(f"All trial PSTHs for {channel_name}")
plt.xlabel("Align Time")
plt.ylabel("EMG Signal")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.figure(figsize=(10,5))


######plotting mean psths################
trial_conds = pd.Series("default", index=trial_data["trial_id"].unique())
psth = PSTH(trial_conds)

# Select the relevant columns
# df = trial_data[[("align_time", ""), ("trial_id", ""), ("lfads_rates", channel_name)]].copy()
# df.columns = ["align_time", "trial_id", channel_name]  # flatten locally for convenience

#group by align time
grouped = df.groupby("align_time")[channel_name]

#compute mean and sem
psth_mean = grouped.mean()
psth_sem = grouped.sem()

#plotting
plt.figure(figsize=(10,5))
plt.plot(psth_mean.index, psth_mean.values, label="Mean") #index represents the align time, values represent the mean EMG signal
plt.fill_between(
    psth_mean.index,
    psth_mean.values - psth_sem.values, #lower bound
    psth_mean.values + psth_sem.values, #higher bound
    alpha=0.3,
    label="SEM"
)
plt.axvline(0, color="red", linestyle="--", label="Foot Off")
plt.title(f"Trial averaged PSTH for {channel_name}")
plt.xlabel("Align Time")
plt.ylabel("EMG Signal")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
############################ making pcs ###########################

#params for lfads
APPLY_PCA = True
APPLY_PCA_FIELDS = [col for col in lfads_data.data.columns if col[0] == 'lfads_rates'] #uses lfads_rates columns for pca, aka the columns are each muscle run thorugh lfads


lfads_for_pca = lfads_data.data[APPLY_PCA_FIELDS] #takes only the data from lfads_rates

#remove nans
nan_mask = lfads_for_pca.isnull().any(axis=1)
lfads_for_pca_no_null = lfads_for_pca[~nan_mask]

#applying pcas
NUM_PCS = 3
scaler = StandardScaler() #standardizes the features by removing the mean and scaling them to unit variance
pca_model = PCA(n_components=NUM_PCS)
pcs = pca_model.fit_transform(scaler.fit_transform(lfads_for_pca_no_null)) #standardizes the data before applying pca and applies pca

#creating a df for principal components
pcs_df = pd.DataFrame(pcs, index=lfads_for_pca_no_null.index, columns=[f"PC{i+1}" for i in range(NUM_PCS)]) #index ensures that the pcs_df has the same index as lfads_for_pca_no_null

# Merge with the main df
lfads_data.data = pd.concat([lfads_data.data, pcs_df], axis=1)

#extracting foot on and off times
foot_off_times = dataset.trial_info["foot_off_time"]  
foot_strike_times = dataset.trial_info["foot_strike_time"]  

#creating a df to hold stance/swing labels
stance_swing_labels = pd.Series(index=lfads_data.data.index, dtype="object")

for trial_id in dataset.trial_info.index: #loops through each trial in the dataset
    foot_off = foot_off_times[trial_id] #marks foot off time per trial
    foot_strike = foot_strike_times[trial_id] #marks foot strike time per trial
    
    if trial_id + 1 in dataset.trial_info.index: #checks for next trial in index
        next_foot_strike = foot_strike_times[trial_id + 1]
    else:
        next_foot_strike = foot_strike + pd.Timedelta(seconds=1)  #adds one second to last value to handle edge case
    
    stance_mask = (lfads_data.data.index >= foot_strike) & (lfads_data.data.index < foot_off) #defines all timepoints between foot strike and next foot off as stance
    stance_swing_labels[stance_mask] = "stance"
    
    swing_mask = (lfads_data.data.index > foot_off) & (lfads_data.data.index <= next_foot_strike) #defines all timepoints between foot off and next foot strike as swing
    stance_swing_labels[swing_mask] = "swing"

#mapping colors
color_map = {"stance": "darkorange", "swing": "black"}
colors = stance_swing_labels.map(color_map) #connects map to color labels
colors_pca = colors.loc[pcs_df.index] #filters the colors series to include only the indices that are present in pcs_df.index

#plotting
fig = go.Figure(data=[
    go.Scatter3d(
        x=pcs_df["PC1"],
        y=pcs_df["PC2"],
        z=pcs_df["PC3"],
        mode="markers",
        marker=dict(
            size=3,
            color=colors_pca.values, 
            colorscale="Viridis",
            opacity=0.8
        )
    )
])

fig.update_layout(
    title="3D PCA Subspace of LFADS Rates",
    scene=dict(
        xaxis_title="PC1",
        yaxis_title="PC2",
        zaxis_title="PC3"
    ),
    margin=dict(l=0, r=0, b=0, t=40) #adjusts margins for better visibility
)

fig.show()

# %%
