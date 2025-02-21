
from pynwb import NWBFile
from pynwb import NWBHDF5IO
import numpy as np
import scipy.signal as signal
from scipy.io import loadmat
from datetime import datetime
from dateutil.tz import tzlocal
import matplotlib.pyplot as plt
from os import path

from pynwb import TimeSeries
from pynwb.behavior import Position
from pynwb.behavior import BehavioralTimeSeries
from pynwb import ProcessingModule
from nwb_convert.nwb_create_utils import create_multichannel_timeseries, apply_filt_to_multi_timeseries
from nwb_convert.filtering import apply_notch_filt, apply_butter_filt, apply_savgol_diff, resample_column, rectify

mat_path = 'J10_s20_i0_pref.mat'
save_path = '/home/pbechef/emg_data_analysis/NWB'


# %%
# load mat data
f = loadmat('J10_s10_i0_pref.mat')
=======


# get file name
file_id = f['fileName'][0].replace('_pref', '')

# get rat id
rat_id = 'ratId'
rat_name = f[rat_id][0]


def convert_datestr_to_datetime(f):
    date_str = f['settings']['day'][0][0][0]
    date_str = date_str.replace('pre_','')
    date_nums = np.array(date_str.split('_'), dtype=int).tolist()
    month_num = date_nums[0]
    day_num = date_nums[1]
    year_num = int(f"20{date_nums[2]}")

    date_time = datetime(year_num, month_num, day_num, tzinfo=tzlocal())

    return date_time

date_time  = convert_datestr_to_datetime(f)

def convert_names_to_list(f, name_field):
    names = f[name_field][0].tolist()
    names = [ name[0] for name in names ]
    return names

# field for names for joint angles, marker positions, emg
jnt_name_field = 'joints_names'
mk_name_field = 'mk_names'
emg_name_field = 'emg_names'

mk_names = convert_names_to_list(f, mk_name_field)
emg_names = convert_names_to_list(f, emg_name_field)
jnt_names = convert_names_to_list(f, jnt_name_field)

# emg named slightly differently depending on rat
if rat_name == 'J10':
    emg_raw_field = 'emg_full_fil' # Methods ,rat locomotion data
else:
    emg_raw_field = 'emg_full' # Methods ,rat locomotion data

# field for continuous data for joint angles, marker positions, emg
jnt_raw_field = 'joints_raw_nonSeg'
mk_raw_field = 'mk_raw_nonSeg'

# == trial info

ti_field = 'gait'
fs_field = 'fStrike_emgIdx'
fo_field = 'fOff_emgIdx'
cs_field = 'gaitCycleStart_emgIdx';
cg_field = 'goodCycle_ic';


# foot strike/off
fs_emg_ixs = f[ti_field][fs_field][0][0].squeeze()
fo_emg_ixs = f[ti_field][fo_field][0][0].squeeze()
# cycle start/end
cs_emg_ixs = f[ti_field][cs_field][0][0].squeeze()

# index before start of next cycle (i.e. foot strike) is = end of prev cycle
ce_emg_ixs = np.roll(cs_emg_ixs, -1)-1

# cycle good label
cg_label = f[ti_field][cg_field][0][0].squeeze()
n_cycles = cg_label.size
cs_emg_ixs = cs_emg_ixs[:n_cycles]
ce_emg_ixs = ce_emg_ixs[:n_cycles]
fs_emg_ixs = fs_emg_ixs[:n_cycles]
fo_emg_ixs = fo_emg_ixs[:n_cycles]

# emg good channel labels
emg_good_label = f['emg_goodCh'][0]

# === create multichannel timeseries for continuous EMG data, joint angles, and markers

# -- data sample rates and timestamps
t_emg = f['t_emg'][0]
dt_emg = np.unique(np.diff(t_emg).round(4))[0]
t_emg = dt_emg*np.arange(t_emg[0]/dt_emg, t_emg[-1]/dt_emg)
fs_emg = 1/dt_emg

t_kin = f['t_kin'][0]
dt_kin = np.unique(np.diff(t_kin).round(4))[0]
t_kin = dt_kin*np.arange(t_kin[0]/dt_kin, t_kin[-1]/dt_kin)
fs_kin = 1/dt_kin

# -- emg data
emg_data = f[emg_raw_field]
emg_gain = f['emg_sett']['gain'][0][0][0][0]

# scale raw emg by emg gain
emg_data = emg_data/emg_gain

# if we have continuous array, helper function covers some overhead
emg_mts = create_multichannel_timeseries('emg_raw', emg_names, emg_data,
                                      timestamps=t_emg, unit='mV')

# if not, build the multichannel timeseries manually using BehavioralTimeSeries

# -- marker data
mk_data = f[mk_raw_field].squeeze()

mk_mts = BehavioralTimeSeries(name='marker_pos')
for i, mk_name in enumerate(mk_names):
    mk_mts.create_timeseries(name=mk_name,
                             data=mk_data[i],
                             comments=f"columns=[{mk_name}]",
                             timestamps=t_kin,
                             unit='cm')
# -- joint angle data
jnt_data = f[jnt_raw_field]    
jnt_mts = BehavioralTimeSeries(name='joint_angles')
for i, jnt_name in enumerate(jnt_names):
    jnt_mts.create_timeseries(name=jnt_name,
                              data=jnt_data[i][0][:,0],
                              comments=f"columns=[{jnt_name}]",
                              timestamps=t_kin,
                              unit='deg')


# === NWBFile Step; create NWB file
nwbfile = NWBFile(session_description='rat locomotion on treadmill at constant speed',
                  identifier=file_id,
                  session_start_time=date_time,
                  file_create_date=datetime.now(tzlocal()),
                  lab='Tresch',
                  experimenter='Dr. Cristiano Alessandro')

    
# === NWBFile Step: add acquisition data

# add marker data
nwbfile.add_acquisition(mk_mts)
# add joint angles
nwbfile.add_acquisition(jnt_mts)
# add emg
nwbfile.add_acquisition(emg_mts)

# === NWBFile Step: add trial info

# add cycle information
nwbfile.add_trial_column(name='good_cycle', description='quality of cycle labeled by experimenter')
nwbfile.add_trial_column(name='good_channel', description='quality of emg channels labeled by experimenter')
nwbfile.add_trial_column(name='foot_off_time', description='timing of foot off')
nwbfile.add_trial_column(name='foot_strike_time', description='timing of foot strike (same as start_time)')
for i in range(n_cycles):
    cs_ix = cs_emg_ixs[i]
    ce_ix = ce_emg_ixs[i]
    fo_ix = fo_emg_ixs[i]
    fs_ix = fs_emg_ixs[i]
    assert fs_ix == cs_ix, 'these should match'
    foot_off_time = t_emg[fo_ix].round(4)
    foot_strike_time = t_emg[fs_ix].round(4)
    start_time = t_emg[cs_ix].round(4)
    end_time = t_emg[ce_ix].round(4)
    nwbfile.add_trial(start_time=start_time,
                      stop_time=end_time,
                      foot_off_time=foot_off_time,
                      foot_strike_time=foot_strike_time,
                      good_cycle=cg_label[i],
                      good_channel=emg_good_label)


# === data pre-processing

# -- emg filtering

# create processing module
emg_filt = nwbfile.create_processing_module('emg_filtering_module',
                                            "module to perform emg pre-processing from raw to rectified emg")

# extract acquisition data
raw_emg = nwbfile.acquisition['emg_raw']

# emg filtering parameters

# notch filtering
notch_cent_freq = [60, 120, 240, 300, 420]
notch_bw_freq = [2, 2, 2, 2, 2]

# high-pass filtering
hp_cutoff_freq = 65 # Hz

# 1) notch filter
notch_emg = apply_filt_to_multi_timeseries(raw_emg, apply_notch_filt, 'emg_notch',
                                           fs_emg, notch_cent_freq, notch_bw_freq)

# 2) high pass filter
hp_emg = apply_filt_to_multi_timeseries(notch_emg, apply_butter_filt, 'emg_hp',
                                        fs_emg, 'high', hp_cutoff_freq)

# 3) rectify
rect_emg = apply_filt_to_multi_timeseries(hp_emg, rectify, 'emg')

# add each step to processing module
emg_filt.add_container(notch_emg)
emg_filt.add_container(hp_emg)
emg_filt.add_container(rect_emg)


# -- resample kinematics (joint angles and marker data)

# create processing module
kin_resample = nwbfile.create_processing_module('kin_resampling_module',
                                                "module to perform resampling of kinematics to EMG sample rate")

# extract acquisition data
jnt_mts = nwbfile.acquisition['joint_angles']
mk_mts = nwbfile.acquisition['marker_pos']

# apply resampling
resample_jnt_p = apply_filt_to_multi_timeseries(jnt_mts, resample_column, 'raw_joint_ang_p', fs_emg, fs_kin, timestamps=t_emg)
resample_mk = apply_filt_to_multi_timeseries(mk_mts, resample_column, 'marker_p', fs_emg, fs_kin, timestamps=t_emg)

kin_resample.add_container(resample_jnt_p)
kin_resample.add_container(resample_mk)

# -- joint angle low-pass filtering
jnt_filt = nwbfile.create_processing_module('jnt_filtering_module',
                                            "module to perform lowpass filtering on joint angle positions prior to differentiation")

lf_cutoff_freq = 40 # Hz

raw_jnt_p_mts = nwbfile.processing['kin_resampling_module']['raw_joint_ang_p']
filt_jnt_p_mts = apply_filt_to_multi_timeseries(raw_jnt_p_mts, apply_butter_filt, 'joint_ang_p',
                                                fs_emg, 'low', lf_cutoff_freq)

jnt_filt.add_container(filt_jnt_p_mts)

jnt_diff = nwbfile.create_processing_module('jnt_diff_module',
                                            "module to perform joint angle position differentiation to get angular velocities and accelerations")

jnt_p_mts = nwbfile.processing['jnt_filtering_module']['joint_ang_p']

SG_WINDOW_LENGTH = 27
SG_POLYORDER = 5
DELTA = dt_emg
jnt_v_mts = apply_filt_to_multi_timeseries(jnt_p_mts, apply_savgol_diff, 'joint_ang_v',
                                           SG_WINDOW_LENGTH, SG_POLYORDER, 1, DELTA)

jnt_diff.add_container(jnt_v_mts)
jnt_a_mts = apply_filt_to_multi_timeseries(jnt_p_mts, apply_savgol_diff, 'joint_ang_a',
                                           SG_WINDOW_LENGTH, SG_POLYORDER, 2, DELTA)

jnt_diff.add_container(jnt_a_mts)

# pop original data at different sample rate out of NWB
nwbfile.acquisition.pop('joint_angles')
nwbfile.acquisition.pop('marker_pos')


save_fname = path.join(save_path, file_id + '.nwb')
# write processed file
with NWBHDF5IO(save_fname, 'w') as io:
    io.write(nwbfile)

