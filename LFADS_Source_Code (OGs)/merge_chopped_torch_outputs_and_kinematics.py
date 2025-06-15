"""
PURPOSE: Merge chopped torch outputs with original dataset (and kinematics if available)

REQUIREMENTS: lfads-torch model outputs, original dataset, interface object
                                                ^                 ^
                                                |_________________|
                                        (created in setup_lfads_datasets.py)

OUTPUTS: merged dataset object with original dataset and lfads outputs
"""

# %% INPUTS AND PATHS
import os
import pickle as pkl
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as colormap
from snel_toolkit.datasets.nwb import NWBDataset
import logging
import sys
import yaml
import dill
from analysis_utils import *

# %%
import analysis_utils
from importlib import reload
reload(analysis_utils)
from analysis_utils import *

# %%
logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# load YAML file
yaml_config_path = "../configs/lfads_dataset_cfg.yaml"
path_config, ld_cfg, merge_config, _ = load_cfgs(yaml_config_path)


# system inputs
run_date = path_config["RUN_DATE"] # 240108 first run, 240112 second run
expt_name = ld_cfg["NAME"] # Ex: "NP_AAV6-2_ReaChR_184500"
initials = path_config["INITIALS"] # Ex: "cw"
run_type = path_config["TYPE"] # Ex: "spikes"
chan_select = ld_cfg["ARRAY_SELECT"] # Ex: "ALL"
bin_size = ld_cfg["BIN_SIZE"] # Ex: 2
project_str = ld_cfg["PROJECT_STR"] # Ex: 2281. This is first defined in run_pbt script
run_name_mod = path_config["RUN_NAME_MOD"] # Ex: "_2" for second run

# create paths
ds_name = f"{expt_name}_{chan_select}_{run_type}_{str(bin_size)}"
base_name = f"binsize_{str(bin_size)}"
run_base_dir = f"/snel/share/runs/{project_str}_{run_type}{run_name_mod}/TORCH_lfads_{ds_name}/{run_date}_{project_str}_{run_type}_PBT_{initials}"
run_dir = os.path.join(run_base_dir,"best_model")
lfads_torch_outputs_path = os.path.join(run_dir,f"lfads_output_TORCH_lfads_{ds_name}.h5")
lfads_save_dir = f"/snel/share/share/derived/scpu_snel/nwb_lfads/runs/{base_name}/{expt_name}/datasets/"
#unchopped_ds_path = os.path.join(lfads_save_dir,"lfads_"+ds_name+"_unchopped.pkl")
interface_path = f"{lfads_save_dir}pkls/{ds_name}_interface.pkl"
DATA_FILE = os.path.join(lfads_save_dir, ds_name)

cache_dataset = f"/snel/share/share/derived/scpu_snel/nwb_lfads/runs/{base_name}/{expt_name}/datasets/pkls/lfads_{expt_name}_{chan_select}_{run_type}_{bin_size}_fulldataset.pkl"

tf2_original_h5 = f"/snel/share/share/derived/scpu_snel/nwb_lfads/runs/{base_name}/{expt_name}/datasets/lfads_{expt_name}_{chan_select}_{run_type}_{bin_size}.h5"


# %% LOAD CONTINUOUS DATA DF, MERGE WITH TORCH OUTPUTS  
with open(interface_path,'rb') as inf:
    interface = pkl.load(inf)

interface.merge_fields_map = merge_config

with open(cache_dataset,'rb') as inf:
    dataset = pkl.load(inf)

torch_outputs = h5py.File(lfads_torch_outputs_path)

# %% Load chop indices pertaining to training and validation; add to torch output obj if not present

train_inds, valid_inds = get_train_valid_inds(tf2_original_h5, torch_outputs, lfads_torch_outputs_path)

# %% Make full output df

data_dict = combine_train_valid_outputs(torch_outputs, train_inds, valid_inds, merge_config)
merged_df = interface.merge(data_dict, smooth_pwr=1)

# %%
# happens when not using the full data
if dataset.data.shape[0] != merged_df.shape[0]:
    # bunch of nans hanging at the end of the merged_df
    merged_df = merged_df.iloc[:dataset.data.shape[0]]
    # and has the index of the full data
    merged_df.index = dataset.data.index
# %% Merge with original dataset
merge_with_original_df(merged_df, dataset)
# %% in the case of data points being included multiple times, take the first one
dataset.data = dataset.data[~dataset.data.index.duplicated(keep='first')]
# %% smooth spikes, rates, factors

spike_smooth_width = 6
rate_smooth_width = 8
factor_smooth_width = 15

# fill na with 0 for lfads outputs due to chopping
dataset.smooth_spk(gauss_width = spike_smooth_width, name=f'smooth_{spike_smooth_width}', overwrite=False)

dataset.smooth_spk(signal_type='lfads_rates', gauss_width=rate_smooth_width, name=f'smooth_{rate_smooth_width}', overwrite=False)
dataset.data[f'lfads_rates_smooth_{rate_smooth_width}'] = dataset.data[f'lfads_rates_smooth_{rate_smooth_width}'].fillna(0)

dataset.smooth_spk(signal_type='lfads_factors', gauss_width=factor_smooth_width, name=f'smooth_{factor_smooth_width}', overwrite=False)
dataset.data[f'lfads_factors_smooth_{factor_smooth_width}'] = dataset.data[f'lfads_factors_smooth_{factor_smooth_width}'].fillna(0)

# %% load locomotion kinematics if applicable
# # NOTE: ASSUMES THE HDF5 FILES IN THE VIDEO DIRECTORY FOR DLC ARE THE DESIRED ONES

# if not hasattr(dataset,'cycles'):
#     print('adding cycles from a run with full cycle info')
#     if 'aav' in expt_name.lower():
#         with open('/snel/share/runs/aav_spikes_0/TORCH_lfads_NP_AAV6-2_ReaChR_184500_ALL_spikes_2/240717_aav_spikes_PBT_cw/best_model/lfads_NP_AAV6-2_ReaChR_184500_ALL_spikes_2_full_merged_output.pkl','rb') as f:
#             dataset_with_cycles = dill.load(f)
#         # load cycles df since we can't add it through kinematics notebook    
#         dataset.cycles = dataset_with_cycles.cycles
#         cw_kin_full = dataset_with_cycles.data.loc[dataset.data.index, ('kin_info_cw', slice(None))]
#         jm_kin_full = dataset_with_cycles.data.loc[dataset.data.index, ('kin_info_jm', slice(None))]
#         dataset.data = dataset.data.join(cw_kin_full,how='left')
#         dataset.data = dataset.data.join(jm_kin_full,how='left')
        
#     else:
#         raise NotImplementedError('trim only implemented for chloro')
# else:
#     kinematic_time_shift = .4 # s 

#     if ld_cfg["KINEMATICS_NAME"]:
#         add_kinematics_to_merged_outputs(dataset, kinematic_time_shift)

#     JERRY = False
#     if JERRY:
#         add_kinematics_to_merged_outputs_JERRY(dataset, kinematic_time_shift)

#         # need to fix the columns. issue likely in the add_kinematics_to_merged_outputs_JERRY function

# %% IF WE DELETED THE MODEL HDF5 GOD WHY (did for 2881 first mentioned in notion)
# load pickle 
# yaml_config_path = "../configs/lfads_dataset_cfg.yaml"
# dataset, BIN_SIZE = load_dataset_and_binsize(yaml_config_path)
# dataset.data.drop(columns=[('stim_kinematics', 'toe_x'), ('stim_kinematics', 'toe_y')], inplace=True)
# dataset.data.columns = pd.MultiIndex.from_tuples(dataset.data.columns)
# %% add stim kinematics
if 'aav' in expt_name.lower():
    stim_kin_path = "/snel/share/share/data/scpu_snel/NP_AAV6-2_ReaChR_1845_kinematics/Reflex Kinematics"
elif '2881' in expt_name:
    stim_kin_path = '/snel/share/share/data/scpu_snel/NP_TrpV1-ReaChR_2881_kinematics/'
# ^sorry
add_stim_kinematics_to_merged_outputs(dataset,stim_kin_path,200,900) #200,900 for 2881 #100,120 for aav

# %% add stim time offsets for actual stim to kinematics when relevant
if 'aav' in expt_name.lower():
    raise ValueError('dont run for chloro just hw so far')
with open(os.path.join(stim_kin_path,'TrpV1_2881_09-20-23_Stim_info.xlsx'),'rb') as f:
    stim_info = pd.read_excel(f)
stim_start_column = [np.NaN for _ in range(dataset.trial_info.shape[0])]
stim_start_column[:stim_info.shape[0]] = stim_info['Stim start']
dataset.trial_info['stim_offset'] = stim_start_column

# %% drop trials with no videos
if 'aav' in expt_name.lower():
    raise ValueError('dont run for chloro just hw so far')
dataset.trial_info.drop(index=4,inplace=True)


# %%
# save dataset to pickle
merged_full_output = os.path.join(run_dir, f"lfads_{expt_name}_{chan_select}_{run_type}_{bin_size}_full_merged_output.pkl")
with open(merged_full_output, "wb") as f:
    dill.dump(dataset, f, protocol=dill.HIGHEST_PROTOCOL, recurse=True)

# %%
