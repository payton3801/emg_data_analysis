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
yaml_config_path = "/home/pbechef/lfads-torch/configs/model/lfads_J10_s20_i0_emg_2.yaml"
#path_config, ld_cfg, merge_config, _ = load_cfgs(yaml_config_path)


# create paths -- need to check these
ds_name = "J10_s20_i0_emg_2"
#base_name = f"binsize_{str(bin_size)}"
run_base_dir = "/snel/share/share/tmp/pbechef/Tresch/nwb_lfads/runs/run_002/torch_output"
run_dir = os.path.join(run_base_dir,"best_model")
#lfads_torch_outputs_path = os.path.join(run_dir,f"lfads_output_TORCH_lfads_{ds_name}.h5")
lfads_torch_outputs_path = '/snel/share/share/tmp/pbechef/Tresch/nwb_lfads/runs/run_002/torch_output/best_model/lfads_output_lfads_J10_s20_i0_emg_2.h5'
lfads_save_dir = "/snel/share/share/tmp/pbechef/Tresch/nwb_lfads/runs/datasets"
#unchopped_ds_path = os.path.join(lfads_save_dir,"lfads_"+ds_name+"_unchopped.pkl")
interface_path = "/snel/share/share/tmp/pbechef/Tresch/nwb_lfads/runs/datasets/pkls/J10_s20_i0_emg_2_interface.pkl"
DATA_FILE = os.path.join(lfads_save_dir, ds_name)

#og dataset
cache_dataset = "/home/pbechef/emg_data_analysis/Data_Files/lfads_J10_s20_i0_emg_2.pkl"

# input passed in torch input file
tf2_original_h5 = "/snel/share/share/tmp/pbechef/Tresch/nwb_lfads/runs/datasets/lfads_J10_s20_i0_emg_2.h5"

MERGE_PARAMETERS = {
    'output_params': 'lfads_rates',
    'factors': 'lfads_factors',
    'gen_inputs': 'lfads_gen_inputs'
    }

# %% LOAD CONTINUOUS DATA DF, MERGE WITH TORCH OUTPUTS  

with open(interface_path,'rb') as inf:
    interface = pkl.load(inf)

interface.merge_fields_map = MERGE_PARAMETERS

with open(cache_dataset,'rb') as inf:
    dataset = pkl.load(inf)

torch_outputs = h5py.File(lfads_torch_outputs_path)

# %% Load chop indices pertaining to training and validation; add to torch output obj if not present

train_inds, valid_inds = get_train_valid_inds(tf2_original_h5, torch_outputs, lfads_torch_outputs_path)

# %% Make full output df

data_dict = combine_train_valid_outputs(torch_outputs, train_inds, valid_inds, MERGE_PARAMETERS)
merged_df = interface.merge(data_dict, smooth_pwr=1)

# %%
merge_with_original_df(merged_df, dataset)


# %%
#filter merged_df to only include lfads_rates, which is the channel info
#merged_df = merged_df['lfads_rates']
#dataset.data = merged_df

# %% smooth spikes, rates, factors

# spike_smooth_width = 6
# rate_smooth_width = 8
# factor_smooth_width = 15

# # fill na with 0 for lfads outputs due to chopping
# dataset.smooth_spk(gauss_width = spike_smooth_width, name=f'smooth_{spike_smooth_width}', overwrite=False)

# dataset.smooth_spk(signal_type='lfads_rates', gauss_width=rate_smooth_width, name=f'smooth_{rate_smooth_width}', overwrite=False)
# dataset.data[f'lfads_rates_smooth_{rate_smooth_width}'] = dataset.data[f'lfads_rates_smooth_{rate_smooth_width}'].fillna(0)

# dataset.smooth_spk(signal_type='lfads_factors', gauss_width=factor_smooth_width, name=f'smooth_{factor_smooth_width}', overwrite=False)
# dataset.data[f'lfads_factors_smooth_{factor_smooth_width}'] = dataset.data[f'lfads_factors_smooth_{factor_smooth_width}'].fillna(0)


# %%
# save dataset to pickle
merged_full_output = os.path.join(run_dir, f"lfads_J10_s20_i0_emg_2_full_merged_output.pkl")
with open(merged_full_output, "wb") as f:
    dill.dump(dataset, f, protocol=dill.HIGHEST_PROTOCOL, recurse=True)

# %%
