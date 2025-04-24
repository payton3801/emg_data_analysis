"""
This script takes the output of setup_lfads_datasets.py and converts it to a format that can be used as input to the LFADS torch model.
run in the data_prep directory with the command: python convert_tf2_output_to_torch_input.py
"""
# %%
import os
import sys
import h5py
import yaml
import numpy as np

# to load from pickled rds
lfads_save_dir = '/snel/share/share/tmp/pbechef/Tresch/nwb_lfads/runs/datasets'
cache_dir = '/snel/share/share/data/Tresch_gaitEMG/data/NWB/'
dataset_name = 'J10_s20_i0'
cached_path = os.path.join(cache_dir, dataset_name+'.nwb')
time_limit = None # seconds
# to load from tmp path of pickled rds (helpful if rds has already been preprocessed)
#cached_path = '/tmp/lwimala/jango_2016_w_lfads_rates.pkl'
lfads_dataset_cfg = [
    {
        'DATASET': {
            'NAME': dataset_name,
            'CONDITION_SEP_FIELD': None, # continuous
            'ALIGN_LIMS': None,
            'TIME_LIMIT': time_limit, # seconds
            'BIN_SIZE': 2,
            'EXCLUDE_TRIALS': [],
            'EXCLUDE_CONDITIONS': [],
            'EXCLUDE_CHANNELS': []
        }
    },
    {
        'CHOP_PARAMETERS': {
            'TYPE': 'emg',
            'DATA_FIELDNAME': 'emg',
            'USE_EXT_INPUT': False,
            'EXT_INPUT_FIELDNAME': '',
            'WINDOW': 200, #ms
            'OVERLAP': 50, #ms
            'MAX_OFFSET': 0,
            'RANDOM_SEED': 0,
            'CHOP_MARGINS': 0
        }
    }
]



#params based on loaded file
speed = 20
incline = 0

# load yaml config file
yaml_config_path = "/home/pbechef/emg_data_analysis/LFADS_Setup_Code/cfg_J10_s20_i0_emg_2.yaml" 
lfads_dataset_cfg = yaml.load(open(yaml_config_path), Loader=yaml.FullLoader)
#path_config = lfads_dataset_cfg["PATH_CONFIG"]
ld_cfg = lfads_dataset_cfg[0]["DATASET"]  # First dictionary in the list
chop_cfg = lfads_dataset_cfg[1]["CHOP_PARAMETERS"] 

expt_name = ld_cfg["NAME"]
#ARRAY_SELECT = ld_cfg["ARRAY_SELECT"]
#TYPE = path_config["TYPE"]
BIN_SIZE = ld_cfg["BIN_SIZE"]
#kin_name = ld_cfg["KINEMATICS_NAME"]

# %%
# -- paths
#base_name = f"binsize_{ld_cfg['BIN_SIZE']}"
#ds_base_dir = "/snel/share/share/derived/scpu_snel/NWB/"
tf2_torch_file = "/snel/share/share/tmp/pbechef/Tresch/nwb_lfads/runs/datasets/lfads_J10_s20_i0_emg_2.h5"

# Open the file
with h5py.File(tf2_torch_file, 'r') as dataset:
    # Print all group names
    train_encod_data = np.abs(dataset["train_data"][:])
    valid_encod_data = np.abs(dataset["valid_data"][:])
    train_recon_data = np.abs(dataset["train_data"][:])
    valid_recon_data = np.abs(dataset["valid_data"][:])
    train_inds = dataset["train_inds"][:]
    valid_inds = dataset["valid_inds"][:]
    if chop_cfg['USE_EXT_INPUT']:
        train_ext_input = dataset["train_ext_input"][:]
        valid_ext_input = dataset["valid_ext_input"][:]
# %%
torch_dataset_str =  'lfads_' + ld_cfg['NAME'] + '_' + chop_cfg['TYPE'] + '_' + str(ld_cfg['BIN_SIZE']) + '.h5'

kwargs = dict(dtype='float32', compression='gzip')
output_file_path_1 = f"/snel/share/share/tmp/pbechef/Tresch/nwb_lfads/runs/run_002/torch_input/{torch_dataset_str}"

with h5py.File(output_file_path_1, 'w') as h5f:
    h5f.create_dataset('train_encod_data', data=train_encod_data, **kwargs)
    h5f.create_dataset('valid_encod_data', data=valid_encod_data, **kwargs)
    h5f.create_dataset('train_recon_data', data=train_recon_data, **kwargs)
    h5f.create_dataset('valid_recon_data', data=valid_recon_data, **kwargs)
    h5f.create_dataset('train_inds', data=train_inds, **kwargs)
    h5f.create_dataset('valid_inds', data=valid_inds, **kwargs)
    if chop_cfg['USE_EXT_INPUT']:
        h5f.create_dataset('train_ext_input', data=train_ext_input, **kwargs)
        h5f.create_dataset('valid_ext_input', data=valid_ext_input, **kwargs)
    print(f"File saved: {torch_dataset_str}")


# %%
