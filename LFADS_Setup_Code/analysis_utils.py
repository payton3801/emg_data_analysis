####################################################################
#This code provides utility functions for the merged_data script
####################################################################

import h5py
import typing 
import numpy as np
import pandas as pd
import yaml 
from snel_toolkit.datasets.nwb import NWBDataset


#this function grabs the training and validation data from the original dataset
def get_train_valid_inds(original_h5: str, torch_outputs: h5py._hl.files.File, lfads_torch_outputs_path: str) -> typing.Tuple[np.ndarray, np.ndarray]:
    original_h5_data = h5py.File(original_h5)
    train_inds = original_h5_data['train_inds'][()]
    valid_inds = original_h5_data['valid_inds'][()]
    # check if torch output already has train/valid inds
    if 'train_inds' not in torch_outputs.keys():
        with h5py.File(lfads_torch_outputs_path,'a') as torch_output_data:
            torch_output_data.create_dataset('train_inds',data=train_inds)
            torch_output_data.create_dataset('valid_inds',data=valid_inds)

    return train_inds, valid_inds

#this function combines training and validation into a single dataset
def combine_train_valid_outputs(torch_outputs: h5py._hl.files.File,
                                train_inds: np.ndarray, 
                                valid_inds: np.ndarray,
                                merge_config: typing.Dict[str, str]) ->\
                                typing.Dict[str, np.ndarray]:

    n_batch = train_inds.size + valid_inds.size
    data_dict = {} # dict with combined data
    for torch_name, snel_toolkit_name in merge_config.items():
        # key is torch names, val is what snel_toolkit name should be
        train_output = torch_outputs[f'train_{torch_name}'][()]
        valid_output = torch_outputs[f'valid_{torch_name}'][()]
        full_output = np.empty((n_batch, train_output.shape[1], train_output.shape[2]))
        full_output[train_inds,:,:] = train_output
        full_output[valid_inds,:,:] = valid_output
        data_dict[torch_name] = full_output
    
    return data_dict

#this function merges the original and LFADS datasets
def merge_with_original_df(merged_df: pd.DataFrame, dataset: NWBDataset):
    for key in merged_df.columns.levels[0].to_list():
        if key == "lfads_rates":
            chan_names = dataset.data['emg'].columns.values
        else: 
            chan_names = np.arange(merged_df[key].shape[1])
        if key in dataset.data.keys():
            dataset.data[key] = merged_df[key]
        else:
            dataset.add_continuous_data(
                merged_df[key].values,
                key,
                chan_names=chan_names,
            )


# def load_dataset_and_binsize(yaml_config_path: str) -> typing.Tuple[NWBDataset, int]:
#     path_config, ld_cfg, merge_config, _ = load_cfgs(yaml_config_path)

#     # system inputs
#     run_date = path_config["RUN_DATE"] # 240108 first run, 240112 second run
#     expt_name = ld_cfg["NAME"] # Ex: "NP_AAV6-2_ReaChR_184500"
#     initials = path_config["INITIALS"] # Ex: "cw"
#     run_type = path_config["TYPE"] # Ex: "spikes"
#     chan_select = ld_cfg["ARRAY_SELECT"] # Ex: "ALL"
#     bin_size = ld_cfg["BIN_SIZE"] # Ex: 2
#     project_str = ld_cfg["PROJECT_STR"] # Ex: 2281. This is first defined in run_pbt script
#     run_name_mod = path_config["RUN_NAME_MOD"]

#     ds_name = f"{expt_name}_{chan_select}_{run_type}_{str(bin_size)}"
#     base_name = f"binsize_{ld_cfg['BIN_SIZE']}"
#     run_base_dir = f"/snel/share/runs/{project_str}_{run_type}{run_name_mod}/TORCH_lfads_{ds_name}/{run_date}_{project_str}_{run_type}_PBT_{initials}"
#     run_dir = os.path.join(run_base_dir,"best_model")

#     merged_full_output = os.path.join(run_dir, f"lfads_{expt_name}_{chan_select}_{run_type}_{bin_size}_full_merged_output.pkl")
#     with open(merged_full_output, "rb") as f:
#         dataset = dill.load(f)

#     return dataset, bin_size








