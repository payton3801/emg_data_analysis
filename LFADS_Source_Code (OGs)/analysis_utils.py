import h5py
import typing 
import numpy as np
import pandas as pd
import yaml 
from snel_toolkit.datasets.nwb import NWBDataset
import os
import dill
import glob
from scipy.io import loadmat
from scipy.signal import resample_poly
import re

# load YAML file
yaml_config_path = "../configs/lfads_dataset_cfg.yaml"
lfads_dataset_cfg = yaml.load(open(yaml_config_path), Loader=yaml.FullLoader)

path_config = lfads_dataset_cfg["PATH_CONFIG"]
ld_cfg = lfads_dataset_cfg["DATASET"]
merge_config = lfads_dataset_cfg["MERGE_PARAMETERS"]
cycle_config = lfads_dataset_cfg["REJECTED_CYCLES"]
BIN_SIZE = ld_cfg["BIN_SIZE"]

def add_stim_kinematics_to_merged_outputs(dataset: NWBDataset, kin_dir_name: str, time_before_stim_ms: int = 200, time_after_stim_ms: int = 1000):
    # check params
    assert time_before_stim_ms <= 200
    
    # grab stim kinematic files
    extensions = {fn.split('.')[-1] for fn in os.listdir(kin_dir_name)}
    if 'mat' in extensions:
        kin_files = glob.glob(os.path.join(kin_dir_name, '*.mat'))
        ft = 'mat'
    elif 'csv' in extensions:
        kin_files = glob.glob(os.path.join(kin_dir_name, '*.csv'))
        ft = 'csv'
    else: 
        raise ValueError(f"Kinematic files in {kin_dir_name} must be .mat or .csv")

    # necessary because jerry labels some videos with 0 indexing and some with 1
    trial_pattern = r"SV(\d+)"    
    video_ids = [int(re.search(trial_pattern, kin_file).group(1)) for kin_file in kin_files]
    min_video_id = min(video_ids) # 0 or 1
    if min_video_id == 0:
        pass
    elif min_video_id == 1:
        video_ids = [v_id - 1 for v_id in video_ids]
    else:  
        raise ValueError("Video IDs must start at 0 or 1")

    full_kin_data = np.full((len(dataset.data),2),np.nan)
    # iterate through kinematic files, trimming 200 frames from the beginning that are not part of the trial
    for kin_file, v_id in zip(kin_files, video_ids):
        kin_data = loadmat(kin_file)['mmData'][:,:2] if ft == 'mat' else pd.read_csv(kin_file).iloc[:,:2].values
        samples_pre = (200-time_before_stim_ms)//BIN_SIZE
        samples_post = (200+time_after_stim_ms)//BIN_SIZE
        kin_data = resample_poly(kin_data, 1, BIN_SIZE, axis=0)
        kin_data = kin_data[samples_pre:samples_post, :]
        kin_data[:,1] = -kin_data[:,1]
        print(kin_data)
        print(kin_data.shape)

        
        if "stimulation" in dataset.trial_info.event_type.values:
            start_time = dataset.trial_info[dataset.trial_info.event_type == "stimulation"].start_time.iloc[v_id]
        else:
            start_time = dataset.trial_info.start_time.iloc[v_id]
        start_ix = dataset.data.index.get_loc(start_time, method='nearest') - time_before_stim_ms//BIN_SIZE
        full_kin_data[start_ix:start_ix+kin_data.shape[0]] = kin_data

    dataset.add_continuous_data(full_kin_data, 'stim_kinematics', chan_names=['toe_x', 'toe_y'])


def load_cfgs(yaml_config_path: str) -> typing.Tuple[typing.Dict[str, str], typing.Dict[str, str], typing.Dict[str, str]]:
    return path_config, ld_cfg, merge_config, cycle_config

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

def merge_with_original_df(merged_df: pd.DataFrame, dataset: NWBDataset):
    for key in merged_df.columns.levels[0].to_list():
        if key == "lfads_rates":
            chan_names = dataset.data['spikes'].columns.values
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

def get_event_start_stop_ix(win_len_ms: int, pre_buffer_ms: int, event_id: int, dataset: NWBDataset) -> typing.Tuple[int, int]:
    win_len = win_len_ms / BIN_SIZE
    event_start_time = dataset.trial_info.iloc[event_id].start_time - pd.to_timedelta(pre_buffer_ms, unit="ms")
    start_ix = dataset.data.index.get_loc(event_start_time, method='nearest')
    stop_ix = int(start_ix + win_len)
    
    return start_ix, stop_ix

def load_dataset_and_binsize(yaml_config_path: str) -> typing.Tuple[NWBDataset, int]:
    path_config, ld_cfg, merge_config, _ = load_cfgs(yaml_config_path)

    # system inputs
    run_date = path_config["RUN_DATE"] # 240108 first run, 240112 second run
    expt_name = ld_cfg["NAME"] # Ex: "NP_AAV6-2_ReaChR_184500"
    initials = path_config["INITIALS"] # Ex: "cw"
    run_type = path_config["TYPE"] # Ex: "spikes"
    chan_select = ld_cfg["ARRAY_SELECT"] # Ex: "ALL"
    bin_size = ld_cfg["BIN_SIZE"] # Ex: 2
    project_str = ld_cfg["PROJECT_STR"] # Ex: 2281. This is first defined in run_pbt script
    run_name_mod = path_config["RUN_NAME_MOD"]

    ds_name = f"{expt_name}_{chan_select}_{run_type}_{str(bin_size)}"
    base_name = f"binsize_{ld_cfg['BIN_SIZE']}"
    run_base_dir = f"/snel/share/runs/{project_str}_{run_type}{run_name_mod}/TORCH_lfads_{ds_name}/{run_date}_{project_str}_{run_type}_PBT_{initials}"
    run_dir = os.path.join(run_base_dir,"best_model")

    merged_full_output = os.path.join(run_dir, f"lfads_{expt_name}_{chan_select}_{run_type}_{bin_size}_full_merged_output.pkl")
    with open(merged_full_output, "rb") as f:
        dataset = dill.load(f)

    return dataset, bin_size

def overwrite_dataset_and_pickle_df(yaml_config, dataset):

    path_config, ld_cfg, merge_config, _ = load_cfgs(yaml_config_path)

    # system inputs
    run_date = path_config["RUN_DATE"] # 240108 first run, 240112 second run
    expt_name = ld_cfg["NAME"] # Ex: "NP_AAV6-2_ReaChR_184500"
    initials = path_config["INITIALS"] # Ex: "cw"
    run_type = path_config["TYPE"] # Ex: "spikes"
    chan_select = ld_cfg["ARRAY_SELECT"] # Ex: "ALL"
    bin_size = ld_cfg["BIN_SIZE"] # Ex: 2
    project_str = ld_cfg["PROJECT_STR"] # Ex: 2281. This is first defined in run_pbt script
    run_name_mod = path_config["RUN_NAME_MOD"]

    ds_name = f"{expt_name}_{chan_select}_{run_type}_{str(bin_size)}"
    base_name = f"binsize_{ld_cfg['BIN_SIZE']}"
    run_base_dir = f"/snel/share/runs/{project_str}_{run_type}{run_name_mod}/TORCH_lfads_{ds_name}/{run_date}_{project_str}_{run_type}_PBT_{initials}"
    run_dir = os.path.join(run_base_dir,"best_model")

    #overwrite the dataset
    merged_full_output = os.path.join(run_dir, f"lfads_{expt_name}_{chan_select}_{run_type}_{bin_size}_full_merged_output.pkl")
    with open(merged_full_output, "wb") as f:
        dill.dump(dataset, f, protocol=dill.HIGHEST_PROTOCOL, recurse=True)

    #save the df
    cycle_df_path = os.path.join(run_dir, f"lfads_{expt_name}_{chan_select}_{run_type}_{bin_size}_cycles.csv")
    dataset.cycles.to_csv(cycle_df_path)

def add_kinematics_to_merged_outputs(dataset: NWBDataset, time_shift: float):
    kin_name = ld_cfg['KINEMATICS_NAME']
    kin_paths = sorted(glob.glob('/snel/share/share/derived/scpu_snel/DLC/{}/videos/*.h5'.format(kin_name)))
    print("kin name:", kin_name)
    print("kin paths:",kin_paths)
    kinematic_data = [pd.read_hdf(kin_path) for kin_path in kin_paths]


    kin_df = pd.DataFrame(np.nan, index=np.arange(len(dataset.data)), columns=kinematic_data[0].columns)


    for i in range(len(kinematic_data)):
        start_time = dataset.trial_info.iloc[i].start_time - pd.Timedelta(time_shift, unit='s')
        kinematic_dataframe = kinematic_data[i]

        start_ix = dataset.data.index.get_loc(start_time, method='nearest')

        # fill data from kinematic dataframe into kin_df
        kin_df.loc[start_ix:start_ix+len(kinematic_dataframe)-1] = kinematic_dataframe.values

    # drop the columns called 'likelihood' at level 2 of the multiindex
    mask = kin_df.columns.get_level_values(2) != 'likelihood'
    kin_df = kin_df.loc[:, mask]
    kin_df.columns = kin_df.columns.droplevel(0)
    kin_df.index = dataset.data.index
    col_names = ['mtarsal_x','mtarsal_y','ankle_back_x','ankle_back_y','ankle_side_x','ankle_side_y','tail_base_x','tail_base_y']
    kin_df.columns = pd.MultiIndex.from_tuples(
        [('kin_info_cw',level1) for level1 in col_names]
    )
    dataset.data = pd.concat([dataset.data, kin_df], axis=1)

def add_kinematics_to_merged_outputs_JERRY(dataset: NWBDataset, time_shift: float):
    point_dict = {
        'pt1_cam1_X': 'toe_x',
        'pt1_cam1_Y': 'toe_y',
        'pt2_cam1_X': 'ankle_x',
        'pt2_cam1_Y': 'ankle_y',
        'pt3_cam1_X': 'knee_x',
        'pt3_cam1_Y': 'knee_y',
        'pt4_cam1_X': 'hip_x',
        'pt4_cam1_Y': 'hip_y',
        'pt5_cam1_X': 'iliac_crest_x',
        'pt5_cam1_Y': 'iliac_crest_y'
    }
    
    jerry_kin_path = '/snel/share/share/data/scpu_snel/NP_AAV6-2_ReaChR_1845_kinematics'
    jerry_kin_csvs = sorted(glob.glob(os.path.join(jerry_kin_path, '*.csv')))
    
    kinematic_data = [pd.read_csv(kin_csv) for kin_csv in jerry_kin_csvs]

    kin_df = pd.DataFrame(np.nan, index=np.arange(len(dataset.data)), columns=kinematic_data[0].columns)
    for i in range(len(kinematic_data)):
        start_time = dataset.trial_info.iloc[i].start_time - pd.Timedelta(time_shift, unit='s')
        kinematic_dataframe = kinematic_data[i]

        start_ix = dataset.data.index.get_loc(start_time, method='nearest')

        # fill data from kinematic dataframe into kin_df
        kin_df.loc[start_ix:start_ix+len(kinematic_dataframe)-1] = kinematic_dataframe.values
    kin_df.index = dataset.data.index
    kin_df.columns = pd.MultiIndex.from_tuples(
        [('kin_info_jm',point_dict[level1]) for level1 in kin_df.columns]
    )
    dataset.data = pd.concat([dataset.data, kin_df], axis=1)

def find_kinematic_ranges(kinematic_data):
    is_tracking = np.where(~kinematic_data.isna())[0]
    change_ixs = np.where(np.diff(is_tracking) != 1)
    start_ixs = [is_tracking[0]]
    end_ixs = []
    for change_ix in change_ixs[0]:
        end_ixs.append(is_tracking[change_ix])
        start_ixs.append(is_tracking[change_ix+1])
    end_ixs.append(is_tracking[-1])
    return start_ixs, end_ixs

def bin_cycles(cycle_df: pd.DataFrame, bin_size: float, criterion: str = "stance") -> typing.Dict[int, pd.DataFrame]:
    # drop final cycles of bouts in stance case
    cycle_df = cycle_df.copy().dropna(subset=[f'{criterion}_duration'])
    cycle_times = cycle_df[f'{criterion}_duration'].dt.total_seconds()
    print(len(cycle_times))
    # compute bin edges
    min_edge = cycle_times.min()
    max_edge = cycle_times.max() + bin_size
    bin_edges = np.arange(min_edge, max_edge, bin_size)

    # bin the cycles
    bin_ids = np.searchsorted(bin_edges, cycle_times, side='right') - 1
    bin_dfs = [cycle_df[bin_ids == i] for i in range(len(bin_edges) - 1)]

    bin_dfs_dict = {upper_bin_bound: bin_df for upper_bin_bound, bin_df in zip(bin_edges[1:], bin_dfs)}

    return bin_dfs_dict
