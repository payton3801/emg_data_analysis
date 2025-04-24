# %%
import h5py
import numpy as np
import os
import sys

# %%
ds_path = "/snel/share/share/tmp/pbechef/Tresch/nwb_lfads/runs/run_002/torch_input/lfads_J10_s20_i0_emg_2.h5"

# %%
f = h5py.File(ds_path, 'r')

# %%
f.keys()