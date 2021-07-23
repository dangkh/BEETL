from braindecode.util import set_random_seeds, np_to_var, var_to_np
import matplotlib.pyplot as plt
from moabb.datasets import BNCI2014001, Cho2017, PhysionetMI
from moabb.paradigms import MotorImagery
import numpy as np
from numpy.random import RandomState
import pickle
import time
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import mne


subj = 1
for dataset in [BNCI2014001(), PhysionetMI(), Cho2017()]:
    data = dataset.get_data(subjects=[subj])
    ds_name = dataset.code
    ds_type = dataset.paradigm
    sess = 'session_T' if ds_name == "001-2014" else 'session_0'
    run = sorted(data[subj][sess])[0]
    ds_ch_names = data[subj][sess][run].info['ch_names']  # [0:22]
    ds_sfreq = data[subj][sess][run].info['sfreq']
    print("{} is an {} dataset, acquired at {} Hz, with {} electrodes\nElectrodes names: ".format(ds_name, ds_type, ds_sfreq, len(ds_ch_names)))
    print(ds_ch_names)
    print()