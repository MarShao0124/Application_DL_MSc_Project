# -*- coding: utf-8 -*-
# """
# main.py
#   Original Author: 2021.05.02. @chanwoo.park
#   Edited by MarShao0124 2025.03.06
#   run PaDiM algorithm
#   Reference:
#       Defard, Thomas, et al. "PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization."
#       arXiv preprint arXiv:2011.08785 (2020).
# """

############
#   IMPORT #
############
# 1. Built-in modules
import os
import argparse

# 2. Third-party modules
import random
import numpy as np
import tensorflow as tf

# 3. Own modules
from padim import padim

# For reproducibility, you can run scripts on CPU
# # Set CPU as available physical device
# my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
# tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')
#
# # To find out which devices your operations and tensors are assigned to
# tf.debugging.set_log_device_placement(True)

# For the reproducibility - please check https://github.com/NVIDIA/framework-determinism
os.environ['PYTHONHASHSEED'] = str(1)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'


################
#   Definition #
################
parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=10, type=int, help='What seed to use')
parser.add_argument("--rd", default=1000, type=int, help='Random sampling dimension')
parser.add_argument("--target", default='carpet', type=str, help="Which target to test")
parser.add_argument("--batch_size", default=32, type=int, help="What batch size to use")
parser.add_argument("--is_plot", default=True, type=bool, help="Whether to plot or not")
parser.add_argument("--net", default='eff', type=str, help="Which embedding network to use", choices=['eff', 'res'])
parser.add_argument("--data", default='mvtec', type=str, help="Which dataset to use", choices=['mvtec', 'visa'])

args = parser.parse_args()

# Target list for different datasets
"""
targets = {
    'mvtec': ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 
              'leather', 'metal_nut', 'pill', 'screw','tile', 'toothbrush', 
              'transistor', 'wood', 'zipper'],

    'visa' : ['pcb3', 'pipe_fryum', 'pcb4', 'pcb2', 'candle', 'fryum', 
              'macaroni2', 'capsules', 'pcb1', 'chewinggum', 'macaroni1', 
              'cashew']
}
"""

if __name__ == "__main__":
    opt = args
    opt.seed = 10
    opt.rd = 1000
    opt.target = 'pipe_fryum'
    opt.batch_size = 32
    opt.is_plot = True
    opt.net = 'eff'
    opt.data = 'visa'

    if opt.seed > -1:
        np.random.seed(opt.seed)
        random.seed(opt.seed)
        tf.random.set_seed(opt.seed)

    padim(category=opt.target, batch_size=opt.batch_size, rd=opt.rd, net_type=opt.net, is_plot=opt.is_plot, data=opt.data)