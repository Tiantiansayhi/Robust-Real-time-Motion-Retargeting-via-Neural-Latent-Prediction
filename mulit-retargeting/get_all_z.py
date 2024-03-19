import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.transforms as transforms
from torch_geometric.data import Batch, DataListLoader
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
import h5py
import argparse
import logging
import time
import os
import copy
from datetime import datetime

import dataset
from dataset import Normalize, parse_all
from models import model_retarget
from models.loss import CollisionLoss, JointLimitLoss, RegLoss
from train import train_epoch
from utils.config import cfg
from utils.util import create_folder

# Argument parse
parser = argparse.ArgumentParser(description='Inference with trained model')
parser.add_argument('--cfg', default='configs/inference/yumi.yaml', type=str, help='Path to configuration file')
args = parser.parse_args()

# Configurations parse
cfg.merge_from_file(args.cfg)
cfg.freeze()
# print(cfg)

# Create folder
create_folder(cfg.OTHERS.LOG)
create_folder(cfg.OTHERS.SUMMARY)

# Create logger & tensorboard writer
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.FileHandler(os.path.join(cfg.OTHERS.LOG, "{:%Y-%m-%d_%H-%M-%S}.log".format(datetime.now()))), logging.StreamHandler()])
logger = logging.getLogger()
writer = SummaryWriter(os.path.join(cfg.OTHERS.SUMMARY, "{:%Y-%m-%d_%H-%M-%S}".format(datetime.now())))

# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    # Load data
    pre_transform = transforms.Compose([Normalize()])
    #load test data : key by key to inference
    h5_file = h5py.File(cfg.INFERENCE.MOTION.SOURCE, 'r')
    keys = h5_file.keys()
    for key in keys:
        if '语句' in key: 
            print('Skipping' + key)
            continue     
        print(key)
        test_data = parse_all(filename=cfg.INFERENCE.MOTION.SOURCE, selected_key=key)
        test_data = [pre_transform(data) for data in test_data]
        test_loader = [test_data]
        test_target = sorted([target for target in getattr(dataset, cfg.DATASET.TEST.TARGET_NAME)(root=cfg.DATASET.TEST.TARGET_PATH)], key=lambda target : target.skeleton_type)

        # Create model
        model = getattr(model_retarget, cfg.MODEL.NAME)().to(device)

        # Load checkpoint
        if cfg.MODEL.CHECKPOINT is not None:
            model.load_state_dict(torch.load(cfg.MODEL.CHECKPOINT))

        # store initial z   
        model.eval()
        z_all = []
        for batch_idx, data_list in enumerate(test_loader):
            for target_idx, target in enumerate(test_target):
                # forward
                z = model.encode(Batch.from_data_list(data_list).to(device)).detach()
                z.requires_grad = True
                z_all.append(z)

        if cfg.INFERENCE.H5.BOOL:
            hf = h5py.File(os.path.join(cfg.INFERENCE.H5.PATH, key), 'w')
            g1 = hf.create_group('group1')
            if z_all:
                # z = torch.cat(z_all, dim=0).view(len(test_data), -1, 64).detach().cpu().numpy() # [T, 50, 64]
                z = torch.cat(z_all, dim=0).detach().cpu().numpy()
                # print(z)
                # print('z', z.shape)
                # print(z.shape)
                z_arm = z[:len(test_data)*14, :].reshape(len(test_data), -1, 64)
                # print(z_arm)
                z_l_hand = z[len(test_data)*14:len(test_data)*32, :].reshape(len(test_data), -1, 64)  
                # print(z_l_hand.shape)    
                z_r_hand = z[len(test_data)*32:, :].reshape(len(test_data), -1, 64)    
                # print(z_r_hand.shape)        
                g1.create_dataset('arm', data = z_arm)
                g1.create_dataset('l_hand', data = z_l_hand)
                g1.create_dataset('r_hand', data = z_r_hand)
                g1.create_dataset('z', data = z)
            hf.close()
            # print(z.shape)
            # print(z_arm.shape)
            # print(z_hand.shape)      
            print('Target H5 file saved!')