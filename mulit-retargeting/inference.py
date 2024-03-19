import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.transforms as transforms
from torch_geometric.data import Batch, DataListLoader
from torch_geometric.data import Data
from tensorboardX import SummaryWriter
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
from models import model
from models.loss import CollisionLoss, JointLimitLoss, RegLoss, SmoothnessLoss
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
print(cfg)

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

    if cfg.INFERENCE.MOTION.KEY:
        # inference single key
        print('Inference single key {}'.format(cfg.INFERENCE.MOTION.KEY))
        test_data = dataset.parse_all(filename = cfg.INFERENCE.MOTION.SOURCE, selected_key=cfg.INFERENCE.MOTION.KEY, sliding_window = cfg.HYPER.PERIOD)
        test_data = [pre_transform(data) for data in test_data]
        print(len(test_data))
        test_loader = [test_data]
        test_target = sorted([target for target in getattr(dataset, cfg.DATASET.TEST.TARGET_NAME)(root=cfg.DATASET.TEST.TARGET_PATH)], key=lambda target : target.skeleton_type)
    
    # Create model
    model = getattr(model, cfg.MODEL.NAME)(cfg.HYPER.PERIOD).to(device)
    
    seq_length = cfg.HYPER.PERIOD

    # Load checkpoint
    if cfg.MODEL.CHECKPOINT is not None:
        model.load_state_dict(torch.load(cfg.MODEL.CHECKPOINT))

    # store initial z
    encode_start_time = time.time()
    model.eval()
    z_all = []
    for batch_idx, data_list in enumerate(test_loader):
        for target_idx, target in enumerate(test_target):
            # forward
            z = model.encode(Batch.from_data_list(data_list).to(device)).detach()
            z.requires_grad = True
            z_all.append(z)
    encode_end_time = time.time()
    print('encode time {} ms'.format((encode_end_time - encode_start_time)*1000))

#=========================optimization ===============================    
    
    # Create loss criterion
    # end effector loss
    ee_criterion = nn.MSELoss() if cfg.LOSS.EE else None
    # vector similarity loss
    vec_criterion = nn.MSELoss() if cfg.LOSS.VEC else None
    # collision loss
    col_criterion = CollisionLoss(cfg.LOSS.COL_THRESHOLD) if cfg.LOSS.COL else None
    # joint limit loss
    lim_criterion = JointLimitLoss() if cfg.LOSS.LIM else None
    # end effector orientation loss
    ori_criterion = nn.MSELoss() if cfg.LOSS.ORI else None
    # finger similarity loss
    fin_criterion = nn.MSELoss() if cfg.LOSS.FIN else None
    # regularization loss
    reg_criterion = RegLoss() if cfg.LOSS.REG else None
    #smoothness loss
    smo_criterion = SmoothnessLoss(cfg.LOSS.SMOOTHNESS) if cfg.LOSS.SMO else None

    # Create optimizer
    optimizer = optim.Adam(z_all, lr=cfg.HYPER.LEARNING_RATE)

    best_loss = float('Inf')
    best_z_all = copy.deepcopy(z_all)
    best_cnt = 0


    # latent optimization
    op_start_time = time.time()

    for epoch in range(cfg.HYPER.EPOCHS):
        train_loss = train_epoch(model, ee_criterion, vec_criterion, col_criterion, lim_criterion, ori_criterion, fin_criterion, reg_criterion, smo_criterion, optimizer, test_loader, test_target, epoch, seq_length, logger, cfg.OTHERS.LOG_INTERVAL, writer, device, z_all)
        if cfg.INFERENCE.MOTION.KEY:
            # Save model
            if train_loss > best_loss:
                best_cnt += 1
            else:
                best_cnt = 0
                best_loss = train_loss
                best_z_all = copy.deepcopy(z_all)
            if best_cnt == 5:
                logger.info("Interation Finished")
                break
    op_end_time = time.time()
    print('optimization time {} ms'.format((op_end_time - op_start_time)*1000)) 

    if cfg.INFERENCE.MOTION.KEY:
        # store final results
        decode_start_time = time.time()
        model.eval()
        pos_all = []
        ang_all = []
        l_hand_ang_all = []
        r_hand_ang_all = []
        l_hand_pos_all = []
        r_hand_pos_all = []
        for batch_idx, data_list in enumerate(test_loader):
            for target_idx, target in enumerate(test_target):
                
                # fetch target
                target_list = []
                for data in data_list:
                    if seq_length > 1:
                        target_list.append(Data(x=target.x.repeat(seq_length, 1),
                                                edge_index_col=target.edge_index,
                                                edge_index=torch.cat([target.edge_index + target.num_nodes*t for t in range(seq_length)], dim=1),
                                                edge_attr=target.edge_attr.repeat(seq_length, 1),
                                                temporal_edge_index=target.temporal_edge_index,
                                                t=seq_length,
                                                skeleton_type=target.skeleton_type,
                                                topology_type=target.topology_type,
                                                ee_mask_col=target.ee_mask,
                                                ee_mask=target.ee_mask.repeat(seq_length, 1),
                                                sh_mask=target.sh_mask.repeat(seq_length, 1),
                                                el_mask=target.el_mask.repeat(seq_length, 1),
                                                root_dist=target.root_dist.repeat(seq_length, 1),
                                                shoulder_dist=target.shoulder_dist.repeat(seq_length, 1),
                                                elbow_dist=target.elbow_dist.repeat(seq_length, 1),
                                                parent=target.parent.repeat(seq_length),
                                                offset=target.offset.repeat(seq_length, 1),
                                                lower=target.lower.repeat(seq_length, 1),
                                                upper=target.upper.repeat(seq_length, 1),
                                                hand_x = target.hand_x.repeat(seq_length, 1),
                                                hand_edge_index = torch.cat([target.hand_edge_index + target.hand_num_nodes*t for t in range(seq_length)], dim=1),
                                                hand_edge_attr = target.hand_edge_attr.repeat(seq_length, 1),
                                                hand_temporal_edge_index=target.hand_temporal_edge_index,
                                                hand_ee_mask = target.hand_ee_mask.repeat(seq_length, 1),
                                                hand_el_mask = target.hand_el_mask.repeat(seq_length, 1),
                                                hand_root_dist = target.hand_root_dist.repeat(seq_length, 1),
                                                hand_elbow_dist = target.hand_elbow_dist.repeat(seq_length, 1),
                                                hand_parent = target.hand_parent.repeat(seq_length),
                                                hand_offset = target.hand_offset.repeat(seq_length, 1),
                                                hand_axis = target.hand_axis.repeat(seq_length, 1),
                                                hand_lower = target.hand_lower.repeat(seq_length, 1),
                                                hand_upper = target.hand_upper.repeat(seq_length, 1)
                                                ))
                    else:
                        target_list.append(target)
                # fetch z
                z = best_z_all[batch_idx]
                # forward
                _, target_ang, _, _, target_global_pos, l_hand_ang, l_hand_pos, r_hand_ang, r_hand_pos = model.decode(z, Batch.from_data_list(target_list).to(z.device))

                decode_end_time = time.time()
                print('decode time {} ms'.format((decode_end_time - decode_start_time)*1000))
                if target_global_pos is not None and target_ang is not None:
                    pos_all.append(target_global_pos)
                    ang_all.append(target_ang)
                if l_hand_ang is not None and r_hand_ang is not None:
                    l_hand_ang_all.append(l_hand_ang)
                    r_hand_ang_all.append(r_hand_ang)
                if l_hand_pos is not None and r_hand_pos is not None:
                    l_hand_pos_all.append(l_hand_pos)
                    r_hand_pos_all.append(r_hand_pos)
                # if z is not None:
                    # z_all.append(z)
                    # print(np.array(z_all).shape) 

        if cfg.INFERENCE.H5.BOOL:
            # hf = h5py.File(os.path.join(cfg.INFERENCE.H5.PATH, cfg.INFERENCE.MOTION.KEY), 'w')
            hf = h5py.File(os.path.join(cfg.INFERENCE.H5.PATH, 'test.h5'), 'w')
            g1 = hf.create_group('group1')
            if pos_all and ang_all:
                pos = torch.cat(pos_all, dim=0).view(len(test_data), cfg.HYPER.PERIOD, -1, 3).detach().cpu().numpy() # [T, window, joint_num, xyz]
                ang = torch.cat(ang_all, dim=0).view(len(test_data), cfg.HYPER.PERIOD, -1).detach().cpu().numpy()
                # print(pos.shape)
                g1.create_dataset('l_joint_pos', data=pos[:, :, :7, :])
                g1.create_dataset('r_joint_pos', data=pos[:, :, 7:, :])
                g1.create_dataset('l_joint_angle', data=ang[:, :, :7])
                g1.create_dataset('r_joint_angle', data=ang[:, :, 7:])
            if l_hand_ang_all and r_hand_ang_all:
                l_hand_angle = torch.cat(l_hand_ang_all, dim=0).view(len(test_data), cfg.HYPER.PERIOD, -1).detach().cpu().numpy()
                r_hand_angle = torch.cat(r_hand_ang_all, dim=0).view(len(test_data), cfg.HYPER.PERIOD, -1).detach().cpu().numpy()
                l_hand_pos = torch.cat(l_hand_pos_all, dim=0).view(len(test_data), cfg.HYPER.PERIOD, -1, 3).detach().cpu().numpy()
                r_hand_pos = torch.cat(r_hand_pos_all, dim=0).view(len(test_data), cfg.HYPER.PERIOD, -1, 3).detach().cpu().numpy()
                
                # remove zeros
                l_hand_angle = np.concatenate([l_hand_angle[:, :, 1:3], l_hand_angle[:, :, 4:6], l_hand_angle[:, :, 7:9], l_hand_angle[:, :, 10:12], l_hand_angle[:, :, 13:17]], axis=2)
                r_hand_angle = np.concatenate([r_hand_angle[:, :, 1:3], r_hand_angle[:, :, 4:6], r_hand_angle[:, :, 7:9], r_hand_angle[:, :, 10:12], r_hand_angle[:, :, 13:17]], axis=2)
                l_hand_pos = np.concatenate([l_hand_pos[:, :, 1:3, :], l_hand_pos[:, :, 4:6, :], l_hand_pos[:, :, 7:9, :], l_hand_pos[:, :, 10:12, :], l_hand_pos[:, :, 13:17, :]], axis=2)
                r_hand_pos = np.concatenate([r_hand_pos[:, :, 1:3, :], r_hand_pos[:, :, 4:6, :], r_hand_pos[:, :, 7:9, :], r_hand_pos[:, :, 10:12, :], r_hand_pos[:, :, 13:17, :]], axis=2)
                # print(l_hand_angle.shape)
                g1.create_dataset('l_glove_angle', data=l_hand_angle)
                g1.create_dataset('r_glove_angle', data=r_hand_angle)
                g1.create_dataset('l_glove_pos', data=l_hand_pos)
                g1.create_dataset('r_glove_pos', data=r_hand_pos)
            hf.close()
            print('Target H5 file saved!') 
