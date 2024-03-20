#inference all keys to test
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.transforms as transforms
from torch_geometric.data import Batch, DataListLoader
from scipy.spatial.transform import Rotation as R
from models.kinematics import ForwardKinematicsURDF, ForwardKinematicsAxis
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import h5py
import argparse
import logging
import os
from datetime import datetime
import time
import datadeal_inf
import dataset
from utils.config import cfg
from utils.util import create_folder
from matplotlib.pyplot import MultipleLocator
import models.model_retarget as model_retarget
import models.model_pre as model_predict
from utils.urdf2graph import yumi2graph

clock = str(datetime.now())
clock = clock.replace(' ', '_')
clock = clock.replace(':', '_')
clock = clock.replace('-', '_')
clock = clock.replace('.', '_')
model_save_path = '/home/wtt/Data/pre_in_latent_mult/result/' + clock
os.mkdir(model_save_path) 

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

def position1(t, pos_1, pos_2, pos_3, pos_4, pos_5, wrist_pos_g_n, wrist_pos_ddq, model_save_path, name):
# def position1(t, pos_1, wrist_pos_g_n, wrist_pos_ddq, model_save_path, name):
    fig, axs = plt.subplots(nrows=3, ncols=1, constrained_layout=True)
    axs[0].set_title('left_arm_position')

    axs[0].plot(t, pos_1[0, :, 0], '--', c='r', label='pos_1',linewidth=1)
    axs[0].plot(t, pos_2[0, :, 0], '--', c='g', label='pos_2',linewidth=1)
    axs[0].plot(t, pos_3[0, :, 0], '--', c='b', label='pos_3',linewidth=1)
    axs[0].plot(t, pos_4[0, :, 0], '--', c='pink', label='pos_4',linewidth=1)
    axs[0].plot(t, pos_5[0, :, 0], '--', c='c', label='pos_5',linewidth=1)
    axs[0].plot(t, wrist_pos_g_n[:,0], c='k', label='human',linewidth=1)
    axs[0].plot(t, wrist_pos_ddq[0, :, 0], c='orange', label='ddq',linewidth=1)


    axs[1].plot(t, pos_1[0, :, 1], '--', c='r', label='pos_1',linewidth=1)
    axs[1].plot(t, pos_2[0, :, 1], '--', c='g', label='pos_2',linewidth=1)
    axs[1].plot(t, pos_3[0, :, 1], '--', c='b', label='pos_3',linewidth=1)
    axs[1].plot(t, pos_4[0, :, 1], '--', c='pink', label='pos_4',linewidth=1)
    axs[1].plot(t, pos_5[0, :, 1], '--', c='c', label='pos_5',linewidth=1)
    axs[1].plot(t, wrist_pos_g_n[:,1], c='k', label='human',linewidth=1)
    axs[1].plot(t, wrist_pos_ddq[0, :, 1], c='orange', label='ddq',linewidth=1)

    axs[2].plot(t, pos_1[0, :, 2], '--', c='r', label='pos_1',linewidth=1)
    axs[2].plot(t, pos_2[0, :, 2], '--', c='g', label='pos_2',linewidth=1)
    axs[2].plot(t, pos_3[0, :, 2], '--', c='b', label='pos_3',linewidth=1)
    axs[2].plot(t, pos_4[0, :, 2], '--', c='pink', label='pos_4',linewidth=1)
    axs[2].plot(t, pos_5[0, :, 2], '--', c='c', label='pos_5',linewidth=1)
    axs[2].plot(t, wrist_pos_g_n[:,2], c='k', label='human',linewidth=1)
    axs[2].plot(t, wrist_pos_ddq[0, :, 2], c='orange', label='ddq',linewidth=1)

    axs[0].legend(loc='upper right')
    plt.savefig(model_save_path + '/' + name + '.png')

def position2(t, pos_1, pos_2, pos_3, pos_4, pos_5, wrist_pos_g_n, wrist_pos_ddq,  model_save_path, name):
# def position2(t, pos_1, wrist_pos_g_n, wrist_pos_ddq, model_save_path, name):
    fig, axs = plt.subplots(nrows=3, ncols=1, constrained_layout=True)
    axs[0].set_title('right_arm_position')
    axs[0].plot(t, pos_1[1, :, 0], '--', c='r', label='pos_1',linewidth=1)
    axs[0].plot(t, pos_2[1, :, 0], '--', c='g', label='pos_2',linewidth=1)
    axs[0].plot(t, pos_3[1, :, 0], '--', c='b', label='pos_3',linewidth=1)
    axs[0].plot(t, pos_4[1, :, 0], '--', c='pink', label='pos_4',linewidth=1)
    axs[0].plot(t, pos_5[1, :, 0], '--', c='c', label='pos_5',linewidth=1)
    axs[0].plot(t, wrist_pos_g_n[:,3], c='k', label='human',linewidth=1)
    axs[0].plot(t, wrist_pos_ddq[1, :, 0], c='orange', label='ddq',linewidth=1)

    axs[1].plot(t, pos_1[1, :, 1], '--', c='r', label='pos_1',linewidth=1)
    axs[1].plot(t, pos_2[1, :, 1], '--', c='g', label='pos_2',linewidth=1)
    axs[1].plot(t, pos_3[1, :, 1], '--', c='b', label='pos_3',linewidth=1)
    axs[1].plot(t, pos_4[1, :, 1], '--', c='pink', label='pos_4',linewidth=1)
    axs[1].plot(t, pos_5[1, :, 1], '--', c='c', label='pos_5',linewidth=1)
    axs[1].plot(t, wrist_pos_g_n[:,4], c='k', label='human',linewidth=1)
    axs[1].plot(t, wrist_pos_ddq[1, :, 1], c='orange', label='ddq',linewidth=1)

    axs[2].plot(t, pos_1[1, :, 2], '--', c='r', label='pos_1',linewidth=1)
    axs[2].plot(t, pos_2[1, :, 2], '--', c='g', label='pos_2',linewidth=1)
    axs[2].plot(t, pos_3[1, :, 2], '--', c='b', label='pos_3',linewidth=1)
    axs[2].plot(t, pos_4[1, :, 2], '--', c='pink', label='pos_4',linewidth=1)
    axs[2].plot(t, pos_5[1, :, 2], '--', c='c', label='pos_5',linewidth=1)
    axs[2].plot(t, wrist_pos_g_n[:,5], c='k', label='human',linewidth=1)
    axs[2].plot(t, wrist_pos_ddq[1, :, 2], c='orange', label='ddq',linewidth=1)    

    axs[0].legend(loc='upper right')
    plt.savefig(model_save_path + '/' + name + '.png')

def ori1(t, pos_1, pos_2, pos_3, pos_4, pos_5, wrist_pos_g_n, model_save_path, name):
# def ori1(t, pos_1, wrist_pos_g_n, wrist_pos_ddq, model_save_path, name):
    fig, axs = plt.subplots(nrows=3, ncols=1, constrained_layout=True)
    axs[0].set_title('left_arm_orientation')

    axs[0].plot(t, pos_1[:, 0], '--', c='r', label='pos_1',linewidth=1)
    axs[0].plot(t, pos_2[:, 0], '--', c='g', label='pos_2',linewidth=1)
    axs[0].plot(t, pos_3[:, 0], '--', c='b', label='pos_3',linewidth=1)
    axs[0].plot(t, pos_4[:, 0], '--', c='pink', label='pos_4',linewidth=1)
    axs[0].plot(t, pos_5[:, 0], '--', c='c', label='pos_5',linewidth=1)
    axs[0].plot(t, wrist_pos_g_n[:,0], c='k', label='human',linewidth=1)
    # axs[0].plot(t, wrist_pos_ddq[:, 0], c='orange', label='ddq',linewidth=1)

    axs[1].plot(t, pos_1[:, 1], '--', c='r', label='pos_1',linewidth=1)
    axs[1].plot(t, pos_2[:, 1], '--', c='g', label='pos_2',linewidth=1)
    axs[1].plot(t, pos_3[:, 1], '--', c='b', label='pos_3',linewidth=1)
    axs[1].plot(t, pos_4[:, 1], '--', c='pink', label='pos_4',linewidth=1)
    axs[1].plot(t, pos_5[:, 1], '--', c='c', label='pos_5',linewidth=1)
    axs[1].plot(t, wrist_pos_g_n[:, 1], c='k', label='human',linewidth=1)
    # axs[1].plot(t, wrist_pos_ddq[:, 1], c='orange', label='ddq',linewidth=1)

    axs[2].plot(t, pos_1[:, 2], '--', c='r', label='pos_1',linewidth=1)
    axs[2].plot(t, pos_2[:, 2], '--', c='g', label='pos_2',linewidth=1)
    axs[2].plot(t, pos_3[:, 2], '--', c='b', label='pos_3',linewidth=1)
    axs[2].plot(t, pos_4[:, 2], '--', c='pink', label='pos_4',linewidth=1)
    axs[2].plot(t, pos_5[:, 2], '--', c='c', label='pos_5',linewidth=1)
    axs[2].plot(t, wrist_pos_g_n[:, 2], c='k', label='human',linewidth=1)
    # axs[2].plot(t, wrist_pos_ddq[:, 2], c='orange', label='ddq',linewidth=1)

    axs[0].legend(loc='upper right')
    plt.savefig(model_save_path + '/' + name + '.png')

def ori2(t, pos_1, pos_2, pos_3, pos_4, pos_5, wrist_pos_g_n, model_save_path, name):
# def ori2(t, pos_1, wrist_pos_g_n, wrist_pos_ddq, model_save_path, name):
    fig, axs = plt.subplots(nrows=3, ncols=1, constrained_layout=True)

    axs[0].set_title('right_arm_orientation')

    axs[0].plot(t, pos_1[:, 0], c='r', label='pos_1',linewidth=1)
    axs[0].plot(t, pos_2[:, 0], c='g', label='pos_2',linewidth=1)
    axs[0].plot(t, pos_3[:, 0], c='b', label='pos_3',linewidth=1)
    axs[0].plot(t, pos_4[:, 0], c='pink', label='pos_4',linewidth=1)
    axs[0].plot(t, pos_5[:, 0], c='c', label='pos_5',linewidth=1)
    axs[0].plot(t, wrist_pos_g_n[:,3], c='k', label='human',linewidth=1)
    # axs[0].plot(t, wrist_pos_ddq[:, 0], c='orange', label='ddq',linewidth=1)

    axs[1].plot(t, pos_1[:, 1], c='r', label='pos_1',linewidth=1)
    axs[1].plot(t, pos_2[:, 1], c='g', label='pos_2',linewidth=1)
    axs[1].plot(t, pos_3[:, 1], c='b', label='pos_3',linewidth=1)
    axs[1].plot(t, pos_4[:, 1], c='pink', label='pos_4',linewidth=1)
    axs[1].plot(t, pos_5[:, 1], c='c', label='pos_5',linewidth=1)
    axs[1].plot(t, wrist_pos_g_n[:, 4], c='k', label='human',linewidth=1)
    # axs[1].plot(t, wrist_pos_ddq[:, 1], c='orange', label='ddq',linewidth=1)

    axs[2].plot(t, pos_1[:, 2], c='r', label='pos_1',linewidth=1)
    axs[2].plot(t, pos_2[:, 2], c='g', label='pos_2',linewidth=1)
    axs[2].plot(t, pos_3[:, 2], c='b', label='pos_3',linewidth=1)
    axs[2].plot(t, pos_4[:, 2], c='pink', label='pos_4',linewidth=1)
    axs[2].plot(t, pos_5[:, 2], c='c', label='pos_5',linewidth=1)
    axs[2].plot(t, wrist_pos_g_n[:, 5], c='k', label='human',linewidth=1)
    # axs[2].plot(t, wrist_pos_ddq[:, 2], c='orange', label='ddq',linewidth=1)
    
    axs[0].legend(loc='upper right')
    plt.savefig(model_save_path + '/' + name + '.png')

def human_demonstration(filename, selected_key):
    h5_file = h5py.File(filename, 'r')
    if selected_key is None:
        keys = h5_file.keys()
    else:
        keys = [selected_key]
    for key in keys:
        if '语句' in key and selected_key is None: # skip 带有语句的数据
            print('Skipping'+key)
            continue
        # position data
        l_shoulder_pos = h5_file[key + '/l_up_pos'][:]
        r_shoulder_pos = h5_file[key + '/r_up_pos'][:]
        l_elbow_pos = h5_file[key + '/l_fr_pos'][:]
        r_elbow_pos = h5_file[key + '/r_fr_pos'][:]
        l_wrist_pos = h5_file[key + '/l_hd_pos'][:]
        r_wrist_pos = h5_file[key + '/r_hd_pos'][:]
        # quaternion data
        l_shoulder_qua = h5_file[key + '/l_up_quat'][:]
        r_shoulder_qua = h5_file[key + '/r_up_quat'][:]
        l_elbow_qua = h5_file[key + '/l_fr_quat'][:]
        r_elbow_qua = h5_file[key + '/r_fr_quat'][:]
        l_wrist_qua = h5_file[key + '/l_hd_quat'][:]
        r_wrist_qua = h5_file[key + '/r_hd_quat'][:]
        # rotation matrix data
        l_shoulder_quat = R.from_quat(h5_file[key + '/l_up_quat'][:])
        r_shoulder_quat = R.from_quat(h5_file[key + '/r_up_quat'][:])
        l_elbow_quat = R.from_quat(h5_file[key + '/l_fr_quat'][:])
        r_elbow_quat = R.from_quat(h5_file[key + '/r_fr_quat'][:])
        l_wrist_quat = R.from_quat(h5_file[key + '/l_hd_quat'][:])
        r_wrist_quat = R.from_quat(h5_file[key + '/r_hd_quat'][:])
        l_shoulder_matrix = l_shoulder_quat.as_matrix()
        r_shoulder_matrix = r_shoulder_quat.as_matrix()
        l_elbow_matrix = l_elbow_quat.as_matrix()
        r_elbow_matrix = r_elbow_quat.as_matrix()
        l_wrist_matrix = l_wrist_quat.as_matrix()
        r_wrist_matrix = r_wrist_quat.as_matrix()
        # euler data
        l_shoulder_euler = R.from_matrix(l_shoulder_matrix).as_euler('zyx', degrees=False)
        r_shoulder_euler = R.from_matrix(r_shoulder_matrix).as_euler('zyx', degrees=False)
        l_elbow_euler = R.from_matrix(l_elbow_matrix).as_euler('zyx', degrees=False)
        r_elbow_euler = R.from_matrix(r_elbow_matrix).as_euler('zyx', degrees=False)
        l_wrist_euler = R.from_matrix(l_wrist_matrix).as_euler('zyx', degrees=False)
        r_wrist_euler = R.from_matrix(r_wrist_matrix).as_euler('zyx', degrees=False)
        # glove data
        l_glove_pos = h5_file[key + '/l_glove_pos'][:]
        r_glove_pos = h5_file[key + '/r_glove_pos'][:]
        # insert zero for root
        total_frames = l_glove_pos.shape[0]
        l_glove_pos = np.concatenate([np.zeros((total_frames, 1, 3)), l_glove_pos], axis=1)
        r_glove_pos = np.concatenate([np.zeros((total_frames, 1, 3)), r_glove_pos], axis=1)
        # print(l_glove_pos.shape, r_glove_pos.shape)
        # switch dimensions
        l_glove_pos = np.stack([-l_glove_pos[..., 2], -l_glove_pos[..., 1], -l_glove_pos[..., 0]], axis=-1)
        r_glove_pos = np.stack([-r_glove_pos[..., 2], -r_glove_pos[..., 1], -r_glove_pos[..., 0]], axis=-1)
        #position 
        pos = np.stack((l_shoulder_pos, r_shoulder_pos, l_elbow_pos, r_elbow_pos, l_wrist_pos, r_wrist_pos), axis=0) #[6, N, 3]
        # euler 
        euler = np.stack((l_shoulder_euler, r_shoulder_euler, l_elbow_euler, r_elbow_euler, l_wrist_euler, r_wrist_euler), axis=0) #[6, N, 3]
        #rotation
        rot = np.stack((l_shoulder_matrix, r_shoulder_matrix, l_elbow_matrix, r_elbow_matrix, l_wrist_matrix, r_wrist_matrix), axis=0) #[6, N, 3, 3]
        #quaternion 
        qua = np.stack((l_shoulder_qua, r_shoulder_qua, l_elbow_qua, r_elbow_qua, l_wrist_qua, r_wrist_qua), axis=0) #[6, N, 4]
        #glove
    
    return pos, euler, qua, rot

def human_deal(pos, qua):
    #pos:[6,N,3], euler:[6,N,4], qua::[6,N,4], rot:[6,N,3,3]
    ee_p_h = []
    el_p_h = []
    ee_r_h = [] 
    el_r_h = []
    sh_p_h = []
    sh_r_h = []
    t = pos.shape[1]
    for i in range(t):
        ee_p = np.concatenate((pos[4,i,:], pos[5,i,:]), axis = 0) 
        ee_r = np.concatenate((qua[4,i,:], qua[5,i,:]), axis = 0)
        el_p = np.concatenate((pos[2,i,:], pos[3,i,:]), axis = 0)
        el_r = np.concatenate((qua[2,i,:], qua[3,i,:]), axis = 0)
        sh_p = np.concatenate((pos[0,i,:], pos[1,i,:]), axis = 0) 
        sh_r = np.concatenate((qua[0,i,:], qua[1,i,:]), axis = 0)
        ee_p_h.append(ee_p)
        el_p_h.append(el_p)
        ee_r_h.append(ee_r) 
        el_r_h.append(el_r)
        sh_p_h.append(sh_p) 
        sh_r_h.append(sh_r)
    return np.array(ee_p_h), np.array(el_p_h), np.array(sh_p_h), np.array(ee_r_h), np.array(el_r_h), np.array(sh_r_h),  #[N,6]

def encoder(model, data, n):
    # store initial z   
    model.eval()
    z_all = []
    for batch_idx, data_list in enumerate(data):
        # forward
        z = model.encode(Batch.from_data_list(data_list).to(device)).detach()
        z.requires_grad = True
        z_all.append(z) 
    z = torch.cat(z_all, dim=0)
    z_arm = z[:n*14, :].reshape(n, -1, 64)
    z_l_hand = z[n*14:n*32, :].reshape(n, -1, 64)  
    z_r_hand = z[n*32:, :].reshape(n, -1, 64)   
    return z_arm, z_l_hand, z_r_hand, z_arm[-1], z_l_hand[-1], z_r_hand[-1]

def decoder(model, arm_z, hand_z, test_target, n):
    period = n
    arm_ang_all = []
    arm_pos_all = []
    arm_rot_all = []
    l_hand_ang_all = []
    r_hand_ang_all = []
    l_hand_pos_all = []
    r_hand_pos_all = []

    z = torch.cat([arm_z, hand_z], dim=0)
    model.eval()
    for target_idx, target in enumerate(test_target):
        target_list = [target for i in range(period)]
        _, target_ang, _, target_rot, target_global_pos, target_l_hand_ang, target_l_hand_pos, target_r_hand_ang, target_r_hand_pos = model.decode(z, Batch.from_data_list(target_list).to(z.device))
        
        if target_ang is not None and target_global_pos is not None:
            arm_ang_all.append(target_ang)
            arm_pos_all.append(target_global_pos)
        if target_rot is not None:
            arm_rot_all.append(target_rot)
        if target_l_hand_ang is not None and target_r_hand_ang is not None:
            l_hand_ang_all.append(target_l_hand_ang)
            r_hand_ang_all.append(target_r_hand_ang)
        if target_l_hand_pos is not None and target_r_hand_pos is not None:
            l_hand_pos_all.append(target_l_hand_pos)
            r_hand_pos_all.append(target_r_hand_pos) 
        
    
    if arm_ang_all and arm_pos_all:
        arm_ang = torch.cat(arm_ang_all, dim=0).view(period, -1).detach().cpu().numpy()          #[b, 14]
        arm_pos = torch.cat(arm_pos_all, dim=0).view(period, -1, 3).detach().cpu().numpy()       #[b, 14, 3]

    if arm_rot_all:
        arm_rot = torch.cat(arm_rot_all, dim=0).view(period, -1, 3, 3).detach().cpu().numpy()     #[b, 14, 3, 3]

    if l_hand_ang_all and l_hand_pos_all:
        l_hand_ang = torch.cat(l_hand_ang_all, dim=0).view(period, -1).detach().cpu().numpy()       
        l_hand_pos = torch.cat(l_hand_pos_all, dim=0).view(period, -1, 3).detach().cpu().numpy()
        #remove zeros  
        l_hand_angle = np.concatenate([l_hand_ang[:,1:3],l_hand_ang[:,4:6],l_hand_ang[:,7:9],l_hand_ang[:,10:12],l_hand_ang[:,13:17]], axis=1)

    
    if r_hand_ang_all and r_hand_pos_all:
        r_hand_ang = torch.cat(r_hand_ang_all, dim=0).view(period, -1).detach().cpu().numpy()        
        r_hand_pos = torch.cat(r_hand_pos_all, dim=0).view(period, -1, 3).detach().cpu().numpy() 
        #remove zeros 
        r_hand_angle = np.concatenate([r_hand_ang[:,1:3],r_hand_ang[:,4:6],r_hand_ang[:,7:9],r_hand_ang[:,10:12],r_hand_ang[:,13:17]], axis=1)
            
    return arm_ang, arm_pos, arm_rot, l_hand_angle, l_hand_pos, r_hand_angle, r_hand_pos

def data_joint(filename, selected_key):
    h5_file = h5py.File(filename, 'r')
    if selected_key is None:
        keys = h5_file.keys()
    else:
        keys = [selected_key]
    for key in keys:
        q_list = []
        if '语句' in key and selected_key is None: #skip 带有语句的数据
            print('Skipping' + key)
            continue       
        #glove angle
        l_glove_angle = h5_file[key + '/l_glove_angle'][:]
        r_glove_angle = h5_file[key + '/r_glove_angle'][:]
        #glove pos
        # l_glove_pos = h5_file[key + '/l_glove_pos'][:]
        # r_glove_pos = h5_file[key + '/r_glove_pos'][:]
        #joint angle
        l_joint_angle = h5_file[key + '/l_joint_angle'][:] #(n,7)
        r_joint_angle = h5_file[key + '/r_joint_angle'][:] 
        #joint pos
        l_joint_pos = h5_file[key + '/l_joint_pos'][:] #(n,7,3)
        r_joint_pos = h5_file[key + '/r_joint_pos'][:]
        
        joint_pos = np.concatenate((l_joint_pos, r_joint_pos), axis = 1)
        q = np.concatenate((l_joint_angle, r_joint_angle), axis = 1)   #[n, 14]

    return joint_pos, q

if __name__ == '__main__':
    #window
    n = 5
    
    # Load human data
    pre_transform = transforms.Compose([dataset.Normalize()])    
    h5_file = h5py.File(cfg.INFERENCE.MOTION.HUMAN, 'r')
    keys = h5_file.keys()
    for key in keys:
        if '语句' in key: 
            print('Skipping' + key)
            continue 
        test_data = dataset.parse_all(filename=cfg.INFERENCE.MOTION.HUMAN, selected_key=key)
        test_data = [pre_transform(data) for data in test_data]
        test_target = sorted([target for target in getattr(dataset, cfg.DATASET.TEST.TARGET_NAME)(root=cfg.DATASET.TEST.TARGET_PATH)], key=lambda target : target.skeleton_type)
        total_frames = len(test_data)
        print(total_frames)

        # Create model
        # predict model
        model_p = getattr(model_predict, cfg.MODEL.NAME)().to(device)
        if cfg.MODEL.CHECKPOINT is not None:
            model_p.load_state_dict(torch.load(cfg.MODEL.CHECKPOINT))

        # retarget model
        model_r = getattr(model_retarget, cfg.RETARGET.NAME)().to(device)
        if cfg.RETARGET.CHECKPOINT is not None:
            model_r.load_state_dict(torch.load(cfg.RETARGET.CHECKPOINT))

        arm_ang_all = []
        arm_pos_all = []
        arm_rot_all = []

        l_hand_ang_all = []
        r_hand_ang_all = []
        l_hand_pos_all = []
        r_hand_pos_all = []
        for i in range(total_frames-n*2):
            test_loader = [test_data[i:i+5]]
            # print(i)
            # print(test_loader)
            start_encoder_time = time.time()        

            #encode
            arm_z, l_hand_z, r_hand_z, arm_z_last, l_hand_z_last, r_hand_z_last = encoder(model_r, test_loader, n) #[n, 14, 64], [n, 18, 64], [n, 18, 64]
            # encoder(model_r, test_loader, n) #[n, 14, 64], [n, 18, 64], [n, 18, 64]

            # print(l_hand_z)
            end_encoder_time = time.time()
            # print('encoder time {} ms'.format((end_encoder_time - start_encoder_time)*1000))      
                
            #predict
            model_p.eval()
            pre_data = datadeal_inf.latent_data(arm_z, l_hand_z, r_hand_z)
            pre_loader = [pre_data]

            for batch_idx, data_list in enumerate(pre_loader):
                arm_z, hand_z = model_p(Batch.from_data_list(data_list).to(device))
            
            # end_predict_time = time.time()  
            # print('predict time {} ms'.format((end_predict_time - end_encoder_time)*1000))         
            
            # decode
            # arm_z = torch.cat((arm_z_last, arm_z), dim = 0)
            # hand_z = torch.cat((l_hand_z_last, r_hand_z_last, hand_z), dim = 0)
            arm_z = arm_z.view(-1, 64)
            hand_z = hand_z.view(-1, 64)

            arm_ang, arm_pos, arm_rot, l_hand_angle, l_hand_pos, r_hand_angle, r_hand_pos = decoder(model_r, arm_z, hand_z, test_target, n)
            
            end_decoder_time = time.time()
            # print('decoder time {} ms'.format((end_decoder_time - end_predict_time)*1000))
            # print('total time {} ms'.format((end_decoder_time - start_encoder_time)*1000))
            
            arm_ang_all.append(arm_ang)   #[len-10, 5, 14]
            arm_pos_all.append(arm_pos)   #[len-10, 5, 14, 3]
            arm_rot_all.append(arm_rot)   #[len-10, 5, 14, 3, 3]

            l_hand_ang_all.append(l_hand_angle)  #[len-10, 5, 12]
            r_hand_ang_all.append(r_hand_angle)  #[len-10, 5, 12] 
            l_hand_pos_all.append(l_hand_pos)  #[len-10, 5, 12, 3]
            r_hand_pos_all.append(r_hand_pos)  #[len-10, 5, 12, 3] 

        # arm_ang_all = np.array(arm_ang_all)  #[t, 5, 14]
        # length = arm_ang_all.shape[0]
        # for i in range(length):
        #     pos_i, euler_i, _ = fk.forward_k_array(arm_ang_all[i, 1, :])
        #     pos_tt.append(pos_i)
        #     euler_tt.append(euler_i)

        # arm_pos_all = np.array(arm_pos_all)
        # arm_rot_all = np.array(arm_rot_all)
        # l_hand_ang_all = np.array(l_hand_ang_all)
        # r_hand_ang_all = np.array(r_hand_ang_all)
        # print(arm_ang_all.shape)
        # print(l_hand_ang_all.shape)
        # print(r_hand_ang_all.shape)   

        # print(arm_ang_all[:, 0, :])
        # print(l_hand_ang_all[:, 0, :])
        # print(r_hand_ang_all[:, 0, :])

        if cfg.INFERENCE.H5.BOOL:
            hf = h5py.File(os.path.join(cfg.INFERENCE.H5.PATH, key), 'w')
            g1 = hf.create_group('group1')
            #arm
            g1.create_dataset('l_joint_pos', data = np.array(arm_pos_all)[:, :, :7, :])
            g1.create_dataset('r_joint_pos', data = np.array(arm_pos_all)[:, :, 7:, :])
            
            g1.create_dataset('l_joint_angle', data = np.array(arm_ang_all)[:, :, :7])
            g1.create_dataset('r_joint_angle', data = np.array(arm_ang_all)[:, :, 7:])
            
            g1.create_dataset('l_joint_rot', data = np.array(arm_rot_all)[:, :, :7, :, :])
            g1.create_dataset('r_joint_rot', data = np.array(arm_rot_all)[:, :, 7:, :, :])
            
            #hand         
            g1.create_dataset('l_glove_angle', data = np.array(l_hand_ang_all))
            g1.create_dataset('r_glove_angle', data = np.array(r_hand_ang_all))

            g1.create_dataset('l_glove_pos', data= np.array(l_hand_pos_all))
            g1.create_dataset('r_glove_pos', data= np.array(l_hand_pos_all))
            hf.close()
            print('Target H5 file saved!')  

