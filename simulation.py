#simulation for pybullet
import gym, yumi_gym
import pybullet as p
import numpy as np
import h5py
import time
import os

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++MPC+++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# simulation all key
""" 
pre = h5py.File('/home/wtt/Data/online_result/pre/stgcn/action/inference/yumi_intro.h5', 'r')
# pre = h5py.File('/home/wtt/Data/online_result/pre/stgcn/action/inference/yiyuan.h5', 'r')
keys = pre.keys()
for key in keys:
    if '语句' in key: 
        print('Skipping' + key)
        continue  
    l_glove_angle_pre = pre[key + '/l_hand'][:]   #[362, 6, 12]
    r_glove_angle_pre = pre[key + '/r_hand'][:]  

    l_joint_angle_pre = pre[key + '/l_arm'][:]   #[362, 6, 7]
    r_joint_angle_pre = pre[key + '/r_arm'][:]
    total_frames = l_joint_angle_pre.shape[0]

    env = gym.make('yumi-v1')
    observation = env.reset()
    env.render()
    joint_all = []
    joint_vel_all = []

    for t in range(total_frames):
        for i in range(30):
            # print(t, l_joint_angle.shape, l_joint_angle[t] * 180 / np.pi)
            # action = l_joint_angle_pre[t, 0, :].tolist() + r_joint_angle_pre[t, 0, :].tolist() + l_glove_angle_pre[t, 0, :].tolist() + r_glove_angle_pre[t, 0, :].tolist()+\
            #          l_joint_angle_pre[t, 1, :].tolist() + r_joint_angle_pre[t, 1, :].tolist() + l_glove_angle_pre[t, 1, :].tolist() + r_glove_angle_pre[t, 1, :].tolist()+\
            #          l_joint_angle_pre[t, 2, :].tolist() + r_joint_angle_pre[t, 2, :].tolist() + l_glove_angle_pre[t, 2, :].tolist() + r_glove_angle_pre[t, 2, :].tolist()+\
            #          l_joint_angle_pre[t, 3, :].tolist() + r_joint_angle_pre[t, 3, :].tolist() + l_glove_angle_pre[t, 3, :].tolist() + r_glove_angle_pre[t, 3, :].tolist()+\
            #          l_joint_angle_pre[t, 4, :].tolist() + r_joint_angle_pre[t, 4, :].tolist() + l_glove_angle_pre[t, 4, :].tolist() + r_glove_angle_pre[t, 4, :].tolist()+\
            #          l_joint_angle_pre[t, 5, :].tolist() + r_joint_angle_pre[t, 5, :].tolist() + l_glove_angle_pre[t, 5, :].tolist() + r_glove_angle_pre[t, 5, :].tolist()
            action = l_joint_angle_pre[t, 2, :].tolist() + r_joint_angle_pre[t, 2, :].tolist() + l_glove_angle_pre[t, 2, :].tolist() + r_glove_angle_pre[t, 2, :].tolist()

            observation, reward, done, info = env.step(action)
            state_joint = np.array(observation[:38])
            state_velocity = np.array(observation[38:])
            joint_all.append(state_joint)
            joint_vel_all.append(state_velocity)             
        

    n1 = np.array(joint_all).shape
    print(n1)
    env.close()

    # save data
    # hf = h5py.File(os.path.join('/home/wtt/Data/online_result/pre/stgcn/control/LQR/last/intro', key), 'w')
    hf = h5py.File(os.path.join('/home/wtt/Data/online_result/pre/stgcn/control/PID/30/2', key), 'w')
    g1 = hf.create_group('group1')
    g1.create_dataset('joint', data=np.array(joint_all))
    g1.create_dataset('joint_vel', data=np.array(joint_vel_all))
    hf.close()
    print('Target H5 file saved!') """


""" #simulation one key

#=====prediction file=======
pre = h5py.File('/home/wtt/Data/compare_result/action/ours/yiyuan/内科-neike', 'r') 
key = 'group1'

#=====label file=======
# ddq = h5py.File('/home/wtt/Data/online_result/ddq_without_op/action/inference/yiyuan/内科-neike', 'r') 

l_joint_angle_pre = pre[key + '/l_joint_angle'][:]   #[362, 6, 7]
r_joint_angle_pre = pre[key + '/r_joint_angle'][:]  

l_glove_angle_pre = pre[key + '/l_glove_angle'][:]   #[362, 12]
r_glove_angle_pre = pre[key + '/r_glove_angle'][:]
total_frames = l_joint_angle_pre.shape[0]

env = gym.make('yumi-v0')
observation = env.reset()
env.render()
joint_all = []
joint_vel_all = []

for t in range(total_frames):
    for i in range(15):    
        # action = l_joint_angle_pre[t, 4, :].tolist() + r_joint_angle_pre[t, 4, :].tolist() + l_glove_angle_pre[t+4, :].tolist() + r_glove_angle_pre[t+4, :].tolist()

        action = l_joint_angle_pre[t, 0, :].tolist() + r_joint_angle_pre[t, 0, :].tolist() + l_glove_angle_pre[t, 0, :].tolist() + r_glove_angle_pre[t, 0, :].tolist()+\
                 l_joint_angle_pre[t, 1, :].tolist() + r_joint_angle_pre[t, 1, :].tolist() + l_glove_angle_pre[t, 1, :].tolist() + r_glove_angle_pre[t, 1, :].tolist()+\
                 l_joint_angle_pre[t, 2, :].tolist() + r_joint_angle_pre[t, 2, :].tolist() + l_glove_angle_pre[t, 2, :].tolist() + r_glove_angle_pre[t, 2, :].tolist()+\
                 l_joint_angle_pre[t, 3, :].tolist() + r_joint_angle_pre[t, 3, :].tolist() + l_glove_angle_pre[t, 3, :].tolist() + r_glove_angle_pre[t, 3, :].tolist()+\
                 l_joint_angle_pre[t, 4, :].tolist() + r_joint_angle_pre[t, 4, :].tolist() + l_glove_angle_pre[t, 4, :].tolist() + r_glove_angle_pre[t, 4, :].tolist()


        observation, reward, done, info = env.step(action)
        state_joint = np.array(observation[:38])
        state_velocity = np.array(observation[38:])
        joint_all.append(state_joint)
        joint_vel_all.append(state_velocity)



n1 = np.array(joint_all).shape
print(n1)
env.close()

# save data
hf = h5py.File(os.path.join('/home/wtt/Data/compare_result/MPC/', 'neike.h5'), 'w')
g1 = hf.create_group('group1')
g1.create_dataset('joint', data=np.array(joint_all))
g1.create_dataset('joint_vel', data=np.array(joint_vel_all))
hf.close()
print('Target H5 file saved!')  """




#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++PID+++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" 
#simulation all key
pre = h5py.File('/home/wtt/Data/compare_result/action/ours/yumi_intro.h5', 'r')
# pre = h5py.File('/home/wtt/Data/pre_in_latent_mult/test_result/arm+hand/yiyuan.h5', 'r')

# =====label file=======
# ddq = h5py.File('/home/wtt/Data/online_result/ddq_without_op/action/inference/yumi_intro.h5', 'r') 
# ddq = h5py.File('/home/wtt/Data/online_result/ddq_without_op/action/inference/yiyuan.h5', 'r') 

frame = 0

keys = pre.keys()
for key in keys:
    if '语句' in key: 
        print('Skipping' + key)
        continue  
    print(key)
    l_glove_angle_pre = pre[key + '/l_hand'][:]   #[362, 6, 12]
    r_glove_angle_pre = pre[key + '/r_hand'][:]  

    l_joint_angle_pre = pre[key + '/l_arm'][:]    #[362, 6, 7]
    r_joint_angle_pre = pre[key + '/r_arm'][:]
    # print(l_joint_angle_pre.shape)
    total_frames = l_joint_angle_pre.shape[0]

    env = gym.make('yumi-v1')
    observation = env.reset()
    env.render()
    joint_all = []
    joint_vel_all = []

    for t in range(total_frames):
        for i in range(15):
            action = l_joint_angle_pre[t, frame, :].tolist() + r_joint_angle_pre[t, frame, :].tolist() + l_glove_angle_pre[t, frame, :].tolist() + r_glove_angle_pre[t, frame, :].tolist()
                    #  l_joint_angle_pre[t, 1, :].tolist() + r_joint_angle_pre[t, 1, :].tolist() + l_glove_angle_pre[t, 1, :].tolist() + r_glove_angle_pre[t, 1, :].tolist()
            observation, reward, done, info = env.step(action)
            state_joint = np.array(observation[:38])
            state_velocity = np.array(observation[38:])
            joint_all.append(state_joint)
            joint_vel_all.append(state_velocity)             
        

    n1 = np.array(joint_all).shape
    print(n1)
    env.close()

    # save data
    hf = h5py.File(os.path.join('/home/wtt/Data/state_result/15/update/intro/1', key), 'w')
    g1 = hf.create_group('group1')
    g1.create_dataset('joint', data=np.array(joint_all))
    g1.create_dataset('joint_vel', data=np.array(joint_vel_all))
    hf.close()
    print('Target H5 file saved!') """ 



#容错仿真
pre = h5py.File('/home/wtt/Data/compare_result/action-pre/ours/yiyuan/排队-paidui', 'r') 
key = 'group1'
start = 46
#=====label file=======
# ddq = h5py.File('/home/wtt/Data/online_result/ddq_without_op/action/inference/yiyuan/内科-neike', 'r') 

l_joint_angle_pre = pre[key + '/l_joint_angle'][:]   #[362, 6, 7]
r_joint_angle_pre = pre[key + '/r_joint_angle'][:]  

l_glove_angle_pre = pre[key + '/l_glove_angle'][:]   #[362, 12]
r_glove_angle_pre = pre[key + '/r_glove_angle'][:]
total_frames = l_joint_angle_pre.shape[0]

env = gym.make('yumi-v1')
observation = env.reset()
env.render()
joint_all = []
joint_vel_all = []

for t in range(total_frames):
    if t == start:
        action = l_joint_angle_pre[t-1, 1, :].tolist() + r_joint_angle_pre[t-1, 1, :].tolist() + l_glove_angle_pre[t-1, 1, :].tolist() + r_glove_angle_pre[t-1, 1, :].tolist()
    elif t == start+1:
        action = l_joint_angle_pre[t-2, 2, :].tolist() + r_joint_angle_pre[t-2, 2, :].tolist() + l_glove_angle_pre[t-2, 2, :].tolist() + r_glove_angle_pre[t-2, 2, :].tolist()
    elif t == start+2:
        action = l_joint_angle_pre[t-3, 3, :].tolist() + r_joint_angle_pre[t-3, 3, :].tolist() + l_glove_angle_pre[t-3, 3, :].tolist() + r_glove_angle_pre[t-3, 3, :].tolist()
    elif t == start+3:
        action = l_joint_angle_pre[t-4, 4, :].tolist() + r_joint_angle_pre[t-4, 4, :].tolist() + l_glove_angle_pre[t-4, 4, :].tolist() + r_glove_angle_pre[t-4, 4, :].tolist()
    else:
        action = l_joint_angle_pre[t, 0, :].tolist() + r_joint_angle_pre[t, 0, :].tolist() + l_glove_angle_pre[t, 0, :].tolist() + r_glove_angle_pre[t, 0, :].tolist()
    
    for i in range(15):    
        # action = l_joint_angle_pre[t, 4, :].tolist() + r_joint_angle_pre[t, 4, :].tolist() + l_glove_angle_pre[t+4, :].tolist() + r_glove_angle_pre[t+4, :].tolist()

        # action = l_joint_angle_pre[t, 0, :].tolist() + r_joint_angle_pre[t, 0, :].tolist() + l_glove_angle_pre[t, 0, :].tolist() + r_glove_angle_pre[t, 0, :].tolist()+\
        #          l_joint_angle_pre[t, 1, :].tolist() + r_joint_angle_pre[t, 1, :].tolist() + l_glove_angle_pre[t, 1, :].tolist() + r_glove_angle_pre[t, 1, :].tolist()+\
        #          l_joint_angle_pre[t, 2, :].tolist() + r_joint_angle_pre[t, 2, :].tolist() + l_glove_angle_pre[t, 2, :].tolist() + r_glove_angle_pre[t, 2, :].tolist()+\
        #          l_joint_angle_pre[t, 3, :].tolist() + r_joint_angle_pre[t, 3, :].tolist() + l_glove_angle_pre[t, 3, :].tolist() + r_glove_angle_pre[t, 3, :].tolist()+\
        #          l_joint_angle_pre[t, 4, :].tolist() + r_joint_angle_pre[t, 4, :].tolist() + l_glove_angle_pre[t, 4, :].tolist() + r_glove_angle_pre[t, 4, :].tolist()

        observation, reward, done, info = env.step(action)
        state_joint = np.array(observation[:38])
        state_velocity = np.array(observation[38:])
        joint_all.append(state_joint)
        joint_vel_all.append(state_velocity)



n1 = np.array(joint_all).shape
print(n1)
env.close()

# save data
hf = h5py.File(os.path.join('/home/wtt/Data/compare_result/fault-error/', 'neike_pred.h5'), 'w')
g1 = hf.create_group('group1')
g1.create_dataset('joint', data=np.array(joint_all))
g1.create_dataset('joint_vel', data=np.array(joint_vel_all))
hf.close()
print('Target H5 file saved!') 