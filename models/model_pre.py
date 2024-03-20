#This is the model for prediction in latent space
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from models.STGCN import *
from utils.config import cfg

import os, inspect, sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir)


class STGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels=64, kernel_size_p = 5, kernel_size_t=3):
        super(STGCN, self).__init__()
        self.stgcn = st_gcn(in_channels, out_channels, (kernel_size_t, kernel_size_p)) 

    def forward(self, x, A):
        #----input-----
        # x:[N, in_channels, T_in, V],   A[K,V,V]   arm: x[bat, 64, t, 6]
        # N:batch
        # K:spatial kernel size
        # T:time
        # V:number of graph nodes
        #----output-----
        # x:[N, out_channels, T_out, V], A[K,V,V]   arm: x[bat, 64, t, 6]
        
        x, A = self.stgcn(x, A)      
        return x


class ArmNet(torch.nn.Module):
    def __init__(self, period):
        super(ArmNet, self).__init__()
        self.period = period
        self.kernel_size_p = 5
        self.encoder = STGCN(in_channels=64)   
        
        self.transform1 = nn.Sequential(nn.Linear(64*self.period, 128), nn.ReLU(), nn.Linear(128, 64), nn.Tanh())
        self.transform2 = nn.Sequential(nn.Linear(64*self.period, 128), nn.ReLU(), nn.Linear(128, 64), nn.Tanh())
        self.transform3 = nn.Sequential(nn.Linear(64*self.period, 128), nn.ReLU(), nn.Linear(128, 64), nn.Tanh())
        self.transform4 = nn.Sequential(nn.Linear(64*self.period, 128), nn.ReLU(), nn.Linear(128, 64), nn.Tanh())
        self.transform5 = nn.Sequential(nn.Linear(64*self.period, 128), nn.ReLU(), nn.Linear(128, 64), nn.Tanh())

    
    def forward(self, data, arm_z, arm_z_last):
        return self.encode(data, arm_z, arm_z_last)

    def encode(self, data, arm_z, arm_z_last):    
        z = self.encoder(arm_z, data.latent_arm_adj_matix[:self.kernel_size_p, :, :])   #x:[b, 64, 5, 14]
        z = z.permute(0, 3, 2, 1).contiguous().view(-1, 5, 64)  #[b*14, 5, 64]
        
        z1 = self.transform1(z.view(data.num_graphs*14, -1))    #[b*14, 64*t]
        z2 = self.transform2(z.view(data.num_graphs*14, -1))
        z3 = self.transform3(z.view(data.num_graphs*14, -1))
        z4 = self.transform4(z.view(data.num_graphs*14, -1))
        z5 = self.transform5(z.view(data.num_graphs*14, -1))
        
        z = torch.stack((z1, z2, z3, z4, z5), dim = 1)                                  #[b*14, 5, 64]
        z = z.view(-1, 14, 5, 64).permute(0, 2, 1, 3).contiguous()                      #[b, 5, 14, 64]
        z_last = torch.stack((arm_z_last, arm_z_last, arm_z_last, arm_z_last, arm_z_last), dim = 1)   #[b, 5, 14, 64]
        z = z + z_last         #[b, 5, 14, 64]
        return z

class HandNet(torch.nn.Module):
    def __init__(self, period):
        super(HandNet, self).__init__()
        self.period = period      
        self.kernel_size_p = 5
        self.encoder = STGCN(in_channels=64)
         
        self.transform1 = nn.Sequential(nn.Linear(64*self.period, 128), nn.ReLU(), nn.Linear(128, 64), nn.Tanh())
        self.transform2 = nn.Sequential(nn.Linear(64*self.period, 128), nn.ReLU(), nn.Linear(128, 64), nn.Tanh())
        self.transform3 = nn.Sequential(nn.Linear(64*self.period, 128), nn.ReLU(), nn.Linear(128, 64), nn.Tanh())
        self.transform4 = nn.Sequential(nn.Linear(64*self.period, 128), nn.ReLU(), nn.Linear(128, 64), nn.Tanh())
        self.transform5 = nn.Sequential(nn.Linear(64*self.period, 128), nn.ReLU(), nn.Linear(128, 64), nn.Tanh())    

    def forward(self, data, hand_z, hand_z_last):
        return self.encode(data, hand_z, hand_z_last)

    def encode(self, data, hand_z, hand_z_last):
        z = self.encoder(hand_z, data.latent_l_hand_adj_matix[:self.kernel_size_p, :, :])  #[b*2, 64, 5, 18]
        z = z.permute(0, 3, 2, 1).contiguous().view(-1, 5, 64)  #[b*18*2, 5, 64]

        z1 = self.transform1(z.view(data.num_graphs*18*2, -1))   #[b*18*2, 64]
        z2 = self.transform2(z.view(data.num_graphs*18*2, -1))
        z3 = self.transform3(z.view(data.num_graphs*18*2, -1))
        z4 = self.transform4(z.view(data.num_graphs*18*2, -1))
        z5 = self.transform5(z.view(data.num_graphs*18*2, -1))

        z = torch.stack((z1,z2,z3,z4,z5), dim = 1)    #[b*18*2, 5, 64]
        z = z.view(-1, 18, 5, 64).permute(0, 2, 1, 3).contiguous()   #[b*2, 5, 18, 64]
        z_last = torch.stack((hand_z_last, hand_z_last, hand_z_last, hand_z_last, hand_z_last), dim = 1)   #[b*2, 5, 18, 64]
        z = z + z_last
        return z  #[b*2, 5, 18, 64]

class PredictNet(torch.nn.Module):
    def __init__(self, period):
        super(PredictNet, self).__init__()
        self.arm_net = ArmNet(period)
        self.hand_net = HandNet(period)

    def forward(self, data, arm_z, arm_z_last, hand_z, hand_z_last):
        return self.encode(data, arm_z, arm_z_last, hand_z, hand_z_last)

    def encode(self, data, arm_z, arm_z_last, hand_z, hand_z_last):
        arm_z_p = self.arm_net.encode(data, arm_z, arm_z_last)             # [b, 5, 14, 64]
        hand_z_p = self.hand_net.encode(data, hand_z, hand_z_last)         # [b*2, 5, 18, 64]
        z = torch.cat((arm_z_p.view(-1, 64), hand_z_p.view(-1, 64)), dim=0)        
        return z 
