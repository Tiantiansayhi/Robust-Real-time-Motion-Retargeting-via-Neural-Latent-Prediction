#Motion Sequence Retargeting model:Spatial-temporal Graph-based Encoder & Decoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing

import os, inspect, sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir)
from kinematics import ForwardKinematicsURDF, ForwardKinematicsAxis


class SpatialBasicBlock(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_channels, aggr='add', batch_norm=False, bias=True, **kwargs):
        super(SpatialBasicBlock, self).__init__(aggr=aggr, **kwargs)
        self.batch_norm = batch_norm
        # network architecture
        self.lin = nn.Linear(2*in_channels + edge_channels, out_channels, bias=bias)
        self.upsample = nn.Linear(in_channels, out_channels, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bn.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None):
        if isinstance(x, torch.Tensor):
            x = (x, x)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.bn(out) if self.batch_norm else out
        out += self.upsample(x[1])
        return out

    def message(self, x_i, x_j, edge_attr):
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return F.leaky_relu(self.lin(z))

class TemporalBasicBlock(MessagePassing):
    def __init__(self, channels, aggr='mean', batch_norm=False, bias=True, **kwargs):
        super(TemporalBasicBlock, self).__init__(aggr=aggr, **kwargs)
        self.batch_norm = batch_norm
        # network architecture
        self.lin = nn.Linear(2*channels, channels, bias=bias)
        self.bn = nn.BatchNorm1d(channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bn.reset_parameters()

    def forward(self, x, edge_index):
        if isinstance(x, torch.Tensor):
            x = (x, x)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x)
        out = self.bn(out) if self.batch_norm else out
        out += x[0]
        return out

    def message(self, x_i, x_j):
        z = torch.cat([x_i, x_j], dim=-1)
        return F.relu(self.lin(z))

class Encoder(torch.nn.Module):
    def __init__(self, channels, dim):
        super(Encoder, self).__init__()
        self.conv1 = SpatialBasicBlock(in_channels=channels, out_channels=16, edge_channels=dim)
        self.conv2 = SpatialBasicBlock(in_channels=16, out_channels=32, edge_channels=dim)
        self.conv3 = SpatialBasicBlock(in_channels=32, out_channels=64, edge_channels=dim)
        self.conv4 = TemporalBasicBlock(channels=64)
        # self.dropout = nn.Dropout(p=0.1)
    
    def forward(self, x, edge_index, edge_attr, temporal_edge_index):
        """
        Keyword arguments:
        x -- joint angles [num_nodes, num_node_features]
        edge_index -- edge index [2, num_edges]
        edge_attr -- edge features [num_edges, num_edge_features]
        temporal_edge_index -- edge index in temporal dimension [2, num_edges]
        """
        out = self.conv1(x, edge_index, edge_attr)     
        out = self.conv2(out, edge_index, edge_attr)
        out = self.conv3(out, edge_index, edge_attr)
        out = self.conv4(out, temporal_edge_index)
        # out = self.dropout(out)
        return out


class Decoder(torch.nn.Module):
    def __init__(self, channels, dim):
        super(Decoder, self).__init__()
        self.conv1 = SpatialBasicBlock(in_channels=64+2, out_channels=32, edge_channels=dim)
        self.conv2 = SpatialBasicBlock(in_channels=32, out_channels=16, edge_channels=dim)
        self.conv3 = SpatialBasicBlock(in_channels=16, out_channels=channels, edge_channels=dim)
        self.conv4 = TemporalBasicBlock(channels=channels)        

    def forward(self, x, edge_index, edge_attr, lower, upper, temporal_edge_index):
        """
        Keyword arguments:
        x -- joint angles [num_nodes, num_node_features]
        edge_index -- edge index [2, num_edges]
        edge_attr -- edge features [num_edges, num_edge_features]
        temporal_edge_index -- edge index in temporal dimension [2, num_edges]
        """
        x = torch.cat([x, lower, upper], dim=1)
        out = self.conv1(x, edge_index, edge_attr)
        out = self.conv2(out, edge_index, edge_attr)
        out = self.conv3(out, edge_index, edge_attr)
        out = self.conv4(out, temporal_edge_index).tanh()
        return out


class ArmNet(torch.nn.Module):
    def __init__(self, period):
        super(ArmNet, self).__init__()

        """
        arm_x:[batch*6, 6]
        edge_index:[2, batch*4]
        edge_attr:[batch*4, 3]
        """
        self.encoder = Encoder(6, 3)
        self.transform = nn.Sequential(nn.Linear(6*64, 14*64), nn.Tanh())
        self.decoder = Decoder(1, 6)
        self.fk = ForwardKinematicsURDF()
        self.period = period
    
    def forward(self, data, target):
        return self.decode(self.encode(data), target)
    
    def encode(self, data):
        # print('encode')
        # print('edge_index', data.edge_index)
        # print('edge_attr', data.edge_attr, data.edge_attr.shape)
        # print('arm_temporal_edge_index', data.temporal_edge_index, data.temporal_edge_index.shape)
        z = self.encoder(data.x, data.edge_index, data.edge_attr, data.temporal_edge_index)     #[batch*6*period, 64]  
        z = self.transform(z.view(data.num_graphs, -1, 64).view(data.num_graphs, self.period, -1, 64).view(data.num_graphs*self.period, -1)).view(data.num_graphs*self.period, -1, 64).view(-1, 64)    #[batch*14*t, 64]
        return z
    
    def decode(self, z, target):  
        # print('decode')   
        # print('edge_index', target.edge_index)  
        # print('edge_attr', target.edge_attr, target.edge_attr.shape)
        # print('temporal_edge_index', target.temporal_edge_index)        
        ang = self.decoder(z, target.edge_index, target.edge_attr, target.lower, target.upper, target.temporal_edge_index)
        ang = target.lower + (target.upper - target.lower)*(ang + 1)/2     #[batch*14*t, 1] 
        pos, rot, global_pos = self.fk(ang, target.parent, target.offset, sum([data.t for data in target.to_data_list()]))
        return z, ang, pos, rot, global_pos, None, None, None, None


class HandNet(torch.nn.Module):
    def __init__(self, period):
        super(HandNet, self).__init__()
        self.encoder = Encoder(3, 3)
        self.transform = nn.Sequential(
            nn.Linear(17*64, 18*64),
            nn.Tanh(),
        )
        self.decoder = Decoder(1, 6)
        self.fk = ForwardKinematicsAxis()
        self.period = period

    def forward(self, data, target):
        return self.decode(self.encode(data), target)

    def encode(self, data):
        x = torch.cat([data.l_hand_x, data.r_hand_x], dim=0)   #x:[bat*34, 3]
        edge_index = torch.cat([data.l_hand_edge_index, data.r_hand_edge_index+data.l_hand_x.size(0)], dim=1)     #[2, 160]    
        # print('hand_edge_index', edge_index, '\n', edge_index.shape)
        edge_attr = torch.cat([data.l_hand_edge_attr, data.r_hand_edge_attr], dim=0)                              #[160, 3]
        temporal_edge_index = torch.cat([data.l_hand_temporal_edge_index, data.r_hand_temporal_edge_index+data.l_hand_x.size(0)], dim=1)        
        # print('hand_temporal_edge_index', temporal_edge_index, '\n', temporal_edge_index.shape)
        z = self.encoder(x, edge_index, edge_attr, temporal_edge_index)
        z = self.transform(z.view(2*data.num_graphs, -1, 64).view(data.num_graphs*2, self.period, -1, 64).view(2*data.num_graphs*self.period, -1)).view(2*data.num_graphs*self.period, -1, 64).view(-1, 64)   #[batch*14*t*2, 64]        
        return z

    def decode(self, z, target):
        edge_index = torch.cat([target.hand_edge_index, target.hand_edge_index+z.size(0)//2], dim=1)      
        edge_attr = torch.cat([target.hand_edge_attr, target.hand_edge_attr], dim=0)    
        temporal_edge_index = torch.cat([target.hand_temporal_edge_index, target.hand_temporal_edge_index+z.size(0)//2], dim=1)        
        lower = torch.cat([target.hand_lower, target.hand_lower], dim=0)
        upper = torch.cat([target.hand_upper, target.hand_upper], dim=0)
        offset = torch.cat([target.hand_offset, target.hand_offset], dim=0)
        parent = torch.cat([target.hand_parent, target.hand_parent], dim=0)
        num_graphs = 2*sum([data.t for data in target.to_data_list()])
        axis = torch.cat([target.hand_axis, target.hand_axis], dim=0)
        hand_ang = self.decoder(z, edge_index, edge_attr, lower, upper, temporal_edge_index)
        hand_ang = lower + (upper - lower)*(hand_ang + 1)/2       
        hand_pos, _, _ = self.fk(hand_ang, parent, offset, num_graphs, axis)
        half = hand_ang.size(0)//2
        l_hand_ang, r_hand_ang = hand_ang[:half, :], hand_ang[half:, :]
        l_hand_pos, r_hand_pos = hand_pos[:half, :], hand_pos[half:, :]
        return z, None, None, None, None, l_hand_ang, l_hand_pos, r_hand_ang, r_hand_pos


class YumiNet(torch.nn.Module):
    def __init__(self, period):
        super(YumiNet, self).__init__()
        self.arm_net = ArmNet(period)
        self.hand_net = HandNet(period)

    def forward(self, data, target):
        return self.decode(self.encode(data), target)

    def encode(self, data):
        arm_z = self.arm_net.encode(data)
        # print(arm_z.shape)
        hand_z = self.hand_net.encode(data)
        # print(hand_z.shape)
        z = torch.cat([arm_z, hand_z], dim=0)
        # print(z.shape)
        return z

    def decode(self, z, target):
        half = target.num_nodes
        arm_z, hand_z = z[:half, :], z[half:, :]
        _, ang, pos, rot, global_pos, _, _, _, _ = self.arm_net.decode(arm_z, target)
        _, _, _, _, _, l_hand_ang, l_hand_pos, r_hand_ang, r_hand_pos = self.hand_net.decode(hand_z, target)
        return z, ang, pos, rot, global_pos, l_hand_ang, l_hand_pos, r_hand_ang, r_hand_pos

