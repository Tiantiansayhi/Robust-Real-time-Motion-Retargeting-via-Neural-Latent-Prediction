import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data, DataListLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def decode(z_pre, model_r, seq_length, batchsize, target_skeleton):
    #window
    arm_ang_all = []
    arm_pos_all = []
    arm_rot_all = []
    l_hand_ang_all = []
    r_hand_ang_all = []
    l_hand_pos_all = []
    r_hand_pos_all = []
    
    model_r.eval() 
    for target_idx, target in enumerate(target_skeleton):
        target_list = []
        for i in range(batchsize):
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
        
        
        _, target_ang, _, target_rot, target_global_pos, target_l_hand_ang, target_l_hand_pos, target_r_hand_ang, target_r_hand_pos = model_r.decode(z_pre, Batch.from_data_list(target_list).to(z_pre.device))


        if target_ang is not None and target_global_pos is not None:
            arm_ang_all.append(target_ang)
            arm_pos_all.append(target_global_pos)
        if target_rot is not None:
            arm_rot_all.append(target_rot)
        if target_l_hand_ang is not None and target_r_hand_ang is not None:
            l_hand_ang_all.append(target_l_hand_ang)
            r_hand_ang_all.append(target_r_hand_ang)
        # if target_l_hand_pos is not None and target_r_hand_pos is not None:
        #     l_hand_pos_all.append(target_l_hand_pos)
        #     r_hand_pos_all.append(target_r_hand_pos) 
        
    
    if arm_ang_all and arm_pos_all:
        arm_ang = torch.cat(arm_ang_all, dim=0).view(batchsize, seq_length, -1)          #[b, t, 14]
        arm_pos = torch.cat(arm_pos_all, dim=0).view(batchsize, seq_length, -1, 3)       #[b, t, 14, 3]

    if arm_rot_all:
        arm_rot = torch.cat(arm_rot_all, dim=0).view(batchsize, seq_length, -1, 3, 3)    #[b, t, 14, 3, 3]

    if l_hand_ang_all and r_hand_ang_all:
        l_hand_ang = torch.cat(l_hand_ang_all, dim=0).view(batchsize, seq_length, -1)    
        r_hand_ang = torch.cat(r_hand_ang_all, dim=0).view(batchsize, seq_length, -1)   
        
        #remove zeros  
        l_hand_angle = torch.cat([l_hand_ang[:, :, 1:3],l_hand_ang[:, :, 4:6],l_hand_ang[:, :, 7:9],l_hand_ang[:, :, 10:12],l_hand_ang[:, :, 13:17]], dim=-1)
        r_hand_angle = torch.cat([r_hand_ang[:, :, 1:3],r_hand_ang[:, :, 4:6],r_hand_ang[:, :, 7:9],r_hand_ang[:, :, 10:12],r_hand_ang[:, :, 13:17]], dim=-1)
   
    # if l_hand_pos_all and r_hand_pos_all:
    #     l_hand_pos = torch.cat(l_hand_pos_all, dim=0).view(config.cfg.HYPER.BATCH_SIZE, -1, 3)       
    #     r_hand_pos = torch.cat(r_hand_pos_all, dim=0).view(config.cfg.HYPER.BATCH_SIZE, -1, 3)

    
    # hand_ang = torch.cat((l_hand_angle, r_hand_angle), dim = 0)       #[b*2, 12]
    # hand_pos = torch.cat((l_hand_pos, r_hand_pos), dim = 1)       #[b, 18*2, 3]    
    # return arm_ang, arm_pos, arm_rot, l_hand_angle, r_hand_angle, l_hand_pos, r_hand_pos, target_list
    return arm_ang, arm_pos, arm_rot, l_hand_angle, r_hand_angle, target_l_hand_pos, target_r_hand_pos, target_list
    



def calculate_loss(data_list, z_pre, model_r, arm_criterion, fin_criterion, acc_criterion, sliding_window, batchsize, target_skeleton, all_losses=[], arm_losses=[], fin_losses=[], acc_losses=[]):

    label_arm_ang = torch.cat([data.label_arm_joint for data in data_list]).to(z_pre.device).view(batchsize, sliding_window, -1)          #[b*5, 14]-->[b, 5, 14]
    label_l_hand_ang = torch.cat([data.label_l_hand_joint for data in data_list]).to(z_pre.device).view(batchsize, sliding_window, -1)
    label_r_hand_ang = torch.cat([data.label_r_hand_joint for data in data_list]).to(z_pre.device).view(batchsize, sliding_window, -1)
  

    arm_ang, arm_pos, arm_rot, l_hand_ang, r_hand_ang, l_hand_pos, r_hand_pos, target_list = decode(z_pre, model_r, sliding_window, batchsize, target_skeleton)


    arm_loss_i = []
    hand_loss_i = []

    arm_all_loss = 0
    fin_all_loss = 0  

    for i in range(sliding_window):
        if arm_criterion:
            arm_loss = arm_criterion(arm_ang[:, i, :], label_arm_ang[:, i, :])*100
            arm_loss_i.append(arm_loss.item()) 
        else:
            arm_loss = 0
            arm_loss_i.append(0) 

        if fin_criterion:    
            fin_loss = (fin_criterion(l_hand_ang[:, i, :], label_l_hand_ang[:, i, :]) + fin_criterion(r_hand_ang[:, i, :], label_r_hand_ang[:, i, :]))*100
            hand_loss_i.append(fin_loss.item())
        else:
            fin_loss = 0
            hand_loss_i.append(0)
        arm_all_loss += arm_loss 
        fin_all_loss += fin_loss
    arm_losses.append(arm_loss_i)
    fin_losses.append(hand_loss_i)

    
    #accerlation loss
    if acc_criterion:
        acc_loss = acc_criterion(arm_ang)*5
        acc_losses.append(acc_loss.item())
    else:
        acc_loss = 0
        acc_losses.append(0)
    
    #all the loss
    loss = arm_all_loss + fin_all_loss + acc_loss
    all_losses.append(loss.item())

    return loss

class SmoothLoss(nn.Module):
    def __init__(self):
        super(SmoothLoss, self).__init__()

    def forward(self, data):   
        diff1 = data[:, 1:, :] - data[:, :-1, :] 
        diff2 = diff1[:, 1:, :] - diff1[:, :-1, :]
        loss = torch.mean(torch.abs(diff2))
        return loss




if __name__ == '__main__':
    arm_z = torch.ones((16*14, 64))
    fin_z = torch.ones((16*2*18, 64))
    arm_ang, arm_pos, hand_ang, hand_pos = decode(arm_z, fin_z)
    print(arm_ang.shape)    #[16,14]
    print(arm_pos.shape)    #[16,14,3]
    print(hand_ang.shape)   #[16,12*2]
    print(hand_pos.shape)   #[16,18*2, 3]