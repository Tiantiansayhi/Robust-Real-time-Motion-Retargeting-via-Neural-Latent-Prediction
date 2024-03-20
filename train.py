import torch
from torch_geometric.data import Batch
from models.loss_pre import calculate_loss
import time
import numpy as np

def train_epoch(model, model_r, arm_criterion, fin_criterion, acc_criterion, sliding_window, batchsize, optimizer, dataloader, target_skeleton, epoch, logger, log_interval, writer, device):
    logger.info("Training Epoch {}".format(epoch+1).center(60, '-'))
    start_time = time.time()

    model.train()
    all_losses = []
    arm_losses = []
    fin_losses = []
    acc_losses = []

    model_r.eval()

    for batch_idx,  data_list in enumerate(dataloader):
        # zero gradient
        optimizer.zero_grad()

        #got initial z
        z = model_r.encode(Batch.from_data_list(data_list).to(device)).detach()    
        # print('z', z)
        arm_z = z[:batchsize*sliding_window*14].view(batchsize, sliding_window, -1, 64).permute(0, 3, 1, 2).contiguous()
        hand_z = z[batchsize*sliding_window*14:].view(batchsize*2, sliding_window, -1, 64).permute(0, 3, 1, 2).contiguous()
        # print('hand_z', hand_z.shape)
        arm_z_last = z[:batchsize*sliding_window*14].view(batchsize, sliding_window, -1, 64)[:, -1, :, :]
        # print('arm_z_last', arm_z_last)
        hand_z_last = z[batchsize*sliding_window*14:].view(batchsize*2, sliding_window, -1, 64)[:, -1, :, :]
        
        # forward
        z_pre = model(Batch.from_data_list(data_list).to(device), arm_z, arm_z_last, hand_z, hand_z_last).detach() 
        loss = calculate_loss(data_list, z_pre, model_r, arm_criterion, fin_criterion, acc_criterion, sliding_window, batchsize, target_skeleton, all_losses, arm_losses, fin_losses, acc_losses)
        
        # backward
        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

        # optimize
        optimizer.step()

        # log
        if (batch_idx + 1) % log_interval == 0:
            logger.info("epoch {:04d} | iteration {:05d} | ARM_JOI Loss {}| FIN_JOI Loss {} | ACC_JOI Loss {}".format(epoch+1, batch_idx+1, arm_losses[-1], fin_losses[-1], acc_losses[-1]))    

    # Compute average loss
    train_loss = sum(all_losses)/len(all_losses)
    # print(np.array(arm_losses).shape)
    arm_loss = np.mean(np.array(arm_losses), axis = 0)
    fin_loss = np.mean(np.array(fin_losses), axis = 0)
    acc_loss = sum(acc_losses)/len(acc_losses)

    # Log
    writer.add_scalars('training_loss', {'train': train_loss}, epoch+1)
    # writer.add_scalars('arm_loss', {'train': arm_loss}, epoch+1) 
    # writer.add_scalars('fin_loss', {'train': fin_loss}, epoch+1)
    end_time = time.time()
    logger.info("Epoch {:04d} | Training Time {:.2f} s | Avg Training Loss {:.6f} | Avg ARM_JOI {} | Avg FIN_JOI {} | Avg ACC {}".format(epoch+1, end_time-start_time, train_loss, arm_loss, fin_loss, acc_loss))    
     
    return train_loss
