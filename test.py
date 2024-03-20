import torch
from torch_geometric.data import Batch
from models.loss_pre import calculate_loss
import time
import numpy as np

def test_epoch(model, model_r, arm_criterion, fin_criterion, acc_criterion, sliding_window, batchsize, dataloader, target_skeleton, epoch, logger, log_interval, writer, device):
    logger.info("Testing Epoch {}".format(epoch+1).center(60, '-'))
    start_time = time.time()

    model.eval()
    all_losses = []
    arm_losses = []
    fin_losses = []
    acc_losses = []

    model_r.eval()
    with torch.no_grad():
        for batch_idx,  data_list in enumerate(dataloader):

            #got initial z
            z = model_r.encode(Batch.from_data_list(data_list).to(device)).detach()    
            arm_z = z[:batchsize*sliding_window*14].view(batchsize, sliding_window, -1, 64).permute(0, 3, 1, 2).contiguous()
            hand_z = z[batchsize*sliding_window*14:].view(batchsize*2, sliding_window, -1, 64).permute(0, 3, 1, 2).contiguous()

            arm_z_last = z[:batchsize*sliding_window*14].view(batchsize, sliding_window, -1, 64)[:, -1, :, :]
            hand_z_last = z[batchsize*sliding_window*14:].view(batchsize*2, sliding_window, -1, 64)[:, -1, :, :]

            # forward
            z_pre = model(Batch.from_data_list(data_list).to(device), arm_z, arm_z_last, hand_z, hand_z_last).detach() 
            loss = calculate_loss(data_list, z_pre, model_r, arm_criterion, fin_criterion, acc_criterion, sliding_window, batchsize, target_skeleton, all_losses, arm_losses, fin_losses, acc_losses)
 
    # Compute average loss
    test_loss = sum(all_losses)/len(all_losses)
    arm_loss = np.mean(np.array(arm_losses), axis = 0)
    fin_loss = np.mean(np.array(fin_losses), axis = 0)
    acc_loss = sum(acc_losses)/len(acc_losses)   
    
    # Log
    writer.add_scalars('testing_loss', {'test': test_loss}, epoch+1)
    # writer.add_scalars('arm_loss', {'test': arm_loss}, epoch+1)    
    # writer.add_scalars('fin_loss', {'test': fin_loss}, epoch+1)
    end_time = time.time()
    logger.info("Epoch {:04d} | Testing Time {:.2f} s | Avg Testing Loss {:.6f} | Avg ARM {}| Avg FIN {} | Avg ACC {}".format(epoch+1, end_time-start_time, test_loss, arm_loss, fin_loss, acc_loss))    
    
    return test_loss

