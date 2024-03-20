#This main function mainly trains prediction model, and the retargeting model is trained before
#So, the retargeting model is off-the-shelf and helps to transfer data "from human to latent" and "latent to robot"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.transforms as transforms
from torch_geometric.data import DataListLoader
from torch.optim.lr_scheduler import StepLR, ExponentialLR, LambdaLR
from tensorboardX import SummaryWriter

import models.model_pre as model
import models.model_retarget as model_retarget
from models.loss_pre import SmoothLoss
from train import train_epoch
from test import test_epoch
import dataset

import os
from utils.config import cfg
from utils.util import create_folder
import logging
import argparse
from datetime import datetime


# Argument parse
parser = argparse.ArgumentParser(description='Command line arguments')
parser.add_argument('--cfg', default='configs/train/yumi.yaml', type=str, help='Path to configuration file')
args = parser.parse_args()

# Configurations parse
cfg.merge_from_file(args.cfg)
cfg.freeze()
print(cfg)

# Create folder
create_folder(cfg.OTHERS.SAVE)
create_folder(cfg.OTHERS.LOG)
create_folder(cfg.OTHERS.SUMMARY)

# Create logger & tensorboard writer
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.FileHandler(os.path.join(cfg.OTHERS.LOG, "{:%Y-%m-%d_%H-%M-%S}.log".format(datetime.now()))), logging.StreamHandler()])
logger = logging.getLogger()
writer = SummaryWriter(os.path.join(cfg.OTHERS.SUMMARY, "{:%Y-%m-%d_%H-%M-%S}".format(datetime.now())))

# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    #Load data
    pre_transform = transforms.Compose([dataset.Normalize()])

    train_set = getattr(dataset, cfg.DATASET.TRAIN.SOURCE_NAME)(sliding_window = cfg.HYPER.PERIOD, root = cfg.DATASET.TRAIN.SOURCE_PATH, pre_transform = pre_transform) 
    train_loader = DataListLoader(train_set, batch_size=cfg.HYPER.BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    train_target = sorted([target for target in getattr(dataset, cfg.DATASET.TRAIN.TARGET_NAME)(root=cfg.DATASET.TRAIN.TARGET_PATH)], key=lambda target : target.skeleton_type)

    test_set = getattr(dataset, cfg.DATASET.TEST.SOURCE_NAME)(sliding_window = cfg.HYPER.PERIOD, root = cfg.DATASET.TEST.SOURCE_PATH, pre_transform = pre_transform)
    test_loader = DataListLoader(test_set, batch_size=cfg.HYPER.BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    test_target = sorted([target for target in getattr(dataset, cfg.DATASET.TEST.TARGET_NAME)(root=cfg.DATASET.TEST.TARGET_PATH)], key=lambda target : target.skeleton_type)

    # Create model for prediction
    model_p = getattr(model, cfg.MODEL.NAME)(cfg.HYPER.PERIOD).to(device)

    #Load checkpoint for sequence retargeting
    model_r = getattr(model_retarget, cfg.RETARGET.NAME)(cfg.HYPER.PERIOD).to(device)    
    if cfg.MODEL.CHECKPOINT is not None:
        model_r.load_state_dict(torch.load(cfg.RETARGET.CHECKPOINT))

    # Create loss criterion
    arm_criterion = nn.MSELoss() if cfg.LOSS.ARM_JOI else None
    fin_criterion = nn.MSELoss() if cfg.LOSS.FIN_JOI else None
    acc_criterion = SmoothLoss() if cfg.LOSS.ACC else None

    # Create optimizer
    optimizer = optim.Adam(model_p.parameters(), lr=cfg.HYPER.LEARNING_RATE)
    # scheduler = StepLR(optimizer, step_size=2, gamma = 0.96)
    # scheduler = ExponentialLR(optimizer, gamma=0.95, last_epoch = -1)
    # scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(epoch+1))

    best_loss = float('Inf') 
    train_loss_all = []
    test_loss_all = []

    for epoch in range(cfg.HYPER.EPOCHS):

        # Start training
        train_loss = train_epoch(model_p, model_r, arm_criterion, fin_criterion, acc_criterion, cfg.HYPER.PERIOD, cfg.HYPER.BATCH_SIZE, optimizer, train_loader, train_target, epoch, logger, cfg.OTHERS.LOG_INTERVAL, writer, device) 
        train_loss_all.append(train_loss)

        # optimizer.step()
        # print("第%d个epoch的学习率%f" % (epoch+1, optimizer.param_groups[0]['lr']))
        # scheduler.step()
        
        #Start testing
        test_loss = test_epoch(model_p, model_r, arm_criterion, fin_criterion, acc_criterion, cfg.HYPER.PERIOD, cfg.HYPER.BATCH_SIZE, test_loader, test_target, epoch, logger, cfg.OTHERS.LOG_INTERVAL, writer, device)
        test_loss_all.append(test_loss)

        # Save model
        #save best model
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model_p.state_dict(), os.path.join(cfg.OTHERS.SAVE, "best_model_epoch_{:04d}.pth".format(epoch)))
            logger.info("Epoch {} Model Saved".format(epoch+1).center(60, '-'))
        #save last model
        if epoch == cfg.HYPER.EPOCHS-1:
            torch.save(model_p.state_dict(), os.path.join(cfg.OTHERS.SAVE, "last_model_epoch_{:04d}.pth".format(epoch)))
            logger.info("Epoch {} Model Saved".format(epoch+1).center(60, '-'))            
    
    print("all of the train loss for each epoch", train_loss_all)
    print("all of the test loss for each epoch", test_loss_all)