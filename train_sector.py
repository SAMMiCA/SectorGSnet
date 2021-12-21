import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # assigned GPU #2, other 3 gpus are not available.
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn

import yaml
import time
import argparse
import os
import shutil
# import wandb

from utils.metrics import AverageMeter
from data_loader import kitti_loader
from network.points_partition import points_to_sector_fixed_ops, points_to_sector_dynamic_ops
from network.model import This_Net

from network.loss_function import Lovasz_softmax
from network.dice_score import dice_loss

#######
## 0 is ground point, 1 is non-ground , -1 is background
######
# ---------------------------------------------------------------------------- #
# Load config ; declare Meter class, checkpoint, etc.
# ---------------------------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument('-conf', '--configs', default='./configs/sector_conf.yaml',
                    help="Choose configs: ./configs/pillar_conf.yaml, ./configs/sector_conf.yaml")
args = parser.parse_args()

try:
    with open(args.configs) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    print('\n'.join('%s:%s' % item for item in config_dict.items()))
    class ConfigToClass:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    cfg = ConfigToClass(**config_dict)
except:
    print("No config file found at workspace.")

if not os.path.exists(cfg.checkpoints_path):
    os.mkdir(cfg.checkpoints_path)
if not os.path.exists(cfg.evaluation_path):
    os.mkdir(cfg.evaluation_path)

def save_checkpoint(epoch_num, state, is_best, path, network):
    timenow = time.strftime('_%m%d%H%M', time.localtime(time.time()))
    filename = path + network + '_epoch_' + str(epoch_num) + '_'+timenow + '_sector.pth.tar'
    torch.save(state, filename)
    if is_best:
        best_filename = filename[:-8] + '_best_sector.pth.tar'
        shutil.copyfile(filename, best_filename)

# ---------------------------------------------------------------------------- #
# Setup dataloader, logging, model, loss, optimizer, scheduler, etc
# ---------------------------------------------------------------------------- #
# 1. create data loaders
train_dataset = kitti_loader(data_dir=cfg.data_path,pc_folder=cfg.pc_folder,lb_folder=cfg.lb_folder,
                             train=True, skip_frames=1)
train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size * cfg.num_gpus, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

val_dataset = kitti_loader(data_dir=cfg.data_path, pc_folder=cfg.pc_folder,lb_folder=cfg.lb_folder,
                           train=False, skip_frames=1)

val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size * cfg.num_gpus, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

# 2. Initialize logging
# experiment = wandb.init(project='SectorNet')

model = This_Net(cfg)
print("Entire Model has {} paramerters in total".format(sum(x.numel() for x in model.parameters())))

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.cuda()

# loss_ft = Focal_tversky().cuda()
# loss_ls = Lovasz_softmax().cuda()
# loss_comb= Lovasz_softmax().cuda()
loss_ce = nn.CrossEntropyLoss(ignore_index=-1).cuda()
# loss_crs = Focal_tversky().cuda()

# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1.0e-8)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1.0e-8)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.1, patience=2, verbose=True, eps=1e-08)

# ---------------------------------------------------------------------------- #
# Train
# ---------------------------------------------------------------------------- #
def train(epoch):
    print("Training Sector Model...")
    model.train()
    losses = AverageMeter()
    for batch_idx, (data, labels) in enumerate(train_dataloader):
        batch_size = data.shape[0]
        point_feature_in_sector = []
        coors_sectors = []
        num_sectors = []
        data = data.numpy()
        for i in range(batch_size):
            p, c, n = points_to_sector_dynamic_ops(data[i],
                                                   sector_shape=[64, 256, 1],
                                                   max_points_in_sector = 100,
                                                   max_sector = 10000)
            # p.shape, c.shape, n.shape - p,n,6; p,3, p,1
            point_feature_in_sector.append(torch.from_numpy(p))
            c = torch.from_numpy(c)
            c = F.pad(c, (1, 0), 'constant', i)  # (p,batch_index + 3 )
            coors_sectors.append(c)
            num_sectors.append(torch.from_numpy(n))

        point_feature_in_sector = torch.cat(point_feature_in_sector).float().cuda()  # bs,p,n,6
        coors_sectors = torch.cat(coors_sectors).float().cuda()  # bs, p, batch_index +3
        num_sectors = torch.cat(num_sectors).float().cuda()  # bs, p,1
        labels = labels.long().cuda()
        optimizer.zero_grad()
        output = model(point_feature_in_sector, coors_sectors, num_sectors)
        loss = loss_ce(output, labels)
        # loss= loss_comb(output, labels)
        # loss = loss_ce(output, labels)\
        #        + dice_loss(F.softmax(output,dim=1).float(),
        #                             F.one_hot(labels,3).permute(0,3,1,2).float(),
        #                             multiclass=True).cuda()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), batch_size)

        # experiment.log({
        #     'train loss': loss.item(),
        #     'epoch': epoch
        # })
        if batch_idx % cfg.print_freq == 0:
            print('Train : [Epoch-{0}][{1}/{2}]\t'
                  'Loss: {loss.val:.4f} Avg Loss: {loss.avg:.4f})'.format(
                epoch, batch_idx, len(train_dataloader), loss=losses))

# ---------------------------------------------------------------------------- #
# Validation
# ---------------------------------------------------------------------------- #
def validate():
    print("Validating Sector Model...")
    model.eval()
    losses= AverageMeter()
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(val_dataloader):
            batch_size = data.shape[0]
            point_feature_in_sector = []
            coors_sectors = []
            num_sectors = []
            data = data.numpy()
            for i in range(batch_size):
                p, c, n = points_to_sector_dynamic_ops(data[i],
                                                       sector_shape=[64, 256, 1],
                                                       max_points_in_sector = 100,
                                                       max_sector = 10000)

                # print(time.time()-time_start)
                # v.shape, c.shape, n.shape - p,n,6 ; p,3, p,1
                point_feature_in_sector.append(torch.from_numpy(p))
                c = torch.from_numpy(c)
                c = F.pad(c, (1, 0), 'constant', i)  # (p x (batch + x,y,z))
                coors_sectors.append(c)
                num_sectors.append(torch.from_numpy(n))

            point_feature_in_sector = torch.cat(point_feature_in_sector).float().cuda()  # p,n,6
            coors_sectors = torch.cat(coors_sectors).float().cuda()  # p,3
            num_sectors = torch.cat(num_sectors).float().cuda()  # p,1
            labels = labels.long().cuda()

            output = model(point_feature_in_sector, coors_sectors, num_sectors)
            # loss = loss_comb(output, labels)
            loss = loss_ce(output, labels)
            # loss = loss_ce(output, labels) + dice_loss(F.softmax(output, dim=1).float(),
            #                                            F.one_hot(labels,3).permute(0, 3, 1, 2).float(),
            #                                            multiclass=True).cuda()
            losses.update(loss.item(), batch_size)
            if batch_idx % cfg.print_freq == 0:
                print('Validate : [{0}/{1}]\t'
                      'Loss : {loss.val:.4f}  Average Loss : {loss.avg:.4f}'.format(
                    batch_idx, len(val_dataloader), loss=losses))
    return losses.avg

lowest_loss = 1
def main():
    global lowest_loss
    for epoch in range(cfg.epochs):
        train(epoch)
        loss_val = validate()
        scheduler.step(metrics=0)  # adjust_learning_rate
        if (cfg.save_checkpoints):
            # remember best prec@1 and save checkpoint
            is_best = loss_val < lowest_loss
            lowest_loss = min(loss_val, lowest_loss)
            save_checkpoint(epoch, {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'lowest_loss': lowest_loss,
                'optimizer': optimizer.state_dict(),
            }, is_best, cfg.checkpoints_path, cfg.exp_name)

if __name__ == '__main__':
    main()