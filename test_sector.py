import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 留出前几个GPU跑其他程序, 需要在导入模型前定义
import torch
import torch.nn.functional as F
import torch.nn as nn

import yaml
import time
import argparse
import numpy as np
import math
import numba

from network.points_partition import points_to_sector_fixed_ops, points_to_sector_dynamic_ops
from network.model import This_Net

model_name= 'unet_multi_epoch_9__10061749_sector_best_sector.pth.tar'

def metrics(gt=None, pre=None):
    "calculate Precision, Recall, Accuracy, mIoU, F1-score"
    pre_inv,gt_inv=np.logical_not(pre), np.logical_not(gt)
    tp=float(np.logical_and(pre,gt).sum())
    tn=np.logical_and(pre_inv,gt_inv).sum()
    fp=np.logical_and(pre,gt_inv).sum()
    fn=np.logical_and(pre_inv,gt).sum()

    prec = tp / (tp + fp + 1e-6)
    rec = tp / (tp + fn + 1e-6)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    F1 = 2 * tp / (2 * tp + fp + fn + 1e-6)
    IoU = tp / (tp + fn + fp + 1e-6)

    return prec,rec,acc,F1,IoU

@numba.jit(nopython=True)
def _label_converter(pre_lables):
    new_labels = np.zeros_like(pre_lables)
    for idx, point_label in enumerate(pre_lables):
        if point_label in [40, 44, 48, 49, 60]:
            new_labels[idx] = 0  # ground points
        else:
            new_labels[idx] = 1  # non-ground points
    return new_labels.reshape(-1, 1)

# ---------------------------------------------------------------------------- #
# Load config ; declare Meter class, LearningRateSchedule class, checkpointer, etc.
# ---------------------------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument('-conf', '--configs',default='./configs/sector_conf.yaml',
                    help="Choose configs: ./configs/basic_conf.yaml, ./configs/unet_conf.yaml")
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


# load model
model = This_Net(cfg)
model.cuda()

pretrain_dir = os.path.join(cfg.checkpoints_path, model_name)
checkpoint = torch.load(pretrain_dir)
model.load_state_dict(checkpoint['state_dict'])

# load test dataset
test_point_dir = os.path.join(cfg.data_path, '08', 'velodyne')
test_point_list = os.listdir(test_point_dir)
test_label_dir = os.path.join(cfg.data_path,'08', 'labels')  # laod raw label file list
test_label_list = os.listdir(test_label_dir)
test_point_list.sort(key=lambda x: str(x[:-4]))
test_label_list.sort(key=lambda x: str(x[:-4]))

def test():
    model.eval()
    with torch.no_grad():
        time_list = []
        acc = []
        miou = []
        F1_score = []
        count = 0
        # for test_file in test_list:
        #     data_dir = os.path.join(test_dir, test_file)
        #     # data = np.load(data_dir)
        #     points = np.fromfile(data_dir, dtype=np.float32).reshape(-1,4)
        #     label_sector_dir = data_dir.replace(cfg.pc_folder, cfg.lb_folder)
        #     label_sector = np.load(label_sector_dir[:-4]+'.npy')

        for test_file in test_point_list:
            data_dir = os.path.join(test_point_dir, test_file)
            points = np.fromfile(data_dir, dtype=np.float32).reshape(-1,4) # load points
            label_dir = os.path.join(test_label_dir, test_file[:-3]+'label')
            labels = np.fromfile(label_dir, dtype=np.int32) # load corresponding labels
            new_labels = _label_converter(labels)
            assert points.shape[0]==new_labels.shape[0],' point file and label file are mismatched. '
            points_with_labels= np.hstack((points,new_labels)) # N x 5 (x,y,z,i,lb)

            points_with_labels = np.array(
                [x for x in points_with_labels if 0 < math.sqrt(x[0] ** 2 + x[1] ** 2) - 3.0 < 47.0 and 0 < x[2] + 3.0 < 5.0])

            if len(points_with_labels) >= cfg.max_points:
                points_with_labels = points_with_labels[:cfg.max_points, :]  # if points are more than 100000
            else:
                points_with_labels = np.pad(points_with_labels, ((0, cfg.max_points - len(points_with_labels)), (0, 0)), 'constant', constant_values=0)

            points_sector = points_with_labels[:, :4]
            labels_gt = points_with_labels[:,4].reshape(-1,1)

            point_feature_in_sector = []
            coors_sectors = []
            num_sectors = []

            start_time = time.time()

            p, c, n = points_to_sector_dynamic_ops(points_sector,
                                                   sector_shape=[64, 256, 1],
                                                   max_points_in_sector=100,
                                                   max_sector=10000)

            point_feature_in_sector.append(torch.from_numpy(p))
            c = torch.from_numpy(c)
            c = F.pad(c, (1, 0), 'constant', 0)
            coors_sectors.append(c)
            num_sectors.append(torch.from_numpy(n))

            point_feature_in_sector = torch.cat(point_feature_in_sector).float().cuda()
            coors_sectors = torch.cat(coors_sectors).float().cuda()
            num_sectors = torch.cat(num_sectors).float().cuda()

            output = model(point_feature_in_sector, coors_sectors, num_sectors)
            end_time = time.time()
            time_list.append(end_time - start_time)
            pred_image = output[0].argmax(0).cpu().numpy() # 64 x 256 image

            pred_labels=[]
            sector_shape = [64, 256, 1]
            sector_size = [0.625, 1.40625]  # 64x256  0.625 is not used in dynamic partition.

            # grow_rate
            rad_id_list = np.zeros((64,))
            rad_id_list[0] = 3.0
            gr = 0.0275
            for j in range(1, 64):
                rad_id_list[j] = rad_id_list[j - 1] + gr * j

            for i in range(len(points_sector)):
                rad_val = math.sqrt(points_sector[i, 0] ** 2 + points_sector[i, 1] ** 2)  # 点到圆心距离
                ang_id = np.floor((math.atan2(points_sector[i, 0], points_sector[i, 1]) / math.pi * (180 / sector_size[1]))) \
                         + sector_shape[1] / 2  # 0~255 start from 3rd phase. same to fixed partition

                cnt_start = 0
                for dis in rad_id_list:
                    if (rad_val - dis) < 0:
                        rad_id = cnt_start
                        break
                    elif (rad_val > rad_id_list[-1]):
                        rad_id = 63
                    cnt_start += 1

                if ang_id > 255 or ang_id < 0:
                    print("[Warning] ang_id out of range : ")
                    ang_id = 0

                if rad_id > 63 or rad_id < 0:
                    print("[Warning] rad_id out of range :")
                    rad_id = 0

                # print(ang_id,rad_id)

                ang_id = ang_id.astype(int)
                pred_label = pred_image[rad_id, ang_id]
                pred_labels.append(pred_label)

            pred_labels = np.array(pred_labels).reshape(-1,1)

            pred_points_with_labels = np.hstack((points_sector,pred_labels))
            np.save(cfg.data_path +'08/pred_label_sector/'+test_file[-10:-4], pred_points_with_labels) # 4+1 成为5列的数组 为了可视化。

            # 问题1： 计算时使用哪种gt？点gt该如何保证对应？
            # 使用sector 作为gt时
            pred_flatten = pred_labels
            label_flatten = labels_gt

            # prec,rec,acc,F1,IoU

            _,_,acc_tem,F1_score_tem,miou_tem = metrics(gt=labels_gt, pre=pred_labels)
            _,_,acc_tem,F1_score_tem,miou_tem = metrics(gt=label_flatten, pre=pred_flatten)

            acc.append(acc_tem)
            F1_score.append(F1_score_tem)
            miou.append(miou_tem)
            if count % 10 == 0:
                print("- %d frames inference processed." % count)
                print("- ACC:", acc_tem)
                print("- F1_score:", F1_score_tem)
                print("- FPS:", 1 / (end_time - start_time))

            count = count + 1

        acc = np.array(acc)
        miou = np.array(miou)
        F1_score = np.array(F1_score)
        time_list = np.array(time_list)

        print("Summary:")
        print('-- ACC:', acc.sum() / len(acc))
        print('-- IOU:', miou.sum() / len(miou))
        print('-- F1_score:', F1_score.sum() / len(F1_score))
        print('-- FPS:',  len(time_list)/ time_list.sum())
        print("Test Completed.")

        if not os.path.exists(os.path.join(cfg.evaluation_path, cfg.model_name)):
            os.mkdir(os.path.join(cfg.evaluation_path, cfg.model_name))

        np.save(os.path.join(cfg.evaluation_path, cfg.model_name, 'Acc'), acc)  # 需要修改
        np.save(os.path.join(cfg.evaluation_path, cfg.model_name, 'mIoU'), miou)
        np.save(os.path.join(cfg.evaluation_path, cfg.model_name, 'F1_score'), F1_score)
        np.save(os.path.join(cfg.evaluation_path, cfg.model_name, 'Time'), time_list)

if __name__ == '__main__':
    test()
