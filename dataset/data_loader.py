import os
import numpy as np
import math
from torch.utils.data import Dataset
from multiprocessing import Manager

class kitti_loader(Dataset):
    def __init__(self, data_dir, pc_folder='', lb_folder='',
                 train=1, skip_frames=1, maxpoint=100000, cache_size=10000):
        self.pointcloud_path = []
        self.label_path = []
        self.maxPoints = maxpoint

        # self.partiton_type =partiton_type
        if train:
            seq = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
            for seq_num in seq:
                folder_pc = os.path.join(data_dir, seq_num, pc_folder)
                folder_lb = os.path.join(data_dir, seq_num, lb_folder)
                file_pc = os.listdir(folder_pc)
                file_pc.sort(key=lambda x: str(x[:-4]))
                file_lb = os.listdir(folder_lb)
                file_lb.sort(key=lambda x: str(x[:-4]))
                for index_pc in range(0, len(file_pc), skip_frames):
                    self.pointcloud_path.append('%s/%s' % (folder_pc, file_pc[index_pc]))
                for index_lb in range(0, len(file_lb), skip_frames):
                    self.label_path.append('%s/%s' % (folder_lb, file_lb[index_lb]))
        else:
            seq = '08'
            folder_pc = os.path.join(data_dir, seq, pc_folder)
            folder_lb = os.path.join(data_dir, seq, lb_folder)
            file_pc = os.listdir(folder_pc)
            file_pc.sort(key=lambda x: str(x[:-4]))
            file_lb = os.listdir(folder_lb)
            file_lb.sort(key=lambda x: str(x[:-4]))
            for index_pc in range(0, len(file_pc), skip_frames):
                self.pointcloud_path.append('%s/%s' % (folder_pc, file_pc[index_pc]))
            for index_lb in range(0, len(file_lb), skip_frames):
                self.label_path.append('%s/%s' % (folder_lb, file_lb[index_lb]))

        print(len(self.pointcloud_path), len(self.label_path))
        assert len(self.pointcloud_path) == len(self.label_path)

    def jitter(self, pc, A, B, C, D):
        d=np.sqrt(pc[:,0]**2+pc[:,1]**2) #
        a_z=A*np.sin(B*d+C)+D
        pc[:,2]=pc[:,2]+a_z
        return pc

    def get_data(self, pointcloud_path, label_path, rough_dataset= False):
        points = np.fromfile(pointcloud_path, dtype=np.float32).reshape(-1, 4)
        labels = np.load(label_path)
        if rough_dataset:
            points= self.jitter(points,1.1,0.25,10,0)
        return points, labels # points are raw points, labels are converted one.

    def __len__(self):
        return len(self.pointcloud_path)

    def __getitem__(self, index):
        point, label = self.get_data(self.pointcloud_path[index], self.label_path[index])

        # round for sector partition
        point = np.array([x for x in point if 2.5 < np.sqrt(x[0] ** 2 + x[1] ** 2) < 50.0 and 0 < x[2] + 4 < 8.0])


        # skip frames with less points than 100k
        if len(point) >= self.maxPoints:
            point= point[:self.maxPoints,:]  # if points are more than 100000
        else:
            point= np.zeros((100000,4))
            label= -np.ones((64,256))

        return point, label # point: 100000x4 | label: 64 x 256
