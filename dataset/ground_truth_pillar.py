# reviewed at 0916
# Add multiprocessing
# modified for AI28 lidar
import os
import numpy as np
import numba
import math
from multiprocessing import Process

#########################################################
# Convert original labels to pillar labels
# -1 for null pillars, 0 for ground pillars; 1 for non-ground pillars
#########################################################


# convert_seq=['10']
convert_seq = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
data_dir= '/root/dataset/kitti/sequences/'
# data_dir = '/home/dong/dataset/sequences/'
workers = 4  # numbers of thread
part_type = '/labels_pillar_0916/'


class kitti_path():  # modified for groundtruth
    def __init__(self, data_dir=data_dir, skip_frames=1):
        self.pointcloud_path = []
        self.label_path = []
        for seq in convert_seq:
            folder_pc = os.path.join(data_dir, str(seq), 'velodyne')
            folder_lb = os.path.join(data_dir, str(seq), 'labels')
            file_pc = os.listdir(folder_pc)
            file_pc.sort(key=lambda x: str(x[:-4]))
            file_lb = os.listdir(folder_lb)
            file_lb.sort(key=lambda x: str(x[:-4]))
            for index in range(0, len(file_pc), skip_frames):
                self.pointcloud_path.append('%s/%s' % (folder_pc, file_pc[index]))
                self.label_path.append('%s/%s' % (folder_lb, file_lb[index]))
    def get_path(self):
        return self.pointcloud_path, self.label_path

class kitti_loader():
    def __init__(self, pointcloud_path, label_path):
        self.pointcloud_path = pointcloud_path
        self.label_path = label_path
        self.max_points = 100000

    def get_data(self, pointcloud_path, label_path):
        point = np.fromfile(pointcloud_path, dtype=np.float32).reshape(-1, 4)
        label = np.fromfile(label_path, dtype=np.int32).reshape(-1, 1)
        point_label= np.hstack((point,label))  # provent dismatching
        return point_label

    def getitem(self):
        point_label = self.get_data(self.pointcloud_path, self.label_path)
        # square
        point_label = np.array(
            [x for x in point_label if 0 < x[0] + 51.2 < 102.4 and 0 < x[1] + 51.2 < 102.4 and 0< x[2]+2.0 < 6.0])
        # point = np.array([x for x in point if 0 < x[0] + 51.2 < 102.4 and 0 < x[1] + 51.2 < 102.4 and 0< x[2]+4.0 < 8.0])

        if len(point_label) >= self.max_points:
            point_label = point_label[:self.max_points,:]
        else:
            point_label=np.pad(point_label,((0,self.max_points- len(point_label)),(0,0)), constant_values=0)

        label_file_idx = self.label_path
        return point_label[:,:3], point_label[:,-1], label_file_idx

@numba.jit(nopython=True)
def _label_converter(pre_lables):
    new_labels = np.zeros_like(pre_lables)
    for idx, point_label in enumerate(pre_lables):
        if point_label in [40, 44, 48, 49, 60]:
            new_labels[idx] = 1  # ground points
        else:
            new_labels[idx] = 2  # non-ground points
    return new_labels.reshape(-1, 1)

@numba.jit(nopython=True)
def _points_to_voxel_kernel_groundtruth(points,
                                                 coor_to_sectoridx,
                                                 sector_features,
                                                 sector_coors,
                                                 num_points_per_sector,
                                                 max_points_in_sector=100,
                                                 max_sector=10000):
    voxel_num = 0
    voxel_size = [0.8, 0.8]  # 128x128
    coors_range = [-51.2, 51.2, -51.2, 51.2]
    voxel_coor = np.zeros(shape=(3,), dtype=np.int32)

    for i in range(points.shape[0]):
        x_id = np.floor((points[i, 0]-coors_range[0])/voxel_size[0]) # id: 0~127
        y_id = np.floor((points[i, 1]-coors_range[2])/voxel_size[1]) # id: 0~127
        z_id = 0

        # consider unexpected cases
        if x_id > 127 :
            print("[Warning] x_id out of range : ")
            print(points[i, 0], x_id)
            x_id = 127
        if x_id  < 0 :
            print("[Warning] x_id out of range : ")
            print(points[i, 0], x_id)
            x_id = 0
        if y_id > 127:
            print("[Warning] y_id out of range : ")
            print(points[i, 1], y_id)
            y_id = 127
        if y_id < 0:
            print("[Warning] y_id out of range : ")
            print(points[i, 1], y_id)
            y_id = 0

        voxel_coor[0] = x_id  # 128
        voxel_coor[1] = y_id  # 128
        voxel_coor[2] = z_id    # ignore z for our case
        sectoridx = coor_to_sectoridx[voxel_coor[0], voxel_coor[1], voxel_coor[2]]  # 128x128x1
        if sectoridx == -1:
            sectoridx = voxel_num
            if voxel_num >= max_sector:
                break
            voxel_num += 1
            coor_to_sectoridx[voxel_coor[0], voxel_coor[1], voxel_coor[2]] = sectoridx # voxel index in 128 x 128
            sector_coors[sectoridx] = voxel_coor # 128 x128
        num = num_points_per_sector[sectoridx]
        if num < max_points_in_sector:
            sector_features[sectoridx, num] = points[i]
            num_points_per_sector[sectoridx] += 1
    return voxel_num


def points_to_voxel_groundtruth(points,
                                         sector_shape=[128, 128, 1],
                                         max_points_in_sector=100,
                                         max_sector=10000):

    sector_shape = tuple(np.round(sector_shape).astype(np.int32).tolist())  # 128x128x1
    coor_to_sectoridx = - np.ones(shape=sector_shape, dtype=np.int32)  # 128x128
    num_points_per_sector = np.zeros(shape=(max_sector,), dtype=np.int32)  # 100000
    sector_features = np.zeros(
        shape=(max_sector, max_points_in_sector, points.shape[-1]), dtype=points.dtype)  # 10000*100*4(convered labels)
    sector_coors = np.zeros(shape=(max_sector, 3), dtype=np.int32)  # 10000*3
    sector_num = _points_to_voxel_kernel_groundtruth(
        points, coor_to_sectoridx,
        sector_features, sector_coors, num_points_per_sector,
        max_points_in_sector, max_sector)

    sector_labels = []
    pts_in_sector = sector_features[:sector_num]  # p x n x 4 # only for non-null pillar

    #  convere point label to sector label
    sector_raw_labels = pts_in_sector[:, :, -1]
    for pts in sector_raw_labels: # ranking
        lb = np.array([x for x in pts if x])
        count_dict = {}
        for i in lb:
            if i in count_dict:
                count_dict[i] += 1
            else:
                count_dict[i] = 1
        dictSortList = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
        label = dictSortList[0][0] # get first index in ranking
        # label = 0 if label > 1.1 else 1  # index =0
        sector_labels.append(label)
    sector_labels = np.array(sector_labels).reshape(-1, 1)
    new_pillar = np.hstack((sector_coors[:sector_num], sector_labels))  # p x 4 (x,y,z,sector_label)
    return new_pillar


class Process(Process):
    def __init__(self, pc_path, label_path, num, workers):
        super(Process, self).__init__()
        self.pc_path = pc_path
        self.label_path = label_path
        self.num = num
        self.workers = workers

    def run(self):
        counter = 0
        while counter * self.workers + self.num < len(self.pc_path):
            data_loader = kitti_loader(self.pc_path[counter * self.workers + self.num],
                                       self.label_path[counter * self.workers + self.num])
            point, label, label_idx = data_loader.getitem()
            point = point[:, :2]  # only need x,y
            convt_label = _label_converter(label) # convert 32 classes to 2 classes
            point_label = np.hstack((point, convt_label)) # get Points with new labels
            sector = points_to_voxel_groundtruth(points = point_label)

            # make new folder to save groundtruth
            labels_folder = np.char.split(label_idx, '/')
            labels_folder_idx = labels_folder.item(0)[-3]
            new_label_folder = data_dir + labels_folder_idx + part_type
            if not os.path.exists(new_label_folder):
                os.mkdir(new_label_folder)
            label_file_name = label_idx[-12:-6]

            pillar_labels = -np.ones((128, 128)) # -1 is empty
            # for i in range(sector.shape[0]):
            pillar_labels[sector[:,0].astype(int), sector[:,1].astype(int)] = sector[:,3]-1 # 0 for ground points; 1 for non-ground points
            np.save(new_label_folder + label_file_name, pillar_labels)
            if counter % 20 ==0:
                print("seq {} No.{} converted~".format(labels_folder_idx, label_file_name))
            counter += 1

def main():
    process_list = []
    pc_path, label_path = kitti_path().get_path()
    for num in range(workers):
        pr = Process(pc_path, label_path, num, workers)
        pr.start()
        process_list.append(pr)
    for pr in process_list:
        pr.join()

if __name__ == '__main__':
    main()
