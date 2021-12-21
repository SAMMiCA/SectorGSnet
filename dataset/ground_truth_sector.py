# Reviewed 0916
# Add multiprocessing
# modified for AI28 lidar
import os
import numpy as np
import numba
import math
from multiprocessing import Process

#########################################################
# Convert original labels to sector labels
# -1 for null sectors, 0 for ground sectors; 1 for non-ground sectors
#########################################################

# convert_seq=['10']
convert_seq = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
data_dir= '/root/dataset/kitti/sequences/'
# data_dir = '/home/dong/dataset/sequences/'
workers = 4  # numbers of thread
part_type = '/labels_sector_1220/'


class kitti_path():  # modified for groundtruth
    def __init__(self, data_dir=data_dir, skip_frames=1):
        self.pointcloud_path = []
        self.label_path = []
        for seq in convert_seq:
            folder_pc = os.path.join(data_dir, str(seq), 'pc_uneven')
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
        # point = np.fromfile(pointcloud_path, dtype=np.float32).reshape(-1, 4)
        point = np.load(pointcloud_path).reshape(-1,4)
        label = np.fromfile(label_path, dtype=np.int32).reshape(-1, 1)
        point_label = np.hstack((point, label))
        return point_label

    def getitem(self):
        point_label = self.get_data(self.pointcloud_path, self.label_path)
        # round
        point_label = np.array(
            [x for x in point_label if 3.0 < math.sqrt(x[0] ** 2 + x[1] ** 2) < 50.0 and 0 < x[2] + 2.0 < 6.0])

        if len(point_label) >= self.max_points:
            point_label = point_label[:self.max_points,:]
        else:
            point_label=np.pad(point_label,((0,self.max_points- len(point_label)),(0,0)), constant_values=0)

        label_file_idx = self.label_path
        return point_label[:,:3], point_label[:,-1], label_file_idx

@numba.jit(nopython=True)
def _label_converter(pre_lables):
    points_new_labels = np.zeros_like(pre_lables)
    for idx, point_label in enumerate(pre_lables):
        if point_label in [40, 44, 48, 49, 60]:
            points_new_labels[idx] = 1  # ground points
        else:
            points_new_labels[idx] = 2  # non-ground points
    return points_new_labels.reshape(-1,1)

@numba.jit(nopython=True)
def _points_to_sector_dynamic_kernel_groundtruth(points,
                                                 coor_to_sectoridx,
                                                 sector_features,
                                                 sector_coors,
                                                 num_points_per_sector,
                                                 max_points_in_sector=100,
                                                 max_sector=10000):
    sector_num = 0
    sector_shape = [64, 256, 1]
    sector_size = [0.625, 1.40625]  # 64x256
    sector_coor = np.zeros(shape=(3,), dtype=np.int32)

    # generate id list for radius
    rad_id_list = np.zeros((64,))
    rad_id_list[0] = 3.0 # start point
    gr = 0.0275  # ratio
    for j in range(1, 64):
        rad_id_list[j] = rad_id_list[j - 1] + gr * j

    # # atan
    # for j in range(64):
    #     rad_id_list[j] = math.tan(math.pi * (65 + 0.35 * j) / 180) * 1.75

    for i in range(points.shape[0]):
        rad_val = math.sqrt(points[i, 0] ** 2 + points[i, 1] ** 2) # radius value
        cnt_start = 0
        for dis in rad_id_list:  # get rad_id
            if (rad_val - dis) < 0:
                rad_id = cnt_start
                break
            elif (rad_val > rad_id_list[-1]):
                rad_id = 63
            cnt_start += 1
        ang_id = np.floor((math.atan2(points[i, 0], points[i, 1]) / math.pi * (180 / sector_size[1]))) \
                 + sector_shape[1] / 2  # 0 ~ 255. start from 3rd phase. same to fixed partition
        height_id = 0

        if ang_id > 255 or ang_id < 0:
            print("[Warning] ang_id out of range : ")
            print(points[i, 0], points[i, 1], ang_id)
            ang_id = 0

        if rad_id > 63 or rad_id < 0:
            print("[Warning] rad_id out of range :")
            print(points[i, 0], points[i, 1], rad_id)
            rad_id = 0

        sector_coor[0] = rad_id  # 64
        sector_coor[1] = ang_id  # 256
        sector_coor[2] = height_id  # ignore z for our case
        sectoridx = coor_to_sectoridx[sector_coor[0], sector_coor[1], sector_coor[2]]  # 64x256x1
        if sectoridx == -1:
            sectoridx = sector_num
            if sector_num >= max_sector:
                break
            sector_num += 1
            coor_to_sectoridx[sector_coor[0], sector_coor[1], sector_coor[2]] = sectoridx
            sector_coors[sectoridx] = sector_coor
        num = num_points_per_sector[sectoridx]
        if num < max_points_in_sector:
            sector_features[sectoridx, num] = points[i]
            num_points_per_sector[sectoridx] += 1
    return sector_num


def points_to_sector_dynamic_groundtruth(points,
                                         sector_shape=[64, 256, 1],
                                         max_points_in_sector=100,
                                         max_sector=10000):

    sector_shape = tuple(np.round(sector_shape).astype(np.int32).tolist())  # 64x256x1
    coor_to_sectoridx = -np.ones(shape=sector_shape, dtype=np.int32)  # 64x256
    num_points_per_sector = np.zeros(shape=(max_sector,), dtype=np.int32)  # 100000
    sector_features = np.zeros(
        shape=(max_sector, max_points_in_sector, points.shape[-1]), dtype=points.dtype)  # 10000*100*4(+convered labels)
    sector_coors = np.zeros(shape=(max_sector, 3), dtype=np.int32)  # 10000*3

    sector_num = _points_to_sector_dynamic_kernel_groundtruth(
        points, coor_to_sectoridx,
        sector_features, sector_coors, num_points_per_sector,
        max_points_in_sector, max_sector)

    sector_labels = []
    pts_in_sector = sector_features[:sector_num]  # p x n x 4 # only for non-null pillar
    #  convere point label to sector label
    sector_raw_labels = pts_in_sector[:, :, -1]
    for pts in sector_raw_labels:
        lb = np.array([x for x in pts if x])
        count_dict = {}
        for i in lb:
            if i in count_dict:
                count_dict[i] += 1
            else:
                count_dict[i] = 1
        dictSortList = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
        label = dictSortList[0][0]
        # label = 0 if label > 1.1 else 1
        sector_labels.append(label)
    sector_labels = np.array(sector_labels).reshape(-1, 1)
    new_sector = np.hstack((sector_coors[:sector_num], sector_labels))  # p x 4 (x,y,z,label)
    return new_sector


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
            point = point[:, :2] # only use x,y
            convt_label = _label_converter(label)  # convert 32 classes to 2 classes
            point_label = np.hstack((point, convt_label))  # get Points with new labels
            sector = points_to_sector_dynamic_groundtruth(points=point_label)

            # new_folder to save groundtruth
            labels_folder = np.char.split(label_idx, '/')
            labels_folder_idx = labels_folder.item(0)[-3]
            new_label_folder = data_dir + labels_folder_idx + part_type
            if not os.path.exists(new_label_folder):
                os.mkdir(new_label_folder)
            label_file_name = label_idx[-12:-6]

            sector_labels = -np.ones((64, 256)) # -1 is empty

            sector_labels[sector[:,0].astype(int), sector[:,1].astype(int)] = sector[:,3]-1 # 0 for ground points; 1 for non-ground points
            np.save(new_label_folder + label_file_name, sector_labels)
            if counter % 20 == 0:
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
