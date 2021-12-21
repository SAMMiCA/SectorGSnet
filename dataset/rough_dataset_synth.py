import numpy as np
import os
from tqdm import tqdm
import open3d as o3d
from multiprocessing import Process

root_path="/root/dataset/kitti/sequences"

class Uneven():
    def __init__(self, pc_path):
        self.pc_path= pc_path
        self.pcs = os.listdir(os.path.join(pc_path,'velodyne'))

    def jitter(self, pc, A, B, C, D):
        d=(pc[:,0]*pc[:,0]+pc[:,1]*pc[:,1])**0.5 #
        a_z=A*np.sin(B*d+C)+D
        pc[:,2]=pc[:,2]+a_z
        return pc

    def run(self):
        for pc in tqdm(self.pcs):
            data = np.fromfile(os.path.join(self.pc_path,'velodyne', pc), dtype=np.float32).reshape(-1,4)
            data = self.jitter(data,1.1,0.25,10,0) # fine tuning
            np.save(os.path.join(self.pc_path, 'pc_uneven', pc[:6]), np.array(data))

if __name__ == '__main__':
    # check the existing of rough dataset folder
    print("start to synthetise rough dataset")
    seqs=['00','01','02','03','04','05','06','07','08','09','10']
    # seqs=['04'] # 使用04号做测试。
    # part_length = {'00': 4541, '01': 1101, '02': 4661, '03': 801, '04': 271, '05': 2761,
    #                    '06': 1101, '07': 1101, '08': 4071, '09': 1591, '10': 1201}
    pc_dir=[]
    lb_dir=[]
    def check_path(root_path):
        if not os.path.exists(os.path.join(root_path, 'pc_uneven')):
            os.mkdir(os.path.join(root_path, 'pc_uneven'))

    for seq in seqs:
        check_path(os.path.join(root_path, seq))
        call_uneven = Uneven(pc_path = os.path.join(root_path, seq))
        call_uneven.run()
    #     for index in range(part_length[seq]):
    #         pc_dir.append(os.path.join(root_path,seq,'pc_uneven','ue_%06d.npy'%index))
    #         lb_dir.append(os.path.join(root_path,seq,'labels','%06d.label'%index))
    # print(len(pc_dir), len(lb_dir))

    # pc_uneven_test_file = "../dataset/000250.npy"
    # pc_uneven = np.load(pc_uneven_test_file).reshape(-1,4)
    # pcd_dis= o3d.geometry.PointCloud()
    # pcd_dis.points= o3d.utility.Vector3dVector(pc_uneven[:,:3])
    # o3d.visualization.draw_geometries([pcd_dis])

