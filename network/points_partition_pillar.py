import numba
import numpy as np
import math

__all__ = ["points_to_voxel"]

############################################
##   point to pillar from Pillarpoints    ##
############################################

@numba.jit(nopython=True)
def _points_to_voxel_kernel(points,
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
        rad_val = math.sqrt(points[i, 0] ** 2 + points[i, 1] ** 2)  # 点到圆心距离
        dis_val = math.sqrt(points[i, 0] ** 2 + points[i, 1] ** 2 + points[i, 2] ** 2)

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
        voxel_coor[2] = z_id  # ignore z for our case
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
    return voxel_num  # get non-empty voxel number

def points_to_voxel(points,
                    sector_shape=[128, 128, 1],
                    max_points_in_voxel=100,
                    max_voxel=10000): # total number of voxels: 128*128= 16384

    voxelmap_shape = np.round(sector_shape).astype(np.int32).tolist() # 128x128x1
    coor_to_voxelidx = - np.ones(shape=voxelmap_shape, dtype=np.int32)  # 128*128*1
    num_points_per_voxel = np.zeros(shape=(max_voxel,), dtype=np.int32) #10000*1
    voxel_features = np.zeros(
        shape=(max_voxel, max_points_in_voxel, points.shape[-1]), dtype=points.dtype) # 10000*100*4
    voxel_coors = np.zeros(shape=(max_voxel, 3), dtype=np.int32) #10000*3

    voxel_num = _points_to_voxel_kernel(
        points, coor_to_voxelidx,
        voxel_features, voxel_coors, num_points_per_voxel,
        max_points_in_voxel,max_voxel)

    pts_in_voxels = voxel_features[:voxel_num]  # voxel_num x 100 x 4
    voxel_coors = voxel_coors[:voxel_num] # voxel_num x3
    num_points_per_voxel = num_points_per_voxel[:voxel_num] #voxel_numx1
    return pts_in_voxels, voxel_coors, num_points_per_voxel


# def points_to_cylinders(points, #
#                      cyliner_size=[3.6,0.5,8], # [3.6, 0.5, 8] 表示扇形旋角度为3.6度，扇形水平距离为1米，高度为8米
#                      dist=50, #一个值，点距离激光雷达的最大水平距离
#                      max_points_in_cyliner=100, #扇形网格中点的最大数量
#                      max_cylinder=8000):
#
#     if not isinstance(cyliner_size, np.ndarray):
#         cyliner_size = np.array(cyliner_size, dtype=points.dtype) #3.6,0.5,8
#     cyliner_coors_range=np.array([360, dist , 8])
#     cylinermap_shape = cyliner_coors_range / cyliner_size # 100x100x1
#     cylinermap_shape = tuple(np.round(cylinermap_shape).astype(np.int32).tolist()) # 100x100x1
#
#     num_points_per_cylinder = np.zeros(shape=(max_cylinder, ), dtype=np.int32)
#     coor_to_cylinderidx = -np.ones(shape=cylinermap_shape, dtype=np.int32) # 100x100
#     cylinder = np.zeros(
#         shape=(max_cylinder, max_points_in_cyliner, points.shape[-1]+1),
#         dtype=points.dtype) #8000*100*4+1 add radius
#     coors = np.zeros(shape=(max_cylinder, 3), dtype=np.int32) #10000*3
#     cylinder_cont = 0
#     N = points.shape[0] # Number of raw points
#     for i in range(N):
#         coor = np.zeros(shape=(3,), dtype=np.int32)
#         rad_val= math.sqrt(points[i,0]**2+points[i,1]**2)
#         # segid=np.floor(math.atan2(points[i,1],points[i,0]) + math.pi/(voxel_size[0]/180*math.pi)) #????
#         ang_id=np.floor((math.atan2(points[i,1],points[i,0]) /math.pi*(180/cyliner_size[0]))+50)  # 起始点从第三象限开始逆时针旋转(0->99)
#         rad_id=np.floor(rad_val/cyliner_size[1]) #
#         height_id=np.floor((points[i,2]+cyliner_size[2]/2)/cyliner_size[2])
#
#         if ang_id >= 100:
#             print(str(i)+ "ang_id out of range")
#         elif rad_id >= 100:
#             print(str(i)+"rad_id out of range")
#
#         coor[0]=ang_id
#         coor[1]=rad_id
#         coor[2]=height_id
#
#         cylinderidx = coor_to_cylinderidx[coor[0], coor[1], coor[2]]
#
#         if cylinderidx==-1:
#             cylinderidx = cylinder_cont
#             if cylinder_cont >= max_cylinder:
#                 break
#             cylinder_cont+=1
#             coor_to_cylinderidx[coor[0], coor[1], coor[2]] = cylinderidx #占位
#             coors[cylinderidx] = coor
#         num = num_points_per_cylinder[cylinderidx]
#
#         if num < max_points_in_cyliner:
#             point=points[i].tolist()
#             point.append(rad_val)
#             cylinder[cylinderidx, num] = point
#             num_points_per_cylinder[cylinderidx] += 1
#
#     pts_in_cylinder = cylinder[:cylinder_cont]  # p x n x 4+1
#     cylinder_coors = coors[:cylinder_cont] # p x 3
#     num_points_per_cylinder = num_points_per_cylinder[:cylinder_cont] # p x 1
#     return pts_in_cylinder, cylinder_coors, num_points_per_cylinder


####重要备份—Start#####
# def points_to_cylinder_dynamic(points,  #
#                                cylinder_size=[3.6, [], 8],  # [3.6, 0.5, 8] 表示扇形旋角度为3.6度，扇形水平距离为1米，高度为8米
#                                lider_range=[360, 50, 8],  # 一个值，点距离激光雷达的最大水平距离
#                                cylinder_shape=[100, 100, 1],
#                                max_points=100,  # 扇形网格中点的最大数量
#                                max_cylinder=8000):
#     # define cylinder_shape first : 128,64,1,
#     cylinder_shape = tuple(np.round(cylinder_shape).astype(np.int32).tolist())  # 100x100x1
#     num_points_per_cylinder = np.zeros(shape=(max_cylinder,), dtype=np.int32)  # 80000
#     coor_to_cylinderidx = -np.ones(shape=cylinder_shape, dtype=np.int32)  # 100x100
#     cylinder = np.zeros(
#         shape=(max_cylinder, max_points, points.shape[-1] + 1), dtype=points.dtype)  # 8000*100*5
#     coors = np.zeros(shape=(max_cylinder, 3), dtype=np.int32)  # 8000*3
#     cylinder_num = 0
#     N = points.shape[0]  # Number of points in a frame
#     cylinder_size[1].append(0)
#     start_r = 0.1
#     k_r = 0.005
#
#     for j in range(1, 101):
#         temp = start_r + k_r * j
#         temp += cylinder_size[1][j - 1]
#         cylinder_size[1].append(temp)
#     ang_id_list = []
#     rad_id_list = []
#
#     for i in range(N):
#         coor = np.zeros(shape=(3,), dtype=np.int32)
#         rad_val = math.sqrt(points[i, 0] ** 2 + points[i, 1] ** 2)  # xy plane. 点到圆心距离
#         for dis in cylinder_size[1]:
#             if rad_val - dis < 0:
#                 rad_id = cylinder_size[1].index(dis) - 1  # 这个点分配到哪个bin
#                 break
#         ang_id = np.floor((math.atan2(points[i, 1], points[i, 0]) / math.pi * (
#                 180 / cylinder_size[0])) + 50)  # 起始点从第三象限开始逆时针旋转(0->99)
#         height_id = np.floor((points[i, 2] + cylinder_size[2] / 2) / cylinder_size[2])
#         ang_id_list.append(ang_id)
#         rad_id_list.append(rad_id)
#
#         if ang_id > 100:
#             print(str(i) + ": " + str(ang_id) + "ang_id out of range")
#         elif rad_id > 100:
#             print(str(i) + ": " + str(rad_id) + "rad_id out of range")
#
#         coor[0] = ang_id
#         coor[1] = rad_id
#         coor[2] = height_id
#         # 需要重新命名
#         cylinderidx = coor_to_cylinderidx[coor[0], coor[1], coor[2]]  # 100x100x1
#
#         if cylinderidx == -1:
#             cylinderidx = cylinder_num
#             if cylinder_num >= max_cylinder:
#                 break
#             cylinder_num += 1
#             coor_to_cylinderidx[coor[0], coor[1], coor[2]] = cylinderidx  # 记录个数在对应位置（一次）
#             coors[cylinderidx] = coor  # 记录cylinder坐标到coors(一次)
#
#         num = num_points_per_cylinder[cylinderidx]  # 与coor_to_cylinderidx 功能一样
#
#         if num < max_points:
#             point = points[i].tolist()
#             point.append(rad_val)
#             cylinder[cylinderidx, num] = point
#             num_points_per_cylinder[cylinderidx] += 1
#
#     pts_in_cylinder = cylinder[:cylinder_num]  # p x n x 4+1
#     cylinder_coors = coors[:cylinder_num]  # p x 3
#     num_points_per_cylinder = num_points_per_cylinder[:cylinder_num]  # p x 1
#     return pts_in_cylinder, cylinder_coors, num_points_per_cylinder# 共同点是 p
#
#
# def points_to_cylinder_fixed(points,
#                              # r= 40 #一个值，点距离激光雷达的最大水平距离
#                              cylinder_size=[3.6, 0.5, 8],  # 表示扇形旋角度为3.6度，扇形水平距离为1米，高度为8米
#                              cylinder_shape=[100, 100, 1],
#                              max_points_in_cylinder=100,  # 扇形网格中点的最大数量
#                              max_cylinder=8000):
#     # define cylinder_shape first : 128,64,1,
#
#     cylinder_shape = tuple(np.round(cylinder_shape).astype(np.int32).tolist())  # 100x100x1
#     num_points_per_cylinder = np.zeros(shape=(max_cylinder,), dtype=np.int32)  # 80000
#     coor_to_cylinderidx = -np.ones(shape=cylinder_shape, dtype=np.int32)  # 100x100
#     cylinder = np.zeros(
#         shape=(max_cylinder, max_points_in_cylinder, points.shape[-1] + 1), dtype=points.dtype)  # 8000*100*5
#     coors = np.zeros(shape=(max_cylinder, 3), dtype=np.int32)  # 8000*3
#     cylinder_num = 0
#     N = points.shape[0]  # Number of points in a frame
#
#     for i in range(N):
#         coor = np.zeros(shape=(3,), dtype=np.int32)
#         rad_val = math.sqrt(points[i, 0] ** 2 + points[i, 1] ** 2)  # 点到圆心距离
#         # rad_val= np.linalg.norm([points[i,0],points[i,1]])
#         rad_id = rad_val / cylinder_size[1]
#         ang_id = np.floor((math.atan2(points[i, 1], points[i, 0]) / math.pi * (
#                 180 / cylinder_size[0])) + 50)  # 起始点从第三象限开始逆时针旋转(0->99)
#         height_id = np.floor((points[i, 2] + cylinder_size[2] / 2) / cylinder_size[2])
#
#         if ang_id >= 100:
#             print(str(i) + "ang_id out of range")
#         elif rad_id >= 100:
#             print(str(i) + "rad_id out of range")
#
#         coor[0] = ang_id
#         coor[1] = rad_id
#         coor[2] = height_id
#
#         # 需要重新命名
#         cylinderidx = coor_to_cylinderidx[coor[0], coor[1], coor[2]]  # 100x100x1
#
#         if cylinderidx == -1:
#             cylinderidx = cylinder_num
#             if cylinder_num >= max_cylinder:
#                 break
#             cylinder_num += 1
#             coor_to_cylinderidx[coor[0], coor[1], coor[2]] = cylinderidx  # 记录个数在对应位置（一次）
#             coors[cylinderidx] = coor  # 记录cylinder坐标到coors(一次)
#
#         num = num_points_per_cylinder[cylinderidx]  # 与coor_to_cylinderidx 功能一样
#         if num < max_points_in_cylinder:
#             point = points[i].tolist()
#             point.append(rad_val)
#             cylinder[cylinderidx, num] = point
#             num_points_per_cylinder[cylinderidx] += 1
#
#     pts_in_cylinder = cylinder[:cylinder_num]  # p x n x 4+1+1
#     cylinder_coors = coors[:cylinder_num]  # p x 3
#     num_points_per_cylinder = num_points_per_cylinder[:cylinder_num]  # p x 1
#
#     return pts_in_cylinder, cylinder_coors, num_points_per_cylinder # 共同点是 p
####重要备份-End#####