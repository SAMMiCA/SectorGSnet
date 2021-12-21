import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["FeatureNet", "PointSectorScatter"]

class Conv1x1(nn.Module):
    def __init__(self, in_channels=160, out_channels=32):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        out = self.relu(x)
        return out

class PN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, inputs):
        # inputs.shape :  sector x points(100) x features
        x = self.linear(inputs) # sector x points(100) x 32
        x = self.norm(x.permute(0, 2, 1).contiguous())  # to bn 32 features. not points. (p x 32 x 100)
        x = F.relu(x)
        x_max = torch.max(x, dim=2, keepdim=True)[0] # (p x 32 x 100) -> (p x 32 x 1)
        return x_max


class MMPN(nn.Module):  # Multi Modal PointNet: local feature extractor
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size= 1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.linear = nn.Linear(128, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = inputs.permute(0, 2, 1).contiguous()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.linear(x)
        x = self.bn2(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x = self.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        return x_max # s x 1 x 32

# def get_paddings_indicator(actual_num, max_num, axis=0):
#     actual_num = torch.unsqueeze(actual_num, axis + 1)
#     # tiled_actual_num: [N, M, 1]
#     max_num_shape = [1] * len(actual_num.shape)
#     max_num_shape[axis + 1] = -1
#     max_num = torch.arange(
#             max_num,
#             dtype = torch.int,
#             device = actual_num.device).view(max_num_shape)
#     # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
#     # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
#     paddings_indicator = actual_num.int() > max_num
#     # paddings_indicator shape: [batch_size, max_num]
#     return paddings_indicator

class FeatureNet(nn.Module):
    def __init__(self):
        """
        Base on Pillar Feature Net.
        """
        super().__init__()

        # # Create FeatureNet layers
        # # All-in-one
        self.fn = PN(9, 32)

        # Multimod
        # self.fn_xyz = PN(3, 32) # p x n x 3 -> p x 32 x 1
        # self.fn_i = PN(1, 32) # p x n x 1 -> p x 32 x 1
        # self.fn_d = PN(1, 32) # p x n x 1 -> p x 32 x 1
        # self.fn_r = PN(1, 32) # p x n x 1 -> p x 32 x 1
        # self.fn_cxyz = PN(3, 32) # p x n x 3 -> p x 32 x 1
        # self.conv1x1= Conv1x1(32*5,32) # p x n x 3 -> p x 32 x 1

        # # Backup
        # self.pfn_layers = nn.ModuleList(pfn_layers)
        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        # self.vx = voxel_size[0]
        # self.vy = voxel_size[1]
        # self.x_offset = self.vx / 2 + pc_range[0]
        # self.y_offset = self.vy / 2 + pc_range[1]

    def forward(self, features, coors, num_points):
        # features: p x n x 6 (x,y,z,i,rad,dis)
        # num_points: p x 1

        # Find distance of x, y, and z from pillar center 这个不能用？可以用半径来算?
        # not available in sector
        # f_center = torch.zeros_like(features[:, :, :2])  # f_center: nx100x2  ; coors 前两列都是0
        # f_center[:, :, 0] = features[:, :, 0] - (coors[:, 3].float().unsqueeze(1) * self.vx + self.x_offset)
        # f_center[:, :, 1] = features[:, :, 1] - (coors[:, 2].float().unsqueeze(1) * self.vy + self.y_offset)

        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_points.type_as(features).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean  # f_cluster: px100x3

        ########################################
        # # Multimod approach
        #
        # # input: x,y,z
        # # output: distance of x, y, and z from cluster center
        # features_xyz = features[:, :, :3] # pxnx3
        # features_sector_core = f_cluster # pxnx3
        # features_intensity = torch.unsqueeze(features[:,:,3],2) # pxnx1
        # features_radius = torch.unsqueeze(features[:,:,4],2) # pxnx1
        # features_distance = torch.unsqueeze(features[:,:,5],2) #pxnx1
        #
        # sector_xyz = self.fn_xyz(features_xyz)
        # sector_core= self.fn_cxyz(features_sector_core)
        # sector_intensity = self.fn_i(features_intensity)
        # sector_distance = self.fn_d(features_distance)
        # sector_radius= self.fn_r(features_radius)
        # sector_cat= torch.cat([sector_xyz,sector_intensity,sector_radius,
        #                                        sector_distance,sector_core],dim=1)
        # sector_all = self.conv1x1(sector_cat)  # P x 160 x1 -> P x 32 x 1
        # sector_all = sector_all.squeeze().transpose(0, 1)
        # return sector_all  # 32 x p

###################################################
        # # All-in-one approach
        feature_all = torch.cat((features, f_cluster), dim=2) #6+3
        sector_all = self.fn(feature_all)
        sector_all = sector_all.squeeze().transpose(0, 1)
        return sector_all

        # sector_cat= torch.transpose(torch.cat([sector_xyz, sector_intensity, sector_radius,
        #                                        sector_distance, sector_core], dim=1), 1, 2)


class PointSectorScatter(nn.Module):
    def __init__(self):
        """
        modified from Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        """
        super().__init__()
        self.nh = 64  # 64
        self.nw = 256 # 256
        self.nchannels = 32 # output channels

    def forward(self, features, coords, batch_size):
        # batch_canvas will be the final output.
        batch_canvas = []
        for batch_id in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(( self.nchannels, self.nh, self.nw),
                                 dtype=features.dtype, device=features.device)
            # Only include non-empty pillars
            batch_mask = coords[:, 0] == batch_id #Px4(b,x,y,z)
            this_coords = coords[batch_mask, :]
            this_voxels = features[:, batch_mask ] # 32 x P

            # Now scatter the blob back to the canvas.
            canvas[:,this_coords[:,1].type(torch.long),this_coords[:,2].type(torch.long)] = this_voxels

            # Append to a list for later stacking.
            batch_canvas.append(canvas)
        # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        return batch_canvas #bs ,32, 64, 256 (batchsize, channels, height, width)

