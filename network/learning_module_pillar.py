# Baseline
import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["FeatureNet", "PointPillarsScatter",]

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


class FeatureNet(nn.Module):
    def __init__(self):
        """
        Base on Pillar Feature Net.
        """
        super().__init__()
        self.fn = PN(4, 32)

    def forward(self, features, coors, num_points):
        # features: p x n x 4 (x,y,z,i)
        sector_all = self.fn(features)
        sector_all = sector_all.squeeze().transpose(0, 1)
        return sector_all

class PointPillarsScatter(nn.Module):
    def __init__(self):
        """
        modified from Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        """
        super().__init__()
        self.nx = 128  # 64
        self.ny = 128 # 256
        self.nchannels = 32 # output channels

    def forward(self, features, coords, batch_size):
        # batch_canvas will be the final output.
        batch_canvas = []
        for batch_id in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros((self.nchannels, self.nx, self.ny),
                                 dtype=features.dtype, device=features.device)
            # canvas = torch.zeros(self.nchannels, self.nx * self.ny,
            #                      dtype=features.dtype, device=features.device)
            # Only include non-empty pillars
            batch_mask = coords[:, 0] == batch_id #Px4(b,x,y,z)
            this_coords = coords[batch_mask, :]
            this_voxels = features[:, batch_mask] # p x 32

            # Now scatter the blob back to the canvas.
            canvas[:, this_coords[:, 1].type(torch.long), this_coords[:, 2].type(torch.long)] = this_voxels

            # Append to a list for later stacking.
            batch_canvas.append(canvas)
        # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        return batch_canvas #bs ,32, 128, 128 (batchsize, channels, height, width)

#
# class PointSectorScatter(nn.Module):
#     def __init__(self):
#         """
#         modified from Point Pillar's Scatter.
#         Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
#         second.pytorch.voxelnet.SparseMiddleExtractor.
#         """
#
#         super().__init__()
#         self.name = 'PointPillarsScatter'
#         self.nh = 64  # 64
#         self.nw = 256 # 256
#         self.nchannels = 32 # output channels
#
#     def forward(self, features, coords, batch_size):
#         # batch_canvas will be the final output.
#         batch_canvas = []
#         for batch_id in range(batch_size):
#             # Create the canvas for this sample
#             canvas = torch.zeros(( self.nchannels, self.nh, self.nw),
#                                  dtype=features.dtype, device=features.device)
#             # Only include non-empty pillars
#             batch_mask = coords[:, 0] == batch_id #Px4(b,x,y,z)
#             this_coords = coords[batch_mask, :]
#             this_voxels = features[:, batch_mask ] # 32 x P
#
#             # Now scatter the blob back to the canvas.
#             canvas[:,this_coords[:,1].type(torch.long),this_coords[:,2].type(torch.long)] = this_voxels
#             # test= canvas.cpu().detach().numpy()
#             # print(test)
#
#             # Append to a list for later stacking.
#             batch_canvas.append(canvas)
#         # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
#         batch_canvas = torch.stack(batch_canvas, 0)
#
#         return batch_canvas #bs ,32, 64, 256 (batchsize, channels, height, width)

