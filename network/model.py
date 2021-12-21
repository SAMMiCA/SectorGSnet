import torch
from torch import nn

from network.learning_module import FeatureNet, PointSectorScatter
from network.segment_unet import UNet
from network.segment_munet import MUNet_lite, MUNet
from network.segment_cgnet import Context_Guided_Network

class This_Net(nn.Module):
    def __init__(self, cfg):
        super(This_Net, self).__init__()
        self.cfg = cfg
        self.feature_extractor = FeatureNet()
        self.middle_feature_extractor = PointSectorScatter()
        # self.encoder_decoder = UNet(cfg.pseudo_features, 2)
        # self.encoder_decoder = Context_Guided_Network(classes=2)
        # self.encoder_decoder = MUNet(cfg.pseudo_features, 2)
        self.encoder_decoder = MUNet_lite(cfg.pseudo_features, 2)

    def forward(self, point_feature_in_sector, coors_sectors, num_sectors):
        # print(point_feature_in_sector.shape, coors_sectors.shape,num_sectors.shape)
        sector_all_features = self.feature_extractor(point_feature_in_sector, coors_sectors, num_sectors)
        spatial_features = self.middle_feature_extractor(sector_all_features, coors_sectors, self.cfg.batch_size) #
        pred = self.encoder_decoder(spatial_features)

        return pred   # pred: batch size x 2 x 64 x 256