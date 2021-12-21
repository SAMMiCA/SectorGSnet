# SectorGSnet: Sector Learning for Efficient Ground Segmentation of Outdoor LiDAR Point Clouds
This is the official source code for SectorGSnet.

## Dependencies

    Python 3.7  
    CUDA (tested on 10.2)
    PyTorch (tested on 1.7)
    argparse
    numba
  

## Data Preparation
We train our model using the SematicKITTI dataset. please find the SemanticKITTI dataset from their [website]("www.semantic-kitti.org").

## Training & Testing
Hyper paramters for training to update in configuration file: /configs/sector_conf.yaml

    python train_sector.py
    python test_sector.py

## Results


## TODO

-  Need to improve the reading speed of the point cloud
-  The current version contains some test code and needs to be streamlined.



## Acknowledgments

- This work was supported by Institute for Information & communications Technology Promotion (IITP) grant funded by the Korea government (MSIT) (No.2020-0-00440, Development of artificial intelligence technology that continuously improves itself as the situation changes in the real world).
