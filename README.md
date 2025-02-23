# FPGA Implementation of Double-Head SalsaNext: A CNN-Based Model for LiDAR Point Cloud Segmentation

This repository contains code to FPGA implementation of a customized double-head version of SalsaNext model for semantic segmentation from LIDAR range images. The figure below shows the architecture of the model.
<p align="center">
  <img src="figures/sonsalsa9.png" alt="Example Image" width="60%" />
</p>

The proposed pipeline is given below. For further details please check the paper.
<p align="center">
  <img src="figures/all3.png" alt="Example Image" width="90%" />
</p>


## Quantization and Deployment

Please refer to https://github.com/dymnmysn/MiDaSFPGA.git
After training, the steps are the same. Only the model is different which can be incorporated with a few line of code change.


## Cite This Work

Please cite our paper if you use this code or any of the models:
```
@ARTICLE {hydrafpga,
    author  = "Adiyaman MY., Baskaya F. ",
    title   = "FPGA Implementation of Double-Head SalsaNext: A CNN-Based Model for LiDAR Point Cloud Segmentation",
    journal = "...",
    year    = "2025",
    volume  = "...",
    number  = "..."
}
```


### License 

Apache 2.0 License 
