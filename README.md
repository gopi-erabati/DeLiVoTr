# DeLiVoTr: Deep and Light-weight Voxel Transformer for 3D Object Detection

This is the official PyTorch implementation of the paper **DeLiVoTr: Deep and Light-weight Voxel Transformer for 3D Object Detection**.

Gopi Krishna Erabati and Helder Araujo, "DeLiVoTr: Deep and light-weight voxel transformer for 3D object detection," in _Intelligent Systems with Applications_, Volume 22, 2024, 200361, ISSN 2667-3053, [https://doi.org/10.1016/j.iswa.2024.200361](https://doi.org/10.1016/j.iswa.2024.200361).

**Contents**
1. [Overview](https://github.com/gopi-erabati/DeLiVoTr#overview)
2. [Results](https://github.com/gopi-erabati/DeLiVoTr#results)
3. [Requirements, Installation and Usage](https://github.com/gopi-erabati/DeLiVoTr#requirements-installation-and-usage)
    1. [Prerequistes](https://github.com/gopi-erabati/DeLiVoTr#prerequisites)
    2. [Installation](https://github.com/gopi-erabati/DeLiVoTr#installation)
    3. [Training](https://github.com/gopi-erabati/DeLiVoTr#training)
    4. [Testing](https://github.com/gopi-erabati/DeLiVoTr#testing)
4. [Acknowlegements](https://github.com/gopi-erabati/DeLiVoTr#acknowlegements)
5. [Citation](https://github.com/gopi-erabati/DeLiVoTr#citation)

## Overview
The image-based backbone (feature extraction) networks downsample the feature maps not only to increase the receptive field but also to efficiently detect objects of various scales. The existing feature extraction networks in LiDAR-based 3D object detection tasks follow the feature map downsampling similar to image-based feature extraction networks to increase the receptive field. But, such downsampling of LiDAR feature maps in large-scale autonomous driving scenarios hinder the detection of small size objects, such as *pedestrians*. To solve this issue we design an architecture that not only maintains the same scale of the feature maps but also the receptive field in the feature extraction network to aid for efficient detection of small size objects. We resort to attention mechanism to build sufficient receptive field and we propose a **De**ep and **Li**ght-weight **Vo**xel **Tr**ansformer (DeLiVoTr) network with voxel intra- and inter-region transformer modules to extract voxel local and global features respectively. We introduce DeLiVoTr block that uses transformations with expand and reduce strategy to vary the width and depth of the network efficiently. This facilitates to learn wider and deeper voxel representations and enables to use not only smaller dimension for attention mechanism but also a light-weight feed-forward network, facilitating the reduction of parameters and operations. In addition to *model* scaling, we employ *layer-level* scaling of DeLiVoTr encoder layers for efficient parameter allocation in each encoder layer instead of fixed number of parameters as in existing approaches. Leveraging *layer-level depth* and *width* scaling we formulate three variants of DeLiVoTr network. We conduct extensive experiments and analysis on large-scale Waymo and KITTI datasets. Our network surpasses state-of-the-art methods for detection of small objects (*pedestrians*) with an inference speed of 20.5 FPS.

![DeLiVoTr](https://github.com/gopi-erabati/DeLiVoTr/assets/22390149/063b18ea-891f-4d3a-a08f-dbfe2cc3ef47)

## Results

### Predictions on Waymo dataset

![DeLiVoTr_-Deep-and-Light-weight-Voxel-Transformer-for-3D-Object-Detection](https://github.com/gopi-erabati/DeLiVoTr/assets/22390149/284661e0-542c-4817-8e85-8a248a6bf168)

| Config | Veh. L1 AP/APH | Veh. L2 AP/APH | Ped. L1 AP/APH | Ped. L2 AP/APH | Cyc. L1 AP/APH | Cyc. L2 AP/APH |
| :---:  | :---:  | :---:  | :---:  | :---:  | :---:  | :---:  |
| [DeLiVoTr Waymo](configs/delivotr_waymo.py) | 73.4/72.8 | 65.0/64.5 | 79.2/70.2 | 71.7/63.4 | 68.5/66.8 | 65.9/64.2 |

We can not distribute the model weights on Waymo dataset due to the [Waymo license terms](https://waymo.com/open/terms).

| Config | Ped. easy | Ped. mod. | Ped. hard | Cyc. easy | Cyc. mod. | Cyc. hard | |
| :---:  | :---:  | :---:  | :---:  | :---:  | :---:  | :---:  | :---: |
| [DeLiVoTr KITTI](configs/delivotr_kitti.py) | 75.2 | 69.6 | 64.4 | 87.6 | 64.6 | 61.3 | [model](https://drive.google.com/file/d/1MJYIhJ6ujHBgwud3vX_YsoRRTBVM1aaZ/view?usp=sharing) |

## Requirements, Installation and Usage

### Prerequisites

The code is tested on the following configuration:
- Ubuntu 20.04.6 LTS
- CUDA==11.7
- Python==3.8.10
- PyTorch==1.13.1
- [mmcv](https://github.com/open-mmlab/mmcv)==1.7.0
- [mmdet](https://github.com/open-mmlab/mmdetection)==2.28.2
- [mmseg](https://github.com/open-mmlab/mmsegmentation)==0.30.0
- [mmdet3d](https://github.com/open-mmlab/mmdetection3d)==1.0.0.rc6

### Installation

### Clone the repository
```
git clone https://github.com/gopi-erabati/DeLiVoTr.git
cd DeLiVoTr
```

```
mkvirtualenv delivotr

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -U openmim
mim install mmcv-full==1.7.0
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.1+cu117.html

pip install -r requirements.txt
```
Also, please install [TorchEx](https://github.com/Abyssaledge/TorchEx)

For evaluation on Waymo, please follow the below code to build the binary file `compute_detection_metrics_main` for metrics computation and put it into ```lib/core/evaluation/waymo_utils/```.
```
# download the code and enter the base directory
git clone https://github.com/waymo-research/waymo-open-dataset.git waymo-od
# git clone https://github.com/Abyssaledge/waymo-open-dataset-master waymo-od # if you want to use faster multi-thread version.
cd waymo-od
git checkout remotes/origin/master

# use the Bazel build system
sudo apt-get install --assume-yes pkg-config zip g++ zlib1g-dev unzip python3 python3-pip
BAZEL_VERSION=3.1.0
wget https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
sudo bash bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
sudo apt install build-essential

# configure .bazelrc
./configure.sh
# delete previous bazel outputs and reset internal caches
bazel clean

bazel build waymo_open_dataset/metrics/tools/compute_detection_metrics_main
cp bazel-bin/waymo_open_dataset/metrics/tools/compute_detection_metrics_main ../DeLiVoTr/lib/core/evaluation/waymo_utils/
```

### Data
Follow [MMDetection3D-1.0.0.rc6](https://github.com/open-mmlab/mmdetection3d/tree/v1.0.0rc6) to prepare the [Waymo](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/waymo.html) and [KITTI](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/kitti.html) datasets and symlink the data directories to `data/` folder of this repository.
**Warning:** Please strictly follow [MMDetection3D-1.0.0.rc6](https://github.com/open-mmlab/mmdetection3d/tree/v1.0.0rc6) code to prepare the data because other versions of MMDetection3D have different coordinate refactoring.

### Training
#### Waymo dataset 
- Single GPU training
    1. `export PYTHONPATH=$(pwd):$PYTHONPATH`
    2. `python tools/train.py configs/delivotr_waymo.py --work-dir {WORK_DIR}`
- Multi GPU training
  `tools/dist_train.sh configs/delivotr_waymo.py {GPU_NUM} --work-dir {WORK_DIR}`
#### KITTI dataset
- Single GPU training
    1. `export PYTHONPATH=$(pwd):$PYTHONPATH`
    2. `python tools/train.py configs/delivotr_kitti.py --work-dir {WORK_DIR}`
- Multi GPU training
  `tools/dist_train.sh configs/delivotr_kitti.py {GPU_NUM} --work-dir {WORK_DIR}`

### Testing
#### Waymo dataset 
- Single GPU testing
    1. `export PYTHONPATH=$(pwd):$PYTHONPATH`
    2. `python tools/test.py configs/delivotr_waymo.py /path/to/ckpt --eval waymo`
- Multi GPU training
  `tools/dist_test.sh configs/delivotr_waymo.py /path/to/ckpt {GPU_NUM} --eval waymo`
#### KITTI dataset
- Single GPU testing
    1. `export PYTHONPATH=$(pwd):$PYTHONPATH`
    2. `python tools/test.py configs/delivotr_kitti.py /path/to/ckpt --eval mAP`
- Multi GPU training
  `tools/dist_test.sh configs/delivotr_kitti.py /path/to/ckpt {GPU_NUM} --eval mAP`

## Acknowlegements
We sincerely thank the contributors for their open-source code: [MMCV](https://github.com/open-mmlab/mmcv), [MMDetection](https://github.com/open-mmlab/mmdetection), [MMDetection3D](https://github.com/open-mmlab/mmdetection3d), [SST](https://github.com/tusen-ai/SST), [DeLighT](https://github.com/sacmehta/delight).

## Citation
```BibTeX
@article{ERABATI2024200361,
author = {Gopi Krishna Erabati and Helder Araujo},
title = {DeLiVoTr: Deep and light-weight voxel transformer for 3D object detection},
journal = {Intelligent Systems with Applications},
volume = {22},
pages = {200361},
year = {2024},
issn = {2667-3053},
doi = {https://doi.org/10.1016/j.iswa.2024.200361},
url = {https://www.sciencedirect.com/science/article/pii/S2667305324000371},
}
```

