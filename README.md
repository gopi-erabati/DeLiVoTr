# DeLiVoTr: Deep and Light-weight Voxel Transformer for 3D Object Detection

This is the official PyTorch implementation of the paper **DeLiVoTr: Deep and Light-weight Voxel Transformer for 3D Object Detection**, by Gopi Krishna Erabati and Helder Araujo.

**Contents**
1. [Overview](https://github.com/gopi-erabati/DeLiVoTr/edit/master/README.md#overview)
2. [Results](https://github.com/gopi-erabati/DeLiVoTr/edit/master/README.md#results)
3. [Requirements, Installation and Usage](https://github.com/gopi-erabati/DeLiVoTr/edit/master/README.md#requirements-installation-and-usage)
4. [Acknowlegements](https://github.com/gopi-erabati/DeLiVoTr/edit/master/README.md#acknowlegements)

## Overview
Existing feature extraction networks for LiDAR-based 3D object detection down-stride the LiDAR feature maps similar to 2D feature extraction networks. Such down-striding of feature maps in large scale autonomous driving scenarios will hinder the detection of small size objects, such as *pedestrians*. To solve this issue we design an architecture that not only maintains the same stride but also the receptive field in the feature extraction network. We resort to attention mechanism to build sufficient receptive field and we propose a **De**ep and **Li**ght-weight **Vo**xel **Tr**ansformer (DeLiVoTr) network with voxel intra- and inter-region transformer modules to extract voxel local and global features respectively. The DeLiVoTr block is the core of the DeLiVoTr network which uses DeLighT transformations with expand and reduce strategy to vary the width and depth of network efficiently. This facilitates to learn wider and deeper voxel representations and enables to use not only smaller dimension for attention mechanism but also a light-weight feed-forward network, facilitating the reduction of parameters and operations. In addition to *model* scaling, we employ *layer-level* scaling for efficient parameter allocation in each encoder layer instead of fixed number of parameters as in existing approaches. Leveraging *layer-level depth* and *width* scaling we formulate three variants of DeLiVoTr network. We conduct extensive experiments and analysis on large-scale Waymo and KITTI datasets. Our network surpasses state-of-the-art methods for detection of small objects (*pedestrian*) with an inference speed of 20.5 FPS.

![DeLiVoTr](https://github.com/gopi-erabati/DeLiVoTr/assets/22390149/063b18ea-891f-4d3a-a08f-dbfe2cc3ef47)

## Results

### Predictions on Waymo dataset

![DeLiVoTr_-Deep-and-Light-weight-Voxel-Transformer-for-3D-Object-Detection](https://github.com/gopi-erabati/DeLiVoTr/assets/22390149/284661e0-542c-4817-8e85-8a248a6bf168)

| Config | Veh. L1 AP/APH | Veh. L2 AP/APH | Ped. L1 AP/APH | Ped. L2 AP/APH | Cyc. L1 AP/APH | Cyc. L2 AP/APH |
| :---:  | :---:  | :---:  | :---:  | :---:  | :---:  | :---:  |
| DeLiVoTr Waymo | 73.4/72.8 | 65.0/64.5 | 79.2/70.2 | 71.7/63.4 | 68.5/66.8 | 65.9/64.2 |

| Config | Ped. easy | Ped. mod. | Ped. hard | Cyc. easy | Cyc. mod. | Cyc. hard |
| :---:  | :---:  | :---:  | :---:  | :---:  | :---:  | :---:  |
| DeliVoTr KITTI | 75.2 | 69.6 | 64.4 | 87.6 | 64.6 | 61.3 |

## Requirements, Installation and Usage

### Prerequisite

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
```
mkvirtualenv delivotr

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

pip install -U openmim
mim install mmdet3d==1.0.0.rc6
pip install spconv-cu117
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install waymo-open-dataset-tf-2-2-0

pip install numpy==1.19.5
pip install pandas==1.1.5
pip install open3d==0.15.2
pip install ipdb
pip install protobuf==3.19.6
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
cp bazel-bin/waymo_open_dataset/metrics/tools/compute_detection_metrics_main ../DeLiVoTr/lib//core/evaluation/waymo_utils/
```

### Data
Follow [MMDetection3D-1.0.0.rc6](https://github.com/open-mmlab/mmdetection3d/tree/v1.0.0rc6) to prepare the [Waymo](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/waymo.html) and [KITTI](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/kitti.html) datasets and symlink the data directories to `data/` folder of this repository.
**Warning:** Please strictly follow [MMDetection3D-1.0.0.rc6](https://github.com/open-mmlab/mmdetection3d/tree/v1.0.0rc6) code to prepare the data because other versions of MMDetection3D have different coordinate refactoring.

### Clone the repository
```
git clone https://github.com/gopi-erabati/DeLiVoTr.git
cd DeLiVoTr
```

### Training
#### Waymo dataset 
- Single GPU training
    1. `export PYTHONPATH=$(pwd):$PYTHONPATH`
    2. `python tools/train.py configs/delivotr_waymo.py --work-dir {WORK_DIR}`
- Multi GPU training
  `tools/dist_train.sh configs/delivotr_waymo.py 2 --work-dir {WORK_DIR}`
#### KITTI dataset
- Single GPU training
    1. `export PYTHONPATH=$(pwd):$PYTHONPATH`
    2. `python tools/train.py configs/delivotr_kitti.py --work-dir {WORK_DIR}`
- Multi GPU training
  `tools/dist_train.sh configs/delivotr_kitti.py 2 --work-dir {WORK_DIR}`

### Testing
#### Waymo dataset 
- Single GPU testing
    1. `export PYTHONPATH=$(pwd):$PYTHONPATH`
    2. `python tools/test.py configs/delivotr_waymo.py /path/to/ckpt --eval waymo`
- Multi GPU training
  `tools/dist_test.sh configs/delivotr_waymo.py /path/to/ckpt 2 --eval waymo`
#### KITTI dataset
- Single GPU testing
    1. `export PYTHONPATH=$(pwd):$PYTHONPATH`
    2. `python tools/test.py configs/delivotr_kitti.py /path/to/ckpt --eval bbox`
- Multi GPU training
  `tools/dist_test.sh configs/delivotr_kitti.py 2 /path/to/ckpt 2 --eval bbox`

## Acknowlegements
We sincerely thank the contributors for their open-source code: [MMCV](https://github.com/open-mmlab/mmcv), [MMDetection](https://github.com/open-mmlab/mmdetection), [MMDetection3D](https://github.com/open-mmlab/mmdetection3d), [SST](https://github.com/tusen-ai/SST), [DeLighT](https://github.com/sacmehta/delight).

