<div align="center">
<h3>RCDINO: Enhancing Radar-Camera 3D Object Detection with DINOv2 Semantic Features</h3>
<h4>Olga Matykina and Dmitry Yudin<h4>
<h5>MIPT, AIRI<h5>
</div>


## Introduction

This repository is an official implementation of RCDINO.

## Environment Setting using Docker
```
cd ./docker
build.sh
change start.sh
start.sh
into.sh
cd ./RCDINO/projects/mmdet3d_plugin/models/backbones/nets/ops
make.sh
cd ~/RCDINO
```
## Data Preparation
```
python3 tools/create_data_nusc.py --root-path ../HPR3/nuscenes --out-dir ../HPR3 --extra-tag nuscenes_radar --version v1.0

python3 tools/create_data_nusc.py --root-path ../HPR3/nuscenes --out-dir ../HPR3 --extra-tag nuscenes_radar --version v1.0-mini
```
Folder structure
```
HPR3
├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
|   ├── nuscenes_radar_temporal_infos_train.pkl
|   ├── nuscenes_radar_temporal_infos_val.pkl
|   ├── nuscenes_radar_temporal_infos_test.pkl
RCDINO
├── projects/
├── mmdetection3d/
├── tools/
├── ckpts/
```
Or you can directly use pre-generated pickles here. [Val](https://drive.usercontent.google.com/download?id=1CLs4zi2tmkBl33XzEkvmUDT9an-2N9c5&export=download&authuser=0&confirm=t&uuid=22c1cee9-3b91-4b7f-84b8-fd69aae10224&at=APvzH3oFQ5HqwWzKXsSTckzGP1gP:1734076238954) [Train](https://drive.usercontent.google.com/download?id=1m2rggU4jzuBPDPfCbC3u0G5ugD-e8P9t&export=download&authuser=0&confirm=t&uuid=61169d3e-e31b-4ad7-920c-3a746eceba74&at=APvzH3qPOu74S9o-v19hxWgZU-ku:1734076306697) [Test](https://drive.usercontent.google.com/download?id=1Xhc1DMbi67YsV7nis26GWOjxjVAmTF3o&export=download&authuser=0&confirm=t&uuid=86051653-5de3-4383-ab97-ab43f0ec93d1&at=APvzH3p-l9SdhykVspp5eDGxmLMa:1734076308824)
## Train & Inference
Train
```
export PYTHONPATH=$PYTHONPATH:/home/docker_rctrans/RCTrans/

bash tools/dist_train.sh projects/configs/RCTrans/rcdetr_90e_256×704_dino.py 1 --work-dir work_dirs/dinov2_with_resnet_from_res18/


```
Evaluation
```
bash tools/dist_test.sh projects/configs/RCTrans/rcdetr_90e_256×704_dino.py ckpts/NDS_epoch_16.pth 1 --eval bbox

```

Visualize
```
python tools/visualize.py
# We also recommand to use the Visualization codes from BEVFormer, which is really nice.
```
## Weights
Download this backbone: [ResNet-18](https://drive.usercontent.google.com/download?id=1QWb74xrZ-HbywXvrLrYjs7hhCBheTS7n&export=download&authuser=0&confirm=t&uuid=6fb7c908-a33a-4bad-879f-25186fb67f14&at=APvzH3pcvUeKZrjbQ7WM818Dv41p:1734079499612), [DINOv2-small](https://drive.google.com/file/d/1lnYYfuZmLbGqKkIkLwPprIEz6IFuTApQ/view?usp=sharing) and put it into the RCDINO/ckpts/.

We give the pre-trained: [ResNet18-train](https://drive.usercontent.google.com/download?id=1zcvGfBU7j6eLi00ho0VXFCrZG1i5YEmL&export=download&authuser=0&confirm=t&uuid=8aced929-3827-4e9d-9968-ff7873b326a2&at=APvzH3pennsYULNE1cWgSQ1gMCaA:1734080088576), [DINOv2-ResNet18-train](https://drive.google.com/file/d/120tCj-jN2YrdTyQT2WK8hoPqUiwQ8qDG/view?usp=sharing)
## Acknowledgements
We thank these great works and open-source codebases:
[MMDetection3d](https://github.com/open-mmlab/mmdetection3d), [BEVFormer](https://github.com/fundamentalvision/BEVFormer),
[DETR3D](https://github.com/WangYueFt/detr3d), [PETR](https://github.com/megvii-research/PETR),
[StreamPETR](https://github.com/exiawsh/StreamPETR), [CMT](https://github.com/junjie18/CMT), [CenterPoint](https://github.com/tianweiy/CenterPoint), [FUTR3D](https://github.com/Tsinghua-MARS-Lab/futr3d).
[RCTrans](https://github.com/liyih/RCTrans)

```