# Faster R-CNN


## pretrained weights download
* ResNet50+FPN weights: https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth

## training procedure
* train on one gpu: `python3.8 train_resnet50_fpn.py`
* train on multi gpus: `python -m torch.distributed.launch --nproc_per_node=8 --use_env train_multi_GPU.py`

## inference procedure
* `python3.8 predict.py`, with arguments passed with saving weights directory path.
