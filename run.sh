#!/bin/bash
python3 train.py  --config /home/LPF/external/lpf/fasterrcnn-pytorch-training-pipeline-main/data_configs/coco.yaml --epochs 81 --model faster_RCNN_abation_resnet101_SM_loss --project-name ssdd_training_Ablation_resnet101_SM_loss --batch-size 4