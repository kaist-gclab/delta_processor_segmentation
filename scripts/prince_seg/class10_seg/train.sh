#!/usr/bin/env bash

## run the training
python3 train.py \
--dataroot datasets/pclass10 \
--name pclass10 \
--arch meshunet \
--dataset_mode segmentation \
--ncf 32 64 128 256 \
--ninput_edges 1500 \
--pool_res 1050 600 300 \
--resblocks 3 \
--lr 0.001 \
--batch_size 4 \
--num_aug 20 \
--slide_verts 0.2 \


#
# python train.py --dataroot datasets/coseg_vases --name coseg_vases --arch meshunet --dataset_mode
# segmentation --ncf 32 64 128 256 --ninput_edges 1500 --pool_res 1050 600 300 --resblocks 3 --lr 0.001 --batch_size 12 --num_aug 20
