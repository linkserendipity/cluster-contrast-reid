CUDA_VISIBLE_DEVICES=0,1 python examples/cluster_contrast_train_usl.py \
-b 256 -a resnet50 -d market1501 \
--iters 200 --momentum 0.1 --eps 0.4 \
--num-instances 16 \
--data-dir /mnt/SSD/ls/data