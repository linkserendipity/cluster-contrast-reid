CUDA_VISIBLE_DEVICES=2,3 python examples/cluster_contrast_train_usl.py \
-b 256 \
-a resnet_ibn50a -d msmt17 \
--iters 400 --momentum 0.1 --eps 0.7 \
--num-instances 16 \
--epochs 50 \
--eval-step 5 \
--pooling-type gem --use-hard \
--logs-dir /mnt/HDD2/ls/cluster-contrast-reid/examples/logs/msmt_ibn50a_b256_eps07_gem_hard_epoch50 \
--print-freq 100
# --lr ??
# -a resnet50
# --pooling-type gem --use-hard \ 