# CUDA_VISIBLE_DEVICES=0,1 python examples/cluster_contrast_train_usl.py \
# -b 256 -a resnet50 -d market1501 \
# --iters 200 --momentum 0.1 --eps 0.4 \
# --num-instances 16 \
# --data-dir /mnt/SSD/ls/data \
# --logs-dir /mnt/HDD2/ls/cluster-contrast-reid/examples/logs/market_resnet50_b256_eps04  最后不能加\ 不然会报错
#! -a resnet_ibn50a
CUDA_VISIBLE_DEVICES=2,3 python examples/cluster_contrast_train_usl.py \
-b 512 -a resnet_ibn50a -d market1501 \
--iters 400 --momentum 0.1 --eps 0.4 \
--num-instances 16 \
--pooling-type gem --use-hard \
--epochs 45 \
--eval-step 5 \
--logs-dir /mnt/HDD2/ls/cluster-contrast-reid/examples/logs/market_ibn50a_b512_eps04_gem_hard_epoch45