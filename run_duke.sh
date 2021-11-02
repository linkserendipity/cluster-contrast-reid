CUDA_VISIBLE_DEVICES=0,1 python examples/cluster_contrast_train_usl.py \
-b 256 -a resnet50 -d dukemtmcreid \
--iters 200 --momentum 0.1 --eps 0.7 \
--num-instances 16 \
--epochs 90 \
--logs-dir /mnt/HDD2/ls/cluster-contrast-reid/examples/logs/duke_resnet50_b128_eps07_epoch90
# CUDA_VISIBLE_DEVICES=0,1 python examples/cluster_contrast_train_usl.py -b 256 -a resnet_ibn50a -d dukemtmcreid --iters 400 --momentum 0.1 --eps 0.6 --num-instances 16 --pooling-type gem --use-hard --logs-dir /mnt/HDD2/ls/cluster-contrast-reid/examples/logs/duke_resnet_ibn50a_b256_eps06_gem_hard