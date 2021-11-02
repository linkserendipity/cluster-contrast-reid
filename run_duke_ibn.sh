CUDA_VISIBLE_DEVICES=2,3 python examples/cluster_contrast_train_usl.py \
-b 256 -a resnet_ibn50a -d dukemtmcreid \
--iters 400 --momentum 0.1 --eps 0.6 \
--num-instances 16 \
--pooling-type gem --use-hard \
--epochs 45 \
--eval-step 5 \
--logs-dir /mnt/HDD2/ls/cluster-contrast-reid/examples/logs/duke_resnet_ibn50a_b256_eps06_gem_hard_epoch45
# epoch 50代不够吧???