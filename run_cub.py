#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GZSL
"""
import os
os.system('''OMP_NUM_THREADS=8  python train_free.py --gammaD 1 --gammaG 1 \
--gzsl --manualSeed 3483 --encoded_noise --preprocessing --cuda --image_embedding res101 --class_embedding sent \
--nepoch 301 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 1 --dataroot data --dataset CUB \
 --a1 1 --a2 1 --feed_lr 0.00001 --dec_lr 0.0001 --loop 2 \
--nclass_all 200 --nclass_seen 150 --batch_size 32 --nz 1024 --latent_size 1024 --attSize 1024 --resSize 768  \
--syn_num 300 --center_margin 200 --center_weight 0.5 --recons_weight 0.1 --incenter_weight 0.9  --GA_weight 1.0 \
--SD_weight 1.0 --NI_weight 1.0 --head_number 4
''')


"""
CZSL
"""
# import os
# os.system('''OMP_NUM_THREADS=8  python train_free.py --gammaD 10 --gammaG 10 \
# --gzsl --manualSeed 3483 --encoded_noise --preprocessing --cuda --image_embedding res101 --class_embedding sent \
# --nepoch 301 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 1 --dataroot data --dataset CUB \
#  --a1 1 --a2 1 --feed_lr 0.00001 --dec_lr 0.0001 --loop 2 \
# --nclass_all 200 --nclass_seen 150 --batch_size 8 --nz 1024 --latent_size 1024 --attSize 1024 --resSize 768  \
# --syn_num 700 --center_margin 200 --center_weight 0.5 --recons_weight 0.001 --incenter_weight 0.999  --GA_weight 1.0 \
# --SD_weight 1.0 --NI_weight 1.0 --head_number 4
# ''')
