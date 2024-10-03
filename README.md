# XAI606_Project
Korea Univ Applications and Practice in Neural Networks Course Project Repo

How to use this code? 

vary simple may -> sh ed-nerf.sh

0. Install requirement.txt

1. Traing ED-NeRF in latent space

python train.py --config './configs/basis/fortress.txt' --n_iters 50000 --device "cuda:0" --basedir './log/core_w_vae' --n_iters 1000

2. Editing NeRF with target prompt

python train_ed.py --config './configs/dds_fortress.txt' --target_prompt 'a marshmallow fortress on the table' --ref_prompt 'a small castle on the table' --mask_prompt 'castle'

you can change target prompt or reference prompt with your own idea!
Please enjoy ED-NeRF!
