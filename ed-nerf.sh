python train.py --config './configs/basis/fortress.txt' --n_iters 50000 --device "cuda:0" --basedir './log/core_w_vae' --n_iters 1000

python train_ed.py --config './configs/dds_fortress.txt' --target_prompt 'a marshmallow fortress on the table' --ref_prompt 'a small castle on the table'
