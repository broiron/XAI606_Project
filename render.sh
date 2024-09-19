python train_ed.py --config './configs/dds_fortress.txt' --render_only 1 --basedir './log4/fortress_snowy/' --ckpt './log4/fortress_snowy/fortress_ed/fortress_ed.th' --svae_ckpt "./log4/fortress_snowy/fortress_ed/fortress_ed_svae.th" --ckpt "./log4/fortress_snowy/fortress_ed/fortress_ed.th" --render_all 1 --render_test 1
python train_ed.py --config './configs/dds_fortress.txt' --render_only 1 --basedir './log4/fortress_burn/' --ckpt './log4/fortress_burn/fortress_ed/fortress_ed.th' --svae_ckpt "./log4/fortress_burn/fortress_ed/fortress_ed_svae.th" --ckpt "./log4/fortress_burn/fortress_ed/fortress_ed.th" --render_all 1 --render_test 1
python train_ed.py --config './configs/dds_fortress.txt' --render_only 1 --basedir './log4/fortress_cat/' --ckpt './log4/fortress_cat/fortress_ed/fortress_ed.th' --svae_ckpt "./log4/fortress_cat/fortress_ed/fortress_ed_svae.th" --ckpt "./log4/fortress_cat/fortress_ed/fortress_ed.th" --render_all 1 --render_test 1
python train_ed.py --config './configs/dds_fortress.txt' --render_only 1 --basedir './log4/fortress_shoe/' --ckpt './log4/fortress_shoe/fortress_ed/fortress_ed.th' --svae_ckpt "./log4/fortress_shoe/fortress_ed/fortress_ed_svae.th" --ckpt "./log4/fortress_shoe/fortress_ed/fortress_ed.th" --render_all 1 --render_test 1

python train_ed.py --config './configs/dds_fortress.txt' --render_only 1 --basedir './log4/fortress_lego/' --ckpt './log4/fortress_lego/fortress_ed/fortress_ed.th' --svae_ckpt "./log4/fortress_lego/fortress_ed/fortress_ed_svae.th" --ckpt "./log4/fortress_lego/fortress_ed/fortress_ed.th" --render_all 1 --render_test 1

# 1. fortress
python train_ed.py --config './configs/fortress.txt' --render_only 1 --basedir './log4/fortress_cat/' --ckpt './log4/core_w_vae/fortress/fortress_ed.th' --svae_ckpt "./log4/core_w_vae/fortress/fortress_ed_svae.th" --ckpt "./log4/core_w_vae/fortress/fortress_ed.th" --render_all 1 --render_test 1 --render_latent 1

python train_ed.py --config './configs/flower.txt' --render_only 1 --basedir './log_flower/core_w_vae/' --ckpt './log_flower/core_w_vae/flower/flower_ed.th' --svae_ckpt "./log_flower/core_w_vae/flower/flower_ed_svae.th" --ckpt "./log_flower/core_w_vae/flower/flower_ed.th" --render_all 1 --render_test 1 --render_latent 1

python train_ed.py --config './configs/dds_flower.txt' --render_only 1 --basedir './log_flower/flower_cherry_blossoms' --ckpt './log_flower/flower_cherry_blossoms/flower_ed/flower_ed.th' --svae_ckpt "./log_flower/flower_cherry_blossoms/flower_ed/flower_ed_svae.th" --render_all 1 --render_test 1 
python train_ed.py --config './configs/dds_flower.txt' --render_only 1 --basedir './log_flower/flower_rose' --ckpt './log_flower/flower_rose/flower_ed/flower_ed.th' --svae_ckpt "./log_flower/flower_rose/flower_ed/flower_ed_svae.th" --render_all 1