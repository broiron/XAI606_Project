python train_ed.py --config './configs/dds_fortress.txt' --target_prompt 'a snowy castle on the table' --ref_prompt 'a small castle on the table' --mask_prompt 'castle' --basedir './log4/fortress_snowy'

python train_ed.py --config './configs/dds_fortress.txt' --target_prompt 'a burning castle on the table' --ref_prompt 'a small castle on the table' --mask_prompt 'castle' --basedir './log4/fortress_burn'

python train_ed.py --config './configs/dds_fortress.txt' --target_prompt 'a shoe on the table' --ref_prompt 'a small castle on the table' --mask_prompt 'castle' --basedir './log4/fortress_shoe'

python train_ed.py --config './configs/dds_fortress.txt' --target_prompt 'a freeze castle on the table' --ref_prompt 'a small castle on the table' --mask_prompt 'castle' --basedir './log4/fortress_freeze'

python train_ed.py --config './configs/dds_fortress.txt' --target_prompt 'a small cat on the table' --ref_prompt 'a small castle on the table' --mask_prompt 'castle' --basedir './log4/fortress_cat'
