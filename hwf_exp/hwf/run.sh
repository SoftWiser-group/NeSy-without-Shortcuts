CUDA_VISIBLE_DEVICES=0 python baseline_training.py --data_used 0.02 --seed 0 --exp_name baseline_0.02_wd00001 --constraint_weight 0.0
CUDA_VISIBLE_DEVICES=0 python baseline_training.py --data_used 0.05 --seed 0 --exp_name baseline_0.05_wd00001 --constraint_weight 0.0

CUDA_VISIBLE_DEVICES=1 python logic_training.py --data_used 0.02 --num_unlabel 0.8 --seed 0 --exp_name logic_trun_0.02_wd00001 --constraint True --target_sigma 0.5
CUDA_VISIBLE_DEVICES=1 python logic_training.py --data_used 0.05 --num_unlabel 0.2 --seed 0 --exp_name logic_trun_0.05_wd00001 --constraint True --target_sigma 0.5


CUDA_VISIBLE_DEVICES=8 python logic_testing.py --resume_from sym_net_0_logic_trun_0.05_wd00001_600_

