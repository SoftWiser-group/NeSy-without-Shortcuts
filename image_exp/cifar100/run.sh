#!/bin/bash


# ==============================================================================================================================================
# train Densenet model
python baseline_training.py --num_labeled 100 --exp_name baseline --constraint False --net_type densenet100 --seed 0 --constraint_weight 0.0
python logic_training.py --num_labeled 100 --constraint True --trun True --exp_name logic_trun --net_type densenet100 --adam_lr 0.01 --seed 0 --constraint_weight 1.0

python logic_testing.py --resume_from densenet100_0_baseline_800_best --net_type densenet100 --tol 1e-2
python logic_testing.py --resume_from densenet100_0_logic_trun_3600_best --net_type densenet100 --tol 1e-2

