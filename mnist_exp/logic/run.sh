#!/bin/bash

# ==============================================================================================================================================
# train lenet model
# python logic_training.py --exp_name baseline --net_type lenet --adam_lr 0.001 --seed 0 --constraint_weight 0.0
# python logic_training.py --exp_name sup --net_type lenet --adam_lr 0.001 --seed 0 --constraint_weight 0.0

python logic_training.py --constraint True --trun True --exp_name logic_trun --net_type lenet --adam_lr 0.001 --seed 0 --constraint_weight 1.0

# python logic_curve.py --constraint True --trun True --exp_name logic_curve --net_type lenet --adam_lr 0.001 --seed 0 --constraint_weight 1.0

