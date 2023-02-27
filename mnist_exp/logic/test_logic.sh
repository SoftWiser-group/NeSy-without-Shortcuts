#!/bin/bash


# run evaluation
# CUDA_VISIBLE_DEVICES=0,1 python logic_testing.py --resume_from lenet_0_baseline_400_best --dataset 'mnist'
# CUDA_VISIBLE_DEVICES=0,1 python logic_testing.py --resume_from lenet_0_baseline_400_best --dataset 'usps'

CUDA_VISIBLE_DEVICES=0,1 python logic_testing.py --resume_from lenet_0_logic_trun_400_best --dataset 'mnist'
# CUDA_VISIBLE_DEVICES=0,1 python logic_testing.py --resume_from lenet_0_logic_trun_400_best --dataset 'usps'

# CUDA_VISIBLE_DEVICES=0,1 python logic_testing.py --resume_from lenet_0_baselinefull_400_best --dataset 'mnist'
# CUDA_VISIBLE_DEVICES=0,1 python logic_testing.py --resume_from lenet_0_baselinefull_400_best --dataset 'usps'

