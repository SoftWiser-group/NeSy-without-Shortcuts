CUDA_VISIBLE_DEVICES=0,1 python logic_training.py --exp_name baseline --adam_lr 0.001 --net_type resnet18  --seed 0 --constraint_weight 0.0
CUDA_VISIBLE_DEVICES=2,3 python logic_training.py --constraint True --trun True --adam_lr 0.001 --exp_name logic_trun --net_type resnet18  --seed 0 --constraint_weight 1.0

CUDA_VISIBLE_DEVICES=6 python logic_testing.py --resume_from resnet18_0_baseline_400_best --dataset cifar10
CUDA_VISIBLE_DEVICES=6 python logic_testing.py --resume_from resnet18_0_baseline_400_best --dataset stl10

CUDA_VISIBLE_DEVICES=6 python logic_testing.py --resume_from resnet18_0_logic_trun_400_best --dataset cifar10
CUDA_VISIBLE_DEVICES=6 python logic_testing.py --resume_from resnet18_0_logic_trun_400_best --dataset stl10