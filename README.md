# NeSy-without-Shortcuts
Code for the paper "Learning with Logical Constraints but without Shortcut Satisfaction"

### Requirements

```
numpy
pytorch
```

### Usage

For each task, run the following command.

> python logic_training.py --num_labeled 100 --constraint True --trun True --exp_name logic_trun \
>
> ​											--net_type densenet100 --adam_lr 0.01 --constraint_weight 1.0

where the parameter "trun" is set to True to enable the truncation of Gaussian distribution.

To reproduce the experimental results, Run the command `sh run.sh`

### Related work

- We also provide baseline methods in MNIST task, one can refer to `/mnist_exp/baselines` for more details.

- Our another work (see paper Softened Symbol Grounding for Neuro-symbolic Systems) can also avoid the shortcuts, but do not need to additionally define the dual variable. 

