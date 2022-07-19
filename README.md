# EqR: Equivariant Representations for Data-Efficient Reinforcement Learning


This repo provides the official implementation of the [paper](https://proceedings.mlr.press/v162/mondal22a.html). The code is based on original [spr](https://arxiv.org/abs/2007.05929) implementation.


## Installation 
To install the requirements, follow these steps:
```bash
# PyTorch
pip install pytorch torchvision

# Install requirements
pip install -r requirements.txt
```

## How to run?

* To run Eqr with group equivariant loss
```bash
python -m scripts.run --game krull --momentum-tau 1. --seed 0 --group-type 'En' --num-blocks 12 --projection-type 'q_shared' --second-projection-type 'mlp'  --acteqv-loss-weight 1. --groupeqv-loss-weight 1. --reward-loss-weight 1.
```

* To run Eqr with just action equivariant loss
```bash
python -m scripts.run --game krull --momentum-tau 1. --seed 0 --group-type 'En' --num-blocks 12 --projection-type 'q_shared' --second-projection-type 'mlp'  --acteqv-loss-weight 1. --groupeqv-loss-weight 0. --reward-loss-weight 1.
```

