from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.envs.atari.atari_env import AtariTrajInfo
from rlpyt.utils.logging.context import logger_context

import wandb
import torch
import numpy as np
import os
from src.model import EqrCatDqnModel

from src.rlpyt_utils import OneToOneSerialEvalCollector, SerialSampler, MinibatchRlEvalWandb
from src.algos import GroupCategoricalDQN
from src.agent import SPRAgent
from src.rlpyt_atari_env import AtariEnv
from src.utils import set_config
from scripts.arguments import get_arguments

def build_and_train(game="pong", run_ID=0, cuda_idx=0, args=None):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = AtariEnv
    config = set_config(args, game)

    sampler = SerialSampler(
        EnvCls=env,
        TrajInfoCls=AtariTrajInfo,  # default traj info + GameScore
        env_kwargs=config["env"],
        eval_env_kwargs=config["eval_env"],
        batch_T=config['sampler']['batch_T'],
        batch_B=config['sampler']['batch_B'],
        max_decorrelation_steps=0,
        eval_CollectorCls=OneToOneSerialEvalCollector,
        eval_n_envs=config["sampler"]["eval_n_envs"],
        eval_max_steps=config['sampler']['eval_max_steps'],
        eval_max_trajectories=config["sampler"]["eval_max_trajectories"],
    )
    args.discount = config["algo"]["discount"]
    algo = GroupCategoricalDQN(optim_kwargs=config["optim"], jumps=args.jumps, **config["algo"])  # Run with defaults.

    model = EqrCatDqnModel
    agent = SPRAgent(ModelCls=model, model_kwargs=config["model"], **config["agent"])

    wandb.config.update(config)
    runner = MinibatchRlEvalWandb(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=args.n_steps,
        affinity=dict(cuda_idx=cuda_idx),
        log_interval_steps=args.n_steps//args.num_logs,
        seed=args.seed,
        final_eval_only=args.final_eval_only,
        ckpt_path=args.checkpoint_dir + '/' + 'Eqr_' + args.game + '_' + str(args.seed),
        save_ckpt=args.save_ckpt
    )
    config = dict(game=game)
    name = "dqn_" + game
    log_dir = "logs"
    with logger_context(log_dir, run_ID, name, config, snapshot_mode="last"):
        runner.train()

    quit()


if __name__ == "__main__":
    args = get_arguments()
    args.project = args.project + '_' + args.game
    if args.use_wandb:
        os.environ["WANDB_MODE"] = "online"
    else:
        os.environ["WANDB_MODE"] = "disabled"
    if args.public:
        wandb.init(anonymous="allow", config=args, dir=args.wandb_dir)
    else:
        wandb.init(project=args.project, entity=args.entity, config=args, dir=args.wandb_dir)

    wandb.config.update(vars(args))
    build_and_train(game=args.game,
                    cuda_idx=args.cuda_idx,
                    args=args)
