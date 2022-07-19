import argparse

def get_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--game', help='Atari game', default='ms_pacman')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--grayscale', type=int, default=1)
    parser.add_argument('--framestack', type=int, default=4)
    parser.add_argument('--imagesize', type=int, default=84)
    parser.add_argument('--n-steps', type=int, default=100000)
    parser.add_argument('--dqn-hidden-size', type=int, default=256)
    parser.add_argument('--target-update-interval', type=int, default=1)
    parser.add_argument('--target-update-tau', type=float, default=1.)
    parser.add_argument('--momentum-tau', type=float, default=1.)
    parser.add_argument('--batch-b', type=int, default=1)
    parser.add_argument('--batch-t', type=int, default=1)
    parser.add_argument('--jumps', type=int, default=5)
    parser.add_argument('--num-logs', type=int, default=10)
    parser.add_argument('--dueling', type=int, default=1)
    parser.add_argument('--replay-ratio', type=int, default=64)
    parser.add_argument('--n-step', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--wandb-dir', type=str, default='', help='Directory for wandb files.')
    parser.add_argument('--norm-type', type=str, default='bn', choices=["bn", "ln", "in", "none"], help='Normalization')
    parser.add_argument('--aug-prob', type=float, default=1., help='Probability to apply augmentation')
    parser.add_argument('--dropout', type=float, default=0., help='Dropout probability in the convnet encoder.')
    parser.add_argument('--distributional', type=int, default=1)
    parser.add_argument('--delta-clip', type=float, default=1., help="Huber Delta")
    parser.add_argument('--prioritized-replay', type=int, default=1)
    parser.add_argument('--momentum-encoder', type=int, default=1)
    parser.add_argument('--shared-encoder', type=int, default=0)
    parser.add_argument('--noisy-nets', type=int, default=1)
    parser.add_argument('--noisy-nets-std', type=float, default=0.5)
    parser.add_argument('--projection-type', type=str, default='q_shared', choices=["mlp", "q_shared", "linear", "none"], help='Style of NCE classifier')
    parser.add_argument('--second-projection-type', type=str, default='mlp', choices=["mlp", "linear", "none"], help='Type of projection head for ')
    parser.add_argument('--augmentation', type=str, default=["shift", "intensity"], nargs="+",
                        choices=["none", "crop", "blur", "shift", "intensity"],
                        help='Style of augmentation')
    parser.add_argument('--q-shared-type', type=str, default=["value", "advantage"], nargs="+",
                        choices=["noisy", "value", "advantage", "relu"],
                        help='Style of q_shared projection')
    parser.add_argument('--target-augmentation', type=int, default=1, help='Use augmentation on inputs to target networks')
    parser.add_argument('--eval-augmentation', type=int, default=0, help='Use augmentation on inputs at evaluation time')
    parser.add_argument('--reward-loss-weight', type=float, default=0.)
    parser.add_argument('--acteqv-loss-weight', type=float, default=5.)
    parser.add_argument('--groupeqv-loss-weight', type=float, default=0.)
    parser.add_argument('--eps-steps', type=int, default=2001)
    parser.add_argument('--min-steps-learn', type=int, default=2000)
    parser.add_argument('--eps-init', type=float, default=1.)
    parser.add_argument('--eps-final', type=float, default=0.)
    parser.add_argument('--final-eval-only', type=int, default=0)
    parser.add_argument('--time-offset', type=int, default=0)
    parser.add_argument('--use-wandb', type=int, default=0)
    parser.add_argument('--project', type=str, default="Eqr")
    parser.add_argument('--entity', type=str, default="")
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=0)
    parser.add_argument('--max-grad-norm', type=float, default=10., help='Max Grad Norm')
    parser.add_argument('--public', action='store_true', help='If set, uses anonymous wandb logging')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--visualization-dir', type=str, default='./visualization',
                        help='Directory with the visualizations')
    parser.add_argument('--save-ckpt', type=int, default=0,
                        help='Whether to save checkpoints during training')

    # Group arguments
    parser.add_argument('--group-type', type=str, default='En',choices=["SOn", "En", "GLn", "Real", "none"],
                        help='Group type options 1)SOn 2)En 3)GLn 4)Real 5)none')
    parser.add_argument('--group-dim', type=int, default=2)
    parser.add_argument('--num-blocks', type=int, default=16)
    parser.add_argument('--parameterization-type', type=str, default='Euler',
                        help='Parametrization type options 1)Lie 2)Euler')
    parser.add_argument('--only-action-transition', type=int, default=0)

    args = parser.parse_args()
    return args