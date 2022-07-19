from rlpyt.experiments.configs.atari.dqn.atari_dqn import configs
import os
import shutil
import torch
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def update_quiver(rep, Q, X, Y):
    """updates the horizontal and vertical vector components by a
    fixed increment on each frame
    """
    vector = torch.Tensor([0, 1, 1])

    Q.set_offsets([rep[0, 2], rep[1, 2]])
    rep[0, 2], rep[1, 2] = 0, 0
    new_vector = (rep @ vector).numpy()
    Q.set_UVC(new_vector[0], new_vector[1])
    return Q

def save_screen_animation(images, path, channel):
    images = images.squeeze()
    print(images.shape)
    fig = plt.figure(figsize=(10, 10))
    plt.axis("off")
    # ims = [
    #     [plt.imshow(np.transpose(images[j], (1, 2, 0)), animated=True)]
    #     for j in range(images.shape[0])
    # ]
    ims = [
        [plt.imshow(images[j], animated=True)]
        for j in range(images.shape[0])
    ]
    anim = animation.ArtistAnimation(
        fig, ims, interval=1000, repeat_delay=1000, blit=True
    )
    anim.save(path + "/" + f"Channel_{channel}_screen_animation.gif", dpi=128, writer="imagemagick")
    plt.show()

def save_latent_animation(reps, block_num, path):
    X, Y = 1, 1
    vector = [0, 1]
    print(reps.shape)

    fig, ax = plt.subplots(1, 1)
    Q = ax.quiver(X, Y, vector[0], vector[1], pivot='mid', color='r', units='inches')

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)

    # you need to set blit=False, or the first set of arrows never gets
    # cleared on subsequent frames
    anim = animation.FuncAnimation(fig, update_quiver, fargs=(Q, X, Y), frames=[m for m in reps],
                                   interval=50, blit=False)
    writergif = animation.PillowWriter(fps=4)
    anim.save(path + "/" + f"Block_{block_num}_animation.gif", writer=writergif, dpi=128)


class dummy_context_mgr:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False

def save_checkpoint(state, suffix, filename='checkpoint.pth.tar'):
    filename = suffix + '_' + filename
    torch.save(state, filename)
    print("checkpoint saved ..")

def load_checkpoint(suffix, filename='checkpoint.pth.tar'):
    filename = suffix + '_' + filename
    return torch.load(filename)

def copy_checkpoints(args, filename='checkpoint.pth.tar'):
    """For compute canada. Copies the models from the compute node to a directory in my home folder."""
    filename = args.suffix + '_' + filename
    shutil.copy(filename, os.path.join(args.save_dest, filename))
    for file_name in os.listdir('./'):
        if 'checkpoint' in file_name:
            shutil.copy(file_name, os.path.join(args.save_dest, file_name))


def set_config(args, game):
    # TODO: Use Hydra to manage configs
    config = configs['ernbw']
    config['env']['game'] = game
    config["env"]["grayscale"] = args.grayscale
    config["env"]["num_img_obs"] = args.framestack
    config["eval_env"]["game"] = config["env"]["game"]
    config["eval_env"]["grayscale"] = args.grayscale
    config["eval_env"]["num_img_obs"] = args.framestack
    config['env']['imagesize'] = args.imagesize
    config['eval_env']['imagesize'] = args.imagesize
    config['env']['seed'] = args.seed
    config['eval_env']['seed'] = args.seed
    config["model"]["dueling"] = bool(args.dueling)
    config["algo"]["min_steps_learn"] = args.min_steps_learn
    config["algo"]["n_step_return"] = args.n_step
    config["algo"]["batch_size"] = args.batch_size
    config["algo"]["learning_rate"] = 0.0001
    config['algo']['replay_ratio'] = args.replay_ratio
    config['algo']['target_update_interval'] = args.target_update_interval
    config['algo']['target_update_tau'] = args.target_update_tau
    config['algo']['eps_steps'] = args.eps_steps
    config["algo"]["clip_grad_norm"] = args.max_grad_norm
    config['algo']['pri_alpha'] = 0.5
    config['algo']['pri_beta_steps'] = int(10e4)
    config['optim']['eps'] = 0.00015
    config["sampler"]["eval_max_trajectories"] = 100
    config["sampler"]["eval_n_envs"] = 100
    config["sampler"]["eval_max_steps"] = 100 * 28000  # 28k is just a safe ceiling
    config['sampler']['batch_B'] = args.batch_b
    config['sampler']['batch_T'] = args.batch_t

    config['agent']['eps_init'] = args.eps_init
    config['agent']['eps_final'] = args.eps_final
    config["model"]["noisy_nets_std"] = args.noisy_nets_std

    if args.noisy_nets:
        config['agent']['eps_eval'] = 0.001

    # New SPR Arguments
    config["model"]["imagesize"] = args.imagesize
    config["model"]["jumps"] = args.jumps
    config["model"]["noisy_nets"] = args.noisy_nets
    config["model"]["momentum_encoder"] = args.momentum_encoder
    config["model"]["shared_encoder"] = args.shared_encoder
    config["model"]["distributional"] = args.distributional
    config["model"]["augmentation"] = args.augmentation
    config["model"]["q_shared_type"] = args.q_shared_type
    config["model"]["dropout"] = args.dropout
    config["model"]["time_offset"] = args.time_offset
    config["model"]["aug_prob"] = args.aug_prob
    config["model"]["target_augmentation"] = args.target_augmentation
    config["model"]["eval_augmentation"] = args.eval_augmentation
    config["model"]["projection_type"] = args.projection_type
    config["model"]["second_projection_type"] = args.second_projection_type
    config['model']['momentum_tau'] = args.momentum_tau
    config["model"]["dqn_hidden_size"] = args.dqn_hidden_size
    config["model"]["group_dim"] = args.group_dim
    config["model"]["group_type"] = args.group_type
    config["model"]["num_blocks"] = args.num_blocks
    config["model"]["parameterization_type"] = args.parameterization_type
    config["model"]["only_action_transition"] = args.only_action_transition
    config["algo"]["reward_loss_weight"] = args.reward_loss_weight
    config["algo"]["acteqv_loss_weight"] = args.acteqv_loss_weight
    config["algo"]["groupeqv_loss_weight"] = args.groupeqv_loss_weight
    config["algo"]["time_offset"] = args.time_offset
    config["algo"]["distributional"] = args.distributional
    config["algo"]["delta_clip"] = args.delta_clip
    config["algo"]["prioritized_replay"] = args.prioritized_replay

    return config