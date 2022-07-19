import torch
import torch.nn.functional as F
import torch.nn as nn

from rlpyt.models.utils import scale_grad, update_state_dict
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from src.utils import count_parameters, dummy_context_mgr
from src.algos import from_categorical
import numpy as np
from kornia.augmentation import RandomCrop,CenterCrop
from kornia.filters import GaussianBlur2d
import copy
from src.group import LieParameterization, EulerParameterization


class EqrCatDqnModel(torch.nn.Module):
    def __init__(
            self,
            image_shape,
            output_size,
            n_atoms,
            dueling,
            jumps,
            augmentation,
            target_augmentation,
            eval_augmentation,
            noisy_nets,
            aug_prob,
            projection_type,
            second_projection_type,
            imagesize,
            time_offset,
            momentum_encoder,
            shared_encoder,
            distributional,
            dqn_hidden_size,
            momentum_tau,
            q_shared_type,
            dropout,
            noisy_nets_std,
            group_type,
            group_dim,
            num_blocks,
            parameterization_type,
            only_action_transition
    ):
        """Instantiates the neural network according to arguments; network defaults
        stored within this method."""
        super().__init__()

        self.noisy = noisy_nets
        self.time_offset = time_offset
        self.aug_prob = aug_prob
        self.projection_type = projection_type
        self.second_projection_type = second_projection_type

        self.distributional = distributional
        n_atoms = 1 if not self.distributional else n_atoms
        self.dqn_hidden_size = dqn_hidden_size
        self.only_action_transition = only_action_transition

        self.transforms = []
        self.eval_transforms = []

        self.uses_augmentation = False
        for aug in augmentation:
            if aug == "crop":
                transformation = RandomCrop((84, 84))
                # Crashes if aug-prob not 1: use CenterCrop((84, 84)) or Resize((84, 84)) in that case.
                eval_transformation = CenterCrop((84, 84))
                self.uses_augmentation = True
                imagesize = 84
            elif aug == "blur":
                transformation = GaussianBlur2d((5, 5), (1.5, 1.5))
                eval_transformation = nn.Identity()
                self.uses_augmentation = True
            elif aug == "shift":
                transformation = nn.Sequential(nn.ReplicationPad2d(4), RandomCrop((84, 84)))
                eval_transformation = nn.Identity()
            elif aug == "intensity":
                transformation = Intensity(scale=0.05)
                eval_transformation = nn.Identity()
            elif aug == "none":
                transformation = eval_transformation = nn.Identity()
            else:
                raise NotImplementedError()
            self.transforms.append(transformation)
            self.eval_transforms.append(eval_transformation)

        if parameterization_type == 'Lie':
            group_parameterization = LieParameterization(group_type, group_dim, num_blocks)
        elif parameterization_type == 'Euler':
            group_parameterization = EulerParameterization(group_type, group_dim, num_blocks)
        else:
            raise Exception('Parameterization Type not implemented ...')

        self.rep_dim = group_parameterization.rep_dim
        self.group_param_dim = group_parameterization.group_param_dim
        self.group_parameterization = group_parameterization
        self.state_space_dim_param = self.group_param_dim * num_blocks
        self.group_type = group_type
        if group_type in ('Real', 'none'):
            self.state_space_dim_rep = self.state_space_dim_param
        else:
            self.state_space_dim_rep = self.rep_dim * self.rep_dim * num_blocks
        self.dueling = dueling
        f, c = image_shape[:2]
        in_channels = np.prod(image_shape[:2])
        self.encoder = Conv2dModel(
            in_channels=in_channels,
            channels=[32, 64, 64],
            kernel_sizes=[8, 4, 3],
            strides=[4, 2, 1],
            paddings=[0, 0, 0],
            use_maxpool=False,
            dropout=dropout,
        )

        fake_input = torch.zeros(1, f * c, imagesize, imagesize)
        fake_output = self.encoder(fake_input)
        self.hidden_size = fake_output.shape[1]
        self.pixels = fake_output.shape[-1] * fake_output.shape[-2]
        print("Spatial latent size is {}".format(fake_output.shape[1:]))

        self.jumps = jumps
        self.target_augmentation = target_augmentation
        self.eval_augmentation = eval_augmentation
        self.num_actions = output_size         # output_size is number of actions and passed as a kwargs from AtariMixin

        if dueling:
            self.rl_head = DQNDistributionalDuelingHeadModel(self.state_space_dim_rep,
                                                          output_size,
                                                          hidden_size=self.dqn_hidden_size,
                                                          pixels=1,
                                                          noisy=self.noisy,
                                                          n_atoms=n_atoms,
                                                          std_init=noisy_nets_std)
        else:
            self.rl_head = DQNDistributionalHeadModel(self.state_space_dim_rep,
                                                   output_size,
                                                   hidden_size=self.dqn_hidden_size,
                                                   pixels=1,
                                                   noisy=self.noisy,
                                                   n_atoms=n_atoms,
                                                   std_init=noisy_nets_std)

        if self.jumps > 0:
            self.dynamics_model = EquivariantTransitionModel(
                                                  state_space_dim_rep=self.state_space_dim_rep,
                                                  state_space_dim_param=self.state_space_dim_param,
                                                  num_actions=output_size,
                                                  group_parameterization= group_parameterization,
                                                  only_action_transition= only_action_transition,
                                                  pixels=self.pixels,
                                                  limit=1,
                                                  group_type=group_type)
        else:
            self.dynamics_model = nn.Identity()
        self.num_blocks = num_blocks

        self.momentum_encoder = momentum_encoder
        self.momentum_tau = momentum_tau
        self.shared_encoder = shared_encoder
        assert not (self.shared_encoder and self.momentum_encoder)

        self.group_head = nn.Sequential(
                                    nn.Flatten(-3, -1),
                                    nn.Linear(self.pixels * self.hidden_size, self.state_space_dim_param),
                                    )
        self.target_group_head = self.group_head

        if group_type in ('Real', 'none'):
            start_dim = -2
        else:
            start_dim = -3
        if self.projection_type == "mlp":
            self.projection_head = nn.Sequential(
                nn.Flatten(start_dim, -1),
                nn.Linear(self.state_space_dim_rep, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 64)
            )
        elif self.projection_type == "q_shared":
            self.projection_head = QsharedHead(self.rl_head, dueling=dueling, type=q_shared_type)
        elif self.projection_type == "linear":
            self.projection_head = nn.Sequential(
                nn.Flatten(start_dim, -1),
                nn.Linear(self.state_space_dim_rep, 128)
            )
        elif self.projection_type == "none":
            self.projection_head = nn.Flatten(start_dim, -1)

        self.action_transformation_detector = nn.Sequential(
            nn.Flatten(start_dim, -1),
            nn.Linear(self.state_space_dim_rep, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.state_space_dim_param)
        )

        if self.second_projection_type == "mlp":
            self.second_projection_head = nn.Sequential(
                nn.Flatten(start_dim, -1),
                nn.Linear(self.state_space_dim_rep, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 64)
            )
        elif self.second_projection_type == "linear":
            self.second_projection_head = nn.Sequential(
                nn.Flatten(start_dim, -1),
                nn.Linear(self.state_space_dim_rep, 128)
            )
        elif self.second_projection_type == "none":
            self.second_projection_head = nn.Flatten(start_dim, -1)

        if self.momentum_encoder:
            self.target_encoder = copy.deepcopy(self.encoder)
            self.target_group_head = copy.deepcopy(self.target_group_head)
            self.target_projection_head = copy.deepcopy(self.projection_head)
            self.target_action_encoder = copy.deepcopy(self.dynamics_model.action_encoder)
            self.target_second_projection_head = copy.deepcopy(self.second_projection_head)

            for param in (list(self.target_encoder.parameters())
                        + list(self.target_group_head.parameters())
                        + list(self.target_projection_head.parameters())
                        + list(self.target_action_encoder.parameters())
                        + list(self.target_second_projection_head.parameters())):
                param.requires_grad = False

        elif self.shared_encoder:
            self.target_encoder = self.encoder
            self.target_projection_head = self.projection_head
            self.target_second_projection_head = self.second_projection_head

        self.mse_loss = nn.MSELoss(reduction='none')

        print("Initialized model with {} parameters".format(count_parameters(self)))


    def set_sampling(self, sampling):
        if self.noisy:
            self.head.set_sampling(sampling)

    def normalised_mse_loss(self, pred, target):
        pred = F.normalize(pred.float(), p=2., dim=-1, eps=1e-3)
        target = F.normalize(target.float(), p=2., dim=-1, eps=1e-3)
        loss = self.mse_loss(pred, target).sum(-1)
        return loss

    def action_eqv_loss(self, pred_group_reps, target_group_reps):
        pred_group_reps_projected = self.projection_head(pred_group_reps)
        target_group_reps_projected = self.target_projection_head(target_group_reps)
        loss = self.normalised_mse_loss(
            pred_group_reps_projected,
            target_group_reps_projected
        )
        return loss

    def group_eqv_loss(self, pred_group_reps, target_group_reps):
        pred_group_reps_projected = self.second_projection_head(pred_group_reps)
        target_group_reps_projected = self.second_projection_head(target_group_reps)
        loss = self.normalised_mse_loss(
            pred_group_reps_projected,
            target_group_reps_projected
        )
        return loss

    def apply_transforms(self, transforms, eval_transforms, image):
        if eval_transforms is None:
            for transform in transforms:
                image = transform(image)
        else:
            for transform, eval_transform in zip(transforms, eval_transforms):
                image = maybe_transform(
                    image, transform, eval_transform, p=self.aug_prob
                )
        return image

    @torch.no_grad()
    def transform(self, images, augment=False):
        images = images.float()/255. if images.dtype == torch.uint8 else images
        flat_images = images.reshape(-1, *images.shape[-3:])
        if augment:
            processed_images = self.apply_transforms(self.transforms,
                                                     self.eval_transforms,
                                                     flat_images)
        else:
            processed_images = self.apply_transforms(self.eval_transforms,
                                                     None,
                                                     flat_images)
        processed_images = processed_images.view(*images.shape[:-3],
                                                 *processed_images.shape[1:])
        return processed_images

    def stem_parameters(self):
        return list(self.encoder.parameters()) + list(self.rl_head.parameters())

    def encoder_forward(self, img):
        """Returns the normalized output of convolutional layers."""
        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

        conv_out = self.encoder(img.view(T * B, *img_shape))  # Fold if T dimension.
        return conv_out

    def rl_head_forward(self, conv_out, logits=False):
        lead_dim, T, B, img_shape = infer_leading_dims(conv_out, 3)
        p = self.rl_head(conv_out)

        if self.distributional:
            if logits:
                p = F.log_softmax(p, dim=-1)
            else:
                p = F.softmax(p, dim=-1)
        else:
            p = p.squeeze(-1)

        p = restore_leading_dims(p, lead_dim, T, B)
        return p

    def forward(self, observation, prev_action, prev_reward,
                train=False, eval=False):
        """
        For convenience reasons with DistributedDataParallel the forward method
        has been split into two cases, one for training and one for eval.

        observation: shape [seq_length, batch_size, 4, num_channels, image_shape, image_shape]
        prev_action: shape [seq_length, batch_size]
        prev_reward:
        """
        if train:
            log_pred_ps = []
            pred_group_reps = []
            pred_reward = []
            input_obs = observation[0].flatten(1, 2)
            # shape [batch_size, 4, image_shape, image_shape]
            input_obs = self.transform(input_obs, augment=True)
            latent = self.encoder_forward(input_obs)
            # shape [batch_size, 64, 7, 7]
            state_rep_group_param = self.group_head(latent)
            # shape [batch_size, num_blocks * group_param_dim]
            state_rep_group_param = state_rep_group_param.reshape(-1, self.num_blocks, self.group_param_dim)
            # shape [batch_size, num_blocks, group_param_dim]
            state_rep_group_rep = self.group_parameterization.get_group_rep(state_rep_group_param)
            # shape [batch_size, num_blocks, rep_dim, rep_dim]
            batch_size = latent.shape[0]

            log_pred_ps.append(self.rl_head_forward(
                state_rep_group_rep.reshape(batch_size, -1).unsqueeze(-1).unsqueeze(-1)
                , logits=True
            ))

            #For the group equivariance loss
            action = prev_action[1]
            state_rep_flattened = state_rep_group_rep.reshape(batch_size, -1).detach()
            batch_range = torch.arange(batch_size, device=action.device)
            action_onehot = torch.zeros(batch_size, self.num_actions, device=action.device)
            action_onehot[batch_range, action] = 1
            action_encoder_input = torch.cat([action_onehot, state_rep_flattened], 1) if not self.only_action_transition else action_onehot
            action_encoder_output_params = self.target_action_encoder(action_encoder_input)
            # [batch_size, num_blocks * group_param_dim]
            action_encoder_output_params = action_encoder_output_params.reshape(batch_size, self.num_blocks, self.group_param_dim)
            # [batch_size, num_blocks, group_param_dim]
            action_encoder_output_reps = self.group_parameterization.get_group_rep(action_encoder_output_params)
            # [batch_size, num_blocks, rep_dim, rep_dim]


            #Dynamics model part
            if self.jumps > 0:
                pred_group_reps.append(state_rep_group_rep)
                pred_rew = self.dynamics_model.reward_predictor(state_rep_group_rep.reshape(batch_size, -1))
                pred_reward.append(F.log_softmax(pred_rew, -1))

                for j in range(1, self.jumps + 1):
                    state_rep_group_rep, pred_rew = self.step(state_rep_group_rep, prev_action[j])
                    pred_group_reps.append(state_rep_group_rep)
                    pred_rew = pred_rew[:observation.shape[1]]
                    pred_reward.append(F.log_softmax(pred_rew, -1))

            pred_group_reps = torch.stack(pred_group_reps, 1)
            # shape [batch_size, num_jumps, num_blocks, rep_dim, rep_dim]

            # For the symmetric MDP part
            pred_group_reps_1 = pred_group_reps[:, 1]
            # shape [batch_size, num_blocks, rep_dim, rep_dim]

            pred_group_reps = pred_group_reps[:observation.shape[1]].flatten(0, 1)  # batch*jumps, *

            target_images = observation[self.time_offset:self.jumps + self.time_offset + 1].transpose(0, 1).flatten(2, 3)
            target_images = self.transform(target_images, True)

            with torch.no_grad() if self.momentum_encoder else dummy_context_mgr():
                target_latents = self.target_encoder(target_images.flatten(0, 1))
                # shape [batch_size * num_jumps, 64, 7, 7]
                target_group_params = self.group_head(target_latents).reshape(
                    -1, self.num_blocks, self.group_param_dim
                )
                # shape [batch_size * num_jumps, num_blocks, group_param_dim]
                target_group_reps = self.group_parameterization.get_group_rep(target_group_params)
                # shape [batch_size * num_jumps, num_blocks, rep_dim, rep_dim]

            # Action equivariant loss part
            action_eqv_loss = self.action_eqv_loss(pred_group_reps, target_group_reps)
            action_eqv_loss = action_eqv_loss.view(-1, observation.shape[1])
            # action_eqv_loss = torch.zeros((self.jumps + 1, observation.shape[1]), device=latent.device)

            # Group equivariance loss part
            if self.group_type != 'Real':
                target_group_reps = target_group_reps.reshape(-1, self.jumps + 1, self.num_blocks, self.rep_dim, self.rep_dim)
            else:
                target_group_reps = target_group_reps.reshape(-1, self.jumps + 1, self.num_blocks, self.rep_dim)

            target_group_reps_0 = target_group_reps[:, 0]
            # shape [batch_size, num_blocks, rep_dim, rep_dim]
            target_group_reps_0_prime = target_group_reps_0[torch.randperm(batch_size)]
            # shape [batch_size, num_blocks, rep_dim, rep_dim]
            group_rep_acting_on_states = self.group_parameterization.compose(
                target_group_reps_0_prime, self.group_parameterization.inverse(target_group_reps_0)
            ).detach()
            # shape [batch_size, num_blocks, rep_dim, rep_dim]
            group_params_acting_on_actions = self.action_transformation_detector(group_rep_acting_on_states).reshape(
                    -1, self.num_blocks, self.group_param_dim
                )
            # shape [batch_size, num_blocks, group_param_dim]
            group_rep_acting_on_actions = self.group_parameterization.get_group_rep(group_params_acting_on_actions)
            # shape [batch_size, num_blocks, rep_dim, rep_dim]

            target_group_reps_1_prime = self.group_parameterization.compose(
                self.group_parameterization.compose(group_rep_acting_on_actions,
                      action_encoder_output_reps), target_group_reps_0_prime
            )

            group_reps_1_prime = self.group_parameterization.compose(
                group_rep_acting_on_states, pred_group_reps_1
            )
            group_eqv_loss = self.group_eqv_loss(
                group_reps_1_prime, target_group_reps_1_prime
            )

            # Reward loss part
            if len(pred_reward) > 0:
                pred_reward = torch.stack(pred_reward, 0)
                with torch.no_grad():
                    reward_target = to_categorical(
                        prev_reward[:self.jumps + 1].flatten().to(action.device),
                        limit=1
                    ).view(*pred_reward.shape)
                reward_loss = -torch.sum(reward_target * pred_reward, 2).mean(0).cpu()
            else:
                reward_loss = torch.zeros(observation.shape[1], )

            if self.momentum_encoder:
                update_state_dict(self.target_encoder, self.encoder.state_dict(), self.momentum_tau)
                update_state_dict(self.target_group_head, self.group_head.state_dict(), self.momentum_tau)
                update_state_dict(self.target_projection_head, self.projection_head.state_dict(), self.momentum_tau)
                update_state_dict(self.target_action_encoder, self.dynamics_model.action_encoder.state_dict(), self.momentum_tau)
                update_state_dict(self.target_second_projection_head, self.second_projection_head.state_dict(), self.momentum_tau)


            return log_pred_ps, reward_loss, action_eqv_loss, group_eqv_loss

        else:
            aug_factor = self.target_augmentation if not eval else self.eval_augmentation
            observation = observation.flatten(-4, -3)
            stacked_observation = observation.unsqueeze(1).repeat(1, max(1, aug_factor), 1, 1, 1)
            stacked_observation = stacked_observation.view(-1, *observation.shape[1:])

            img = self.transform(stacked_observation, aug_factor)

            # Infer (presence of) leading dimensions: [T,B], [B], or [].
            lead_dim, T, B, img_shape = infer_leading_dims(img, 3)
            conv_out = self.encoder(img.view(T * B, *img_shape))  # Fold if T dimension.

            state_rep_group_param = self.group_head(conv_out)
            # shape [batch_size, num_blocks * group_param_dim]
            state_rep_group_param = state_rep_group_param.reshape(-1, self.num_blocks, self.group_param_dim)
            # shape [batch_size, num_blocks, group_param_dim]
            state_rep_group_rep = self.group_parameterization.get_group_rep(state_rep_group_param)
            # shape [batch_size, num_blocks, rep_dim, rep_dim]
            batch_size = conv_out.shape[0]

            p = self.rl_head(state_rep_group_rep.reshape(batch_size, -1).unsqueeze(-1).unsqueeze(-1))

            if self.distributional:
                p = F.softmax(p, dim=-1)
            else:
                p = p.squeeze(-1)

            p = p.view(observation.shape[0], max(1, aug_factor), *p.shape[1:]).mean(1)

            # Restore leading dimensions: [T,B], [B], or [], as input.
            p = restore_leading_dims(p, lead_dim, T, B)

            return p

    def select_action(self, obs):
        value = self.forward(obs, None, None, train=False, eval=True)
        if self.distributional:
            value = from_categorical(value, logits=False, limit=10)
        return value

    def step(self, state, action):
        next_state, reward_logits = self.dynamics_model(state, action)
        return next_state, reward_logits


class QsharedHead(nn.Module):
    def __init__(self, head, dueling=False, type="noisy advantage"):
        super().__init__()
        self.head = head
        self.noisy = "noisy" in type
        self.dueling = dueling
        self.encoders = nn.ModuleList()
        self.relu = "relu" in type
        value = "value" in type
        advantage = "advantage" in type
        if self.dueling:
            if value:
                self.encoders.append(self.head.value[1])
            if advantage:
                self.encoders.append(self.head.advantage_hidden[1])
        else:
            self.encoders.append(self.head.network[1])

        self.out_features = sum([e.out_features for e in self.encoders])

    def forward(self, x):
        x = x.flatten(-3, -1)
        representations = []
        for encoder in self.encoders:
            encoder.noise_override = self.noisy
            representations.append(encoder(x))
            encoder.noise_override = None
        representation = torch.cat(representations, -1)
        if self.relu:
            representation = F.relu(representation)

        return representation


class DQNDistributionalHeadModel(torch.nn.Module):
    def __init__(self,
                 input_channels,
                 output_size,
                 hidden_size=256,
                 pixels=30,
                 n_atoms=51,
                 noisy=0,
                 std_init=0.1):
        super().__init__()
        if noisy:
            linear = NoisyLinear
            self.linears = [linear(input_channels*pixels, hidden_size, std_init=std_init),
                            linear(hidden_size, output_size * n_atoms, std_init=std_init)]
        else:
            linear = nn.Linear
            self.linears = [linear(input_channels*pixels, hidden_size),
                            linear(hidden_size, output_size * n_atoms)]
        layers = [nn.Flatten(-3, -1),
                  self.linears[0],
                  nn.ReLU(),
                  self.linears[1]]
        self.network = nn.Sequential(*layers)
        if not noisy:
            self.network.apply(weights_init)
        self._output_size = output_size
        self._n_atoms = n_atoms

    def forward(self, input):
        return self.network(input).view(-1, self._output_size, self._n_atoms)

    def reset_noise(self):
        for module in self.linears:
            module.reset_noise()

    def set_sampling(self, sampling):
        for module in self.linears:
            module.sampling = sampling


class DQNDistributionalDuelingHeadModel(torch.nn.Module):
    """An MLP head with optional noisy layers which reshapes output to [B, output_size, n_atoms]."""

    def __init__(self,
                 input_channels,
                 output_size,
                 pixels=30,
                 n_atoms=51,
                 hidden_size=256,
                 grad_scale=2 ** (-1 / 2),
                 noisy=0,
                 std_init=0.1):
        super().__init__()
        if noisy:
            self.linears = [NoisyLinear(pixels * input_channels, hidden_size, std_init=std_init),
                            NoisyLinear(hidden_size, output_size * n_atoms, std_init=std_init),
                            NoisyLinear(pixels * input_channels, hidden_size, std_init=std_init),
                            NoisyLinear(hidden_size, n_atoms, std_init=std_init)
                            ]
        else:
            self.linears = [nn.Linear(pixels * input_channels, hidden_size),
                            nn.Linear(hidden_size, output_size * n_atoms),
                            nn.Linear(pixels * input_channels, hidden_size),
                            nn.Linear(hidden_size, n_atoms)
                            ]
        self.advantage_layers = [nn.Flatten(-3, -1),
                                 self.linears[0],
                                 nn.ReLU(),
                                 self.linears[1]]
        self.value_layers = [nn.Flatten(-3, -1),
                             self.linears[2],
                             nn.ReLU(),
                             self.linears[3]]
        self.advantage_hidden = nn.Sequential(*self.advantage_layers[:3])
        self.advantage_out = self.advantage_layers[3]
        self.advantage_bias = torch.nn.Parameter(torch.zeros(n_atoms), requires_grad=True)
        self.value = nn.Sequential(*self.value_layers)
        self.network = self.advantage_hidden
        self._grad_scale = grad_scale
        self._output_size = output_size
        self._n_atoms = n_atoms

    def forward(self, input):
        x = scale_grad(input, self._grad_scale)
        advantage = self.advantage(x)
        value = self.value(x).view(-1, 1, self._n_atoms)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

    def advantage(self, input):
        x = self.advantage_hidden(input)
        x = self.advantage_out(x)
        x = x.view(-1, self._output_size, self._n_atoms)
        return x + self.advantage_bias

    def reset_noise(self):
        for module in self.linears:
            module.reset_noise()

    def set_sampling(self, sampling):
        for module in self.linears:
            module.sampling = sampling


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='linear')
        torch.nn.init.zeros_(m.bias)

def to_categorical(value, limit=300):
    value = value.float()  # Avoid any fp16 shenanigans
    value = value.clamp(-limit, limit)
    distribution = torch.zeros(value.shape[0], (limit*2+1), device=value.device)
    lower = value.floor().long() + limit
    upper = value.ceil().long() + limit
    upper_weight = value % 1
    lower_weight = 1 - upper_weight
    distribution.scatter_add_(-1, lower.unsqueeze(-1), lower_weight.unsqueeze(-1))
    distribution.scatter_add_(-1, upper.unsqueeze(-1), upper_weight.unsqueeze(-1))
    return distribution


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.1, bias=True):
        super(NoisyLinear, self).__init__()
        self.bias = bias
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.sampling = True
        self.noise_override = None
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features), requires_grad=bias)
        self.bias_sigma = nn.Parameter(torch.empty(out_features), requires_grad=bias)
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        if not self.bias:
            self.bias_mu.fill_(0)
            self.bias_sigma.fill_(0)
        else:
            self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
            self.bias_mu.data.uniform_(-mu_range, mu_range)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        if self.noise_override is None:
            use_noise = self.training or self.sampling
        else:
            use_noise = self.noise_override
        if use_noise:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)


def maybe_transform(image, transform, alt_transform, p=0.8):
    processed_images = transform(image)
    if p >= 1:
        return processed_images
    else:
        base_images = alt_transform(image)
        mask = torch.rand((processed_images.shape[0], 1, 1, 1),
                          device=processed_images.device)
        mask = (mask < p).float()
        processed_images = mask * processed_images + (1 - mask) * base_images
        return processed_images


class Intensity(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        r = torch.randn((x.size(0), 1, 1, 1), device=x.device)
        noise = 1.0 + (self.scale * r.clamp(-2.0, 2.0))
        return x * noise


class Conv2dModel(torch.nn.Module):
    """2-D Convolutional model component, with option for max-pooling vs
    downsampling for strides > 1.  Requires number of input channels, but
    not input shape.  Uses ``torch.nn.Conv2d``.
    """

    def __init__(
            self,
            in_channels,
            channels,
            kernel_sizes,
            strides,
            paddings=None,
            nonlinearity=torch.nn.ReLU,  # Module, not Functional.
            use_maxpool=False,  # if True: convs use stride 1, maxpool downsample.
            head_sizes=None,  # Put an MLP head on top.
            dropout=0.,
            ):
        super().__init__()
        if paddings is None:
            paddings = [0 for _ in range(len(channels))]
        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings)
        in_channels = [in_channels] + channels[:-1]
        ones = [1 for _ in range(len(strides))]
        if use_maxpool:
            maxp_strides = strides
            strides = ones
        else:
            maxp_strides = ones
        conv_layers = [torch.nn.Conv2d(in_channels=ic, out_channels=oc,
            kernel_size=k, stride=s, padding=p) for (ic, oc, k, s, p) in
            zip(in_channels, channels, kernel_sizes, strides, paddings)]
        sequence = list()
        for i, (conv_layer, maxp_stride) in enumerate(zip(conv_layers, maxp_strides)):
            sequence.append(conv_layer)
            if i < len(channels)-1:
                sequence.append(nonlinearity())
            if dropout > 0:
                sequence.append(nn.Dropout(dropout))
            if maxp_stride > 1:
                sequence.append(torch.nn.MaxPool2d(maxp_stride))  # No padding.
        self.conv = torch.nn.Sequential(*sequence)

    def forward(self, input):
        """Computes the convolution stack on the input; assumes correct shape
        already: [B,C,H,W]."""
        return self.conv(input)


def init_normalization(channels, type="bn", affine=True, one_d=False):
    assert type in ["bn", "ln", "in", "none", None]
    if type == "bn":
        if one_d:
            return nn.BatchNorm1d(channels, affine=affine)
        else:
            return nn.BatchNorm2d(channels, affine=affine)
    elif type == "ln":
        if one_d:
            return nn.LayerNorm(channels, elementwise_affine=affine)
        else:
            return nn.GroupNorm(1, channels, affine=affine)
    elif type == "in":
        return nn.GroupNorm(channels, channels, affine=affine)
    elif type == "none" or type is None:
        return nn.Identity()


class EquivariantTransitionModel(nn.Module):
    def __init__(self,
                 state_space_dim_rep,
                 state_space_dim_param,
                 num_actions,
                 group_parameterization,
                 only_action_transition,
                 args=None,
                 hidden_size=256,
                 pixels=36,
                 limit=300,
                 action_dim=6,
                 group_type='En'):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.args = args
        self.num_blocks = group_parameterization.num_blocks
        self.group_param_dim = group_parameterization.group_param_dim
        self.rep_dim = group_parameterization.rep_dim

        self.action_embedding = nn.Embedding(num_actions, pixels*action_dim)

        in_dim = state_space_dim_rep + num_actions if not only_action_transition else num_actions
        self.action_encoder = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, state_space_dim_param)
        )
        self.reward_predictor = nn.Sequential(
            nn.Linear(state_space_dim_rep, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, limit*2 + 1)
        )
        self.group_type = group_type
        self.group_parameterization = group_parameterization
        self.only_action_transition = only_action_transition
        self.train()

    def forward(self, state, action):
        batch_size = action.shape[0]
        state_flattened = state.reshape(batch_size, -1)
        batch_range = torch.arange(batch_size, device=action.device)
        action_onehot = torch.zeros(batch_size, self.num_actions, device=action.device)
        action_onehot[batch_range, action] = 1
        action_encoder_input = torch.cat([action_onehot, state_flattened], 1) if not self.only_action_transition else action_onehot

        action_params = self.action_encoder(action_encoder_input)
        # [batch_size, num_blocks * group_param_dim]
        action_params = action_params.reshape(batch_size, self.num_blocks, self.group_param_dim)
        # [batch_size, num_blocks, group_param_dim]
        action_reps = self.group_parameterization.get_group_rep(action_params)
        # [batch_size, num_blocks, rep_dim, rep_dim]
        if self.group_type == 'none':
            next_state = action_reps
        else:
            next_state = self.group_parameterization.compose(action_reps, state)
        next_reward = self.reward_predictor(next_state.reshape(batch_size, -1))
        return next_state, next_reward




