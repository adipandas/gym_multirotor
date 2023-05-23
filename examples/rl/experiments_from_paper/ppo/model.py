import copy
import numpy as np
import torch
import torch.nn as nn

from ppo.distributions import Bernoulli, Categorical, DiagGaussian
from ppo.utils import init


class Flatten(nn.Module):
    """
    Custom layer to flatten the input data. Input data with shape (k, m, n, ...) is formated to shape (k, m*n*...)
    """

    def forward(self, x):
        """
        Args:
            x (torch.tensor): Tensor of shape (k, m, n, ...). For example, input tensor can be of shape (10, 20, 20)

        Returns:
            torch.tensor: Output tensor is of shape (k, m*n*...). For example, if input tensor is of shape (10, 20, 20), output tensor will be of shape (10, 20*20) = (10, 400).
        """
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    """
    Control policy to train using Reinforcement Learning.

    Args:
        obs_shape (tuple): Observation dimensions from Gym environment as tuple of integers.
        action_space (gym.Env.action_space): Action space object from Gym environment.
        base (nn.Module): Network base architecture object. Default is ``None``.
        base_kwargs: Network architecture details.

    """
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.base = base(obs_shape[0], **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        """
        Returns:
            bool: `True` if the policy is a recurrent neural network.
        """
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx"""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        """
        Forward pass.

        Args:
            inputs (torch.tensor):
            rnn_hxs (Union[torch.tensor, Tuple[torch.tensor, torch.tensor]]): Hidden state of the recurrent network. [GRU or Simple RNN - tensor, LSTM - tuple of tensors]
            masks (torch.tensor): Mask used in recurrent network
            deterministic (bool): If True, returns the mode of stochastic policy else return a random sample from the policy distribution.

        Returns:
            tuple: Tuple containing the following elements:
                - value (torch.tensor): State value function tensor.
                - action (torch.tensor): Predicted action.
                - action_log_probs (torch.tensor): Action log probs.
                - rnn_hxs (torch.tensor or tuple[torch.tensor, torch.tensor]): Recurrent hidden state.

        """
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        """
        State value function.

        Args:
            inputs (torch.tensor): Input state
            rnn_hxs (torch.tensor or tuple[torch.tensor, torch.tensor]): Recurrent network hidden state.
            masks (torch.tensor): Mask for the input state.

        Returns:
            torch.tensor: State value.

        """
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        """
        Evaluate actions.

        Args:
            inputs (torch.tensor): Observation or state input tensor.
            rnn_hxs (torch.tensor or tuple[torch.tensor, torch.tensor]):
            masks (torch.tensor):
            action (torch.tensor):

        Returns:
            tuple: The output tuple contains the following elements:
                - value (torch.tensor): State value function.
                - action_log_probs (torch.tensor): Action log probability.
                - dist_entropy (torch.tensor): Distribution entropy.
                - rnn_hxs (torch.tensor): Recurrent network hidden state.

        """
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):
    """
    Base of Neural Policy.

    Args:
        recurrent (bool): If True, the base is recurrent with GRU.
        recurrent_input_size (int): Input size of recurrent.
        hidden_size (int): Hidden size.
    """

    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        """
        Forward pass for Gated Recurrent Unit (GRU) based neural policy.

        Args:
            x (torch.tensor):
            hxs (torch.tensor):
            masks (torch.tensor):

        Returns:
            tuple: Output tuple contains

        """
        if x.size(0) == hxs.size(0):        # if single sample is passed as input
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0)
                         .any(dim=-1)
                         .nonzero()
                         .squeeze()
                         .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    """
    Base if neural network is convolutional network and inputs are images.

    Args:
        num_inputs (int): number of input channels
        recurrent (bool): If True, include a Recurrent layer in the network. Default is `False`.
        hidden_size (int): Size of fully connected layer (hidden layer) post convolution.

    Notes:
        Recurrent layer input is the flattened output of the convolutional layer.

    """
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(in_channels=num_inputs, out_channels=32, kernel_size=8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        def init__(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init__(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class QuadActor(nn.Module):
    def __init__(self, num_inputs, hidden_size):
        super(QuadActor, self).__init__()

        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.A1 = init_(nn.Linear(num_inputs, hidden_size))
        self.A2 = init_(nn.Linear(hidden_size, hidden_size))
        self.tanh = nn.Tanh()
        self.output_size = hidden_size

    def forward(self, x):
        x = self.tanh(self.A1(x))
        x = self.tanh(self.A2(x))
        return x


class TiltActor(nn.Module):
    """
    Tiltrotor Neural policy architecture.

    Args:
        num_inputs_1 (int): Shape of input 1. This size is calculated as the sum of sizes of vectors (pos, orien, vel, angvel.
        num_inputs_2 (int): Shape of input 2. This is equal to the number of independent servos on the tiltrotor.
        hidden_size (int): Size of hidden layer. Default is `64`.
        bhidden_size (int): Size of hidden layer in the branch. Default is `16`.

    Notes:
        ########### Schematic of neural network architecture ####################
        #                                                                       #
        #   IN_1 = (pos/orien/vel/angvel)                                       #
        #   IN_2 = (servos_angles)                                              #
        #                                                                       #
        #       IN_1      IN_2                                                  #
        #        |          |                                                   #
        #       A1         A3                                                   #
        #        |----+     |                                                   #
        #        |    |     |                                                   #
        #        |   (+--+--+)      <---[concatenation of A1 and A3 outputs]    #
        #        |       |                                                      #
        #        A2     A4                                                      #
        #        |       |                                                      #
        #        +---+---+                                                      #
        #            |                                                          #
        #          A_out                                                        #
        #                                                                       #
        #########################################################################
    """

    def __init__(self, num_inputs_1, num_inputs_2, hidden_size=64, bhidden_size=16):
        super(TiltActor, self).__init__()

        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.num_inputs_1 = num_inputs_1
        self.num_inputs_2 = num_inputs_2
        self.A1 = init_(nn.Linear(num_inputs_1, hidden_size))
        self.A2 = init_(nn.Linear(hidden_size, hidden_size))
        self.A3 = init_(nn.Linear(num_inputs_2, bhidden_size))
        self.A4 = init_(nn.Linear(bhidden_size + hidden_size, bhidden_size))
        self.tanh = nn.Tanh()
        self.output_size = hidden_size + bhidden_size

    def forward(self, x):
        """
        Forward pass of the tilt rotor policy.

        Args:
            x (torch.tensor): Tensor of shape (batch_size, 22)

        Returns:
            torch.tensor: Output of shape `hidden_size+b_hidden_size`.

        """
        x1 = torch.cat((x[:, 0:12], x[:, 16:22]), dim=1)  # pos, rot_flat, vel, ang_vel
        x2 = x[:, 12:16]  # tilt_angles

        x1A1 = self.tanh(self.A1(x1))
        x1A2 = self.tanh(self.A2(x1A1))

        x2A3 = self.tanh(self.A3(x2))

        xA4_in = torch.cat((x1A1, x2A3), dim=1)
        x2A4 = self.tanh(self.A4(xA4_in))

        out = torch.cat((x1A2, x2A4), dim=1)

        return out

    def load_quadrotor(self, quadrotor_model, quadrotor_weights_require_grad=True):
        """
        Load the trained quadrotor parameter weights

        Args:
            quadrotor_model: Quadrotor neural_network model saved after training the on quadrotor environment.
            quadrotor_weights_require_grad (bool): If ``False``, freeze the model weights which are taken from quadrotor. Default is `True`.

        Returns:

        """

        quadrotor_actor_layer_names = ["base.actor.A1.weight", "base.actor.A1.bias", "base.actor.A2.weight", "base.actor.A2.bias"]
        tiltrotor_actor_layer_names = ['A1.weight', 'A1.bias', 'A2.weight', 'A2.bias']

        quadrotor_param = quadrotor_model.state_dict()
        tiltrotor_param = self.state_dict()
        dict_tiltrotor_param = dict(tiltrotor_param)

        for qlname, tlname in zip(quadrotor_actor_layer_names, tiltrotor_actor_layer_names):
            dict_tiltrotor_param[tlname].data.copy_(quadrotor_param[qlname].data)

        self.load_state_dict(dict_tiltrotor_param)

        for param in self.A1.parameters():
            param.requires_grad = quadrotor_weights_require_grad

        for param in self.A2.parameters():
            param.requires_grad = quadrotor_weights_require_grad

    def info(self):
        print("\nTiltActor:\n", self)
        print("\nTiltActor state-dict keys:\n", dict(self.state_dict()).keys())


class MLPBase(NNBase):
    """
    Multi-layer perceptron (MLP) base for policy, i.e. no convolution-layers. This class contains actor and critic network definition.

    Args:
        num_inputs (int): Number of inputs.
        recurrent (bool): If ``True`` use recurrent module in the network (default is ``False``).
        hidden_size (int): Number of neurons in hidden layer.
        model (str): Robot model for which the policy is used (default: "default"). Available types include ["quadrotor", "tiltrotor", "default"].
        quadrotor_model_path (str): Model stored for quadrotor (default: ``None``).
        quadrotor_weights_require_grad (bool): If `False`, freeze quadrotor weights (parameters) when transfering the policy to tiltrotor. Default is ``True``.
        bhidden_size (int): Size of hidden layers in tiltrotor additional branch. Default is `16`.
    """

    metadata = {"models": ["quadrotor", "tiltrotor", "default"]}

    def __init__(self, num_inputs, recurrent=False, hidden_size=64,
                 model="default", quadrotor_model_path=None, quadrotor_weights_require_grad=True, bhidden_size=16):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0),
                        np.sqrt(2))

        self.model = model

        if model == "quadrotor":
            self.actor = QuadActor(num_inputs=num_inputs, hidden_size=hidden_size)
        elif model == "tiltrotor":
            self.actor = TiltActor(num_inputs_1=num_inputs-4, num_inputs_2=4, hidden_size=hidden_size, bhidden_size=bhidden_size)
        else:
            print("Loading default model-architecture...\n")
            self.actor = nn.Sequential(init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(), init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        if model == "tiltrotor" and quadrotor_model_path is not None:
            str_separation = "-------------------------------------------------------------------"
            print(str_separation + '\n'
                  + 'Loading quadrotor-actor model and freezing transfered weights...\n'
                  + 'WARNING: In the current form, it is assumes the quadrotor observation space is 18 dimensional and tiltrotor observation space is 22 dimensional.'
                  + 'If this has changed, then you will have to update the code.\n')

            quadrotor_model = torch.load(quadrotor_model_path, map_location=lambda storage, loc: storage)[0]
            self.actor.load_quadrotor(quadrotor_model=quadrotor_model, quadrotor_weights_require_grad=quadrotor_weights_require_grad)
            with torch.no_grad():
                self.critic[0].weight[:, :18] = copy.deepcopy(quadrotor_model.base.critic[0].weight)   # layer 1 of critic
                self.critic[2].weight = copy.deepcopy(quadrotor_model.base.critic[2].weight)           # layer 2 of critic
                self.critic_linear.weight = copy.deepcopy(quadrotor_model.base.critic_linear.weight)
            print('Finished loading quadrotor-actor model!!\n'+str_separation+'\n')

        self.train()

    @property
    def output_size(self):
        """
        Output size of the network base. This method used in Tiltrotor policy training.

        :return: output size of the hidden layer
        :rtype: int
        """

        if self.model == "default":
            op_sz = self._hidden_size
        else:
            op_sz = self.actor.output_size
        return op_sz

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
