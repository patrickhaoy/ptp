import abc  
import torch
from torch import nn

from rlkit.torch.core import PyTorchModule
from rlkit.torch.core import torch_ify, elem_or_tuple_to_numpy  


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class MyMlpQNetwork(PyTorchModule):

    def __init__(
            self,
            obs_dim,
            action_dim,
            hidden_dim=256,
            hidden_sizes=None,
            last_b_init_value=None,
    ):
        super().__init__()

        self._layers = nn.Sequential(
            nn.Linear(
                # 2 * obs_dim + action_dim,
                2 * obs_dim,
                hidden_dim,
            ),
            nn.ReLU(True),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(True),

            nn.Linear(
                hidden_dim,
                1,
                bias=False,
            ),
        )

    def forward(self, obs_and_goal, action, training=False):
        output = self._layers(obs_and_goal)
        return output


class MyMlpVNetwork(PyTorchModule):

    def __init__(
            self,
            obs_dim,
            hidden_dim=16,
            hidden_sizes=None,
            last_b_init_value=None,
    ):
        super().__init__()

        del hidden_sizes

        self._layers = nn.Sequential(
            nn.Linear(
                2 * obs_dim,
                hidden_dim,
            ),
            nn.ReLU(True),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(True),

            nn.Linear(
                hidden_dim,
                1,
                bias=False,
            ),
        )

    def forward(self, obs_and_goal, training=False):
        output = self._layers(torch.cat([obs_and_goal], -1))
        return output


class L2QNetwork(PyTorchModule):

    def __init__(
            self,
            obs_dim,
            action_dim,
            hidden_dim=16,
            hidden_sizes=None,
            last_b_init_value=None,
    ):
        super().__init__()

        del hidden_sizes
        self.obs_dim = obs_dim

        self.obs_encoder = nn.Sequential(
            nn.Linear(
                obs_dim + action_dim,
                hidden_dim,
            ),
            nn.ReLU(True),

            nn.Linear(
                hidden_dim,
                hidden_dim,
                bias=False,
            ),
        )

        self.goal_encoder = nn.Sequential(
            nn.Linear(
                obs_dim,
                hidden_dim,
            ),
            nn.ReLU(True),

            nn.Linear(
                hidden_dim,
                hidden_dim,
                bias=False,
            ),
        )

    def forward(self, obs_and_goal, action, training=False):
        obs = obs_and_goal[..., :self.obs_dim]
        goal = obs_and_goal[..., self.obs_dim:]

        obs_feat = self.obs_encoder(torch.cat([obs, action], -1))
        goal_feat = self.goal_encoder(goal)
        h = -torch.norm(obs_feat - goal_feat, dim=-1)

        output = h[..., None]

        return output


class L2VNetwork(PyTorchModule):

    def __init__(
            self,
            obs_dim,
            hidden_dim=16,
            hidden_sizes=None,
            last_b_init_value=None,
    ):
        super().__init__()

        del hidden_sizes

        self.obs_dim = obs_dim

        self.obs_encoder = nn.Sequential(
            nn.Linear(
                obs_dim,
                hidden_dim,
            ),
            nn.ReLU(True),

            nn.Linear(
                hidden_dim,
                hidden_dim,
                bias=False,
            ),
        )

    def forward(self, obs_and_goal, training=False):
        obs = obs_and_goal[..., :self.obs_dim]
        goal = obs_and_goal[..., self.obs_dim:]

        obs_feat = self.obs_encoder(obs)
        goal_feat = self.obs_encoder(goal)
        h = -torch.norm(obs_feat - goal_feat, dim=-1)

        output = h[..., None]
        return output


class BilinearQNetwork(PyTorchModule):

    def __init__(
            self,
            obs_dim,
            action_dim,
            hidden_dim=256,
            hidden_sizes=None,
            last_b_init_value=None,
    ):
        super().__init__()

        del hidden_sizes
        self.obs_dim = obs_dim

        self.obs_encoder = nn.Sequential(
            nn.Linear(
                obs_dim + action_dim,
                hidden_dim,
                bias=False,
            ),
            nn.ReLU(True),

            nn.Linear(
                hidden_dim,
                hidden_dim,
                bias=False,
            ),
        )

        self.goal_encoder = nn.Sequential(
            nn.Linear(
                obs_dim + obs_dim,
                hidden_dim,
                bias=False,
            ),
            nn.ReLU(True),

            nn.Linear(
                hidden_dim,
                hidden_dim,
                bias=False,
            ),
        )

    def forward(self, obs_and_goal, action, training=False):
        obs = obs_and_goal[..., :self.obs_dim]
        goal = obs_and_goal[..., self.obs_dim:]

        obs_feat = self.obs_encoder(torch.cat([obs, action], -1))

        goal_feat = self.goal_encoder(torch.cat([obs, goal], -1))

        h = -torch.norm(obs_feat - goal_feat, dim=-1)
        output = h[..., None]

        return output


class BilinearVNetwork(PyTorchModule):

    def __init__(
            self,
            obs_dim,
            hidden_dim=256,
            hidden_sizes=None,
            last_b_init_value=None,
    ):
        super().__init__()

        del hidden_sizes

        self.obs_dim = obs_dim

        self.obs_encoder = nn.Sequential(
            nn.Linear(
                obs_dim,
                hidden_dim,
                bias=False,
            ),
        )

    def forward(self, obs_and_goal, training=False):
        obs = obs_and_goal[..., :self.obs_dim]
        goal = obs_and_goal[..., self.obs_dim:]

        obs_feat = self.obs_encoder(obs)
        goal_feat = self.obs_encoder(goal)
        h = -torch.norm(obs_feat - goal_feat, dim=-1)

        output = h[..., None]

        return output


class DiffVFNetwork(PyTorchModule):

    def __init__(
        self,
        obs_dim,
        hidden_dim=128,
        init_w=3e-3,
    ):
        super().__init__()

        self.obs_dim = obs_dim

        obs_encoding_dim = max(1, int(obs_dim / 4))
        self._obs_fc = nn.Sequential(
            nn.Linear(
                obs_dim,
                obs_encoding_dim,
            ),
            nn.ReLU(True),
        )

        diff_encoding_dim = obs_dim
        self._diff_fc = nn.Sequential(
            nn.Linear(
                diff_encoding_dim,
                obs_dim,
            ),
            nn.ReLU(True),
        )

        self._joint_layers = nn.Sequential(
            nn.Linear(
                obs_encoding_dim + diff_encoding_dim,
                hidden_dim,
            ),
            nn.ReLU(True),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(True),
        )

        self._output_layer = nn.Linear(hidden_dim, 1)
        self._output_layer.weight.data.uniform_(-init_w, init_w)
        self._output_layer.bias.data.fill_(0)

    def forward(self, input_obs):

        obs = input_obs[..., :self.obs_dim]
        context = input_obs[..., self.obs_dim:]
        diff = context - obs

        obs_encoding = self._obs_fc(obs)
        diff_encoding = self._diff_fc(diff)

        net = torch.cat((obs_encoding, diff_encoding), dim=-1)
        net = self._joint_layers(net)

        output = self._output_layer(net)

        return output
