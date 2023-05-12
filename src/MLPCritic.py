from __future__ import print_function
import torch
import torch.nn as nn

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from common.networks import MLPNetwork
from common import util

class MlpCritic(nn.Module):
    """a weight-sharing dynamic graph policy that changes its Relation based on different morphologies and passes messages between nodes"""

    def __init__(
        self,
        state_dim,
        action_dim,
        msg_dim,
        batch_size,
        max_children,
        disable_fold,
        td,
        bu,
        args=None,
    ):
        super().__init__()
        self.msg_dim = msg_dim
        self.batch_size = batch_size
        self.max_children = max_children
        self.disable_fold = disable_fold
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.critic1 = MLPNetwork((state_dim + action_dim)*len(args.graphs[args.envs_train_names[-1]]), 1, **args.agent['q_network']).to(util.device)
        

        # print("critic",self.critic1)
        self.critic2 = MLPNetwork((state_dim + action_dim)*len(args.graphs[args.envs_train_names[-1]]), 1, **args.agent['q_network']).to(util.device)

    def forward(self, state, action):
        inpt = torch.cat([state, action], dim=-1)

        self.x1 = self.critic1(inpt)
        self.x2 = self.critic2(inpt)
        return self.x1, self.x2

    def Q1(self, state, action):
        inpt = torch.cat([state, action], dim=-1)
        self.x1 = self.critic1(inpt)
        return self.x1


    def change_morphology(self, graph):
        self.graph = graph
        self.parents = graph['parents']
        self.num_limbs = len(self.parents)
        self.msg_down = [None] * self.num_limbs
        self.msg_up = [None] * self.num_limbs
        self.action = [None] * self.num_limbs
        self.input_state = [None] * self.num_limbs
