import math
import copy
import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn as nn
from common.networks import PolicyNetworkFactory
from common import util



class MlpPolicy(torch.nn.Module):
    """a weight-sharing dynamic graph policy that changes its structure based on different morphologies and passes messages between nodes,
        while injecting structural bias"""
    def __init__(
        self,
        state_dim,
        action_dim,
        msg_dim,
        batch_size,
        max_action,
        max_children,
        disable_fold,
        td,
        bu,
        args=None,
    ):
        super(MlpPolicy, self).__init__()
        # self.num_limbs = 1
        # self.msg_down = [None] * self.num_limbs
        # self.msg_up = [None] * self.num_limbs
        # self.action = [None] * self.num_limbs
        # self.input_state = [None] * self.num_limbs
        self.max_action = max_action
        self.msg_dim = msg_dim
        self.batch_size = batch_size
        # self.max_children = max_children
        # self.disable_fold = disable_fold
        self.state_dim = state_dim
        self.action_dim = action_dim
        # print("body",len(args.graphs[args.envs_train_names[-1]]),state_dim)

        self.actor = PolicyNetworkFactory.get(self.state_dim*len(args.graphs[args.envs_train_names[-1]]), args.action_space[args.envs_train_names[-1]],  args.agent['policy_network']["hidden_dims"],args.agent['policy_network']["deterministic"]).to(util.device)
        # print("actor",self.actor)

    def forward(self, state, mode="train"):
        # if mode == "inference":
        #     temp = self.batch_size
        #     self.batch_size = 1

        # self.input_state = state.reshape(state.shape[0], self.num_limbs, -1).permute(
        #     1, 0, 2
        # )
        # print("state",state.shape)
        self.action = self.actor(state)
        
        self.action = self.max_action * torch.tanh(self.action)
        # print("self.action", self.action.shape)

        # because of the permutation of the states, we need to unpermute the actions now so that the actions are (batch,actions)
        # self.action = self.action.permute(1, 0, 2)
        # self.action = self.action.contiguous().view(self.action.shape[0], -1)
        # print("self.action", self.action.shape)

        # if mode == "inference":
        #     self.batch_size = temp
        return self.action


    # def forward_attn(self, state, mode="train"):
    #     self.clear_buffer()
    #     # if mode == "inference":
    #     #     temp = self.batch_size
    #     #     self.batch_size = 1

    #     self.input_state = state.reshape(state.shape[0], self.num_limbs, -1).permute(
    #         1, 0, 2
    #     )
    #     self.action, attn_weights_rel_pos = self.actor(self.input_state, self.graph)
        
    #     self.action = self.max_action * torch.tanh(self.action)

    #     # because of the permutation of the states, we need to unpermute the actions now so that the actions are (batch,actions)
    #     self.action = self.action.permute(1, 0, 2)

    #     # if mode == "inference":
    #     #     self.batch_size = temp

    #     return torch.squeeze(self.action), attn_weights_rel_pos

    def change_morphology(self, graph):
        self.graph = graph
        self.parents = graph['parents']
        self.num_limbs = len(self.parents)
        self.msg_down = [None] * self.num_limbs
        self.msg_up = [None] * self.num_limbs
        self.action = [None] * self.num_limbs
        self.input_state = [None] * self.num_limbs
