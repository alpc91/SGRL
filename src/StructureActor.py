import math
import copy
import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, MultiheadAttention
from ModularActor import ActorGraphPolicy
from attentions import multi_head_attention_forward
from common import util


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class ConcatPositionalEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0, max_node=15, **kwargs):
        super(ConcatPositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.num_positions = kwargs['num_positions']
        unit_d = d_model // self.num_positions
        self.embeddings = nn.ModuleList(
            [nn.Embedding(
                max_node, unit_d + d_model % self.num_positions if i == self.num_positions - 1 else unit_d)
                for i in range(self.num_positions)])

    def forward(self, positional_indices=[]):
        embeddings = torch.cat([self.embeddings[i](pos) for i, pos in enumerate(positional_indices)], dim=1) # (N, d)
        return self.dropout(embeddings)


class MyMultiheadAttention(MultiheadAttention):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super(MyMultiheadAttention, self).__init__(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim)
    
    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)


class MyTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(MyTransformerEncoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout, activation)
        self.self_attn = MyMultiheadAttention(d_model, nhead, dropout=dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # attn weight
        src2, attn_weight = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        if hasattr(self, "activation"):
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        else:  # for backward compatibility
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class RepeatTransformerEncoder(TransformerEncoder):
    def __init__(self, encoder_layer, num_layers, ninp, nhead,
                norm=None, d_rel=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.nhead = nhead
        self.pos_scaling = float(ninp / nhead * 2.) ** -0.5
        self.rel_encoder = nn.Linear(d_rel, nhead)

    def forward(self, src, pos=None, rel=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layers in turn.
        Args:
            src: the sequnce to the encoder (required).
            pos: position embedding (N, d)
            rel (mask): the mask for the src sequence (optional). (N, N, d_rel)
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        output = src

        N, B, d = src.shape
        
        # process position embedding
        output = output + pos.unsqueeze(1)  # (N, B, d) + (N, 1, d)

        # process relation embedding
        rel = self.rel_encoder(rel).unsqueeze(0) # (1, N, N, H)
        rel = torch.cat([rel] * B, dim=0)   # (B, N, N, H)
        rel = rel.permute(2, 1, 0, 3).contiguous().view(N, N, B * self.nhead).transpose(0, 2) # (BH, N, N)
            
        # attn_weight
        for i in range(self.num_layers):
            # attn_weight
            output = self.layers[i](output, src_mask=rel if i == 0 else None,
                                    src_key_padding_mask=src_key_padding_mask)
        if self.norm:
            output = self.norm(output)

        return output


class TransformerModel(nn.Module):
    def __init__(
        self,
        feature_size,
        output_size,
        ninp,
        nhead,
        nhid,
        nlayers,
        dropout=0.5,
        condition_decoder=False,
        transformer_norm=False,
        num_positions=0,
        rel_size=1,
    ):
        super(TransformerModel, self).__init__()
        self.model_type = "Structure"
        
        ninp_att = ninp

        self.pos_encoder = ConcatPositionalEmbedding(
                                ninp, dropout, num_positions=num_positions
                            )

        encoder_layers = MyTransformerEncoderLayer(ninp_att, nhead, nhid, dropout)

        self.transformer_encoder = RepeatTransformerEncoder(
            encoder_layers,
            nlayers,
            ninp,
            nhead,
            norm=nn.LayerNorm(ninp_att) if transformer_norm else None,
            d_rel=rel_size,
        )
        self.encoder = nn.Linear(feature_size, ninp)
        self.ninp = ninp
        self.condition_decoder = condition_decoder

        ninp_dec = ninp_att + feature_size if condition_decoder else ninp_att

        self.decoder = nn.Linear(ninp_dec, output_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, graph=None):
        encoded = self.encoder(src) * math.sqrt(self.ninp)
        pos = self.pos_encoder(graph['traversals']) # (N, d)        
        rel = graph['relation']   # (N, N, d_rel)
        output = self.transformer_encoder(encoded, pos, rel)
        if self.condition_decoder:
            output = torch.cat([output, src], axis=2)

        output = self.decoder(output)

        return output

    # @property
    # def weights(self):
    #     return [net.weight for net in self.networks if isinstance(net, torch.nn.modules.linear.Linear)]


class StructurePolicy(ActorGraphPolicy):
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
        super(ActorGraphPolicy, self).__init__()
        self.num_limbs = 1
        self.msg_down = [None] * self.num_limbs
        self.msg_up = [None] * self.num_limbs
        self.action = [None] * self.num_limbs
        self.input_state = [None] * self.num_limbs
        self.max_action = max_action
        self.msg_dim = msg_dim
        self.batch_size = batch_size
        self.max_children = max_children
        self.disable_fold = disable_fold
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor = TransformerModel(
            self.state_dim,
            action_dim,
            args.attention_embedding_size,
            args.attention_heads,
            args.attention_hidden_size,
            args.attention_layers,
            args.dropout_rate,
            condition_decoder=args.condition_decoder_on_features,
            transformer_norm=args.transformer_norm,
            num_positions=len(args.traversal_types),
            rel_size=args.rel_size,
        ).to(util.device)
        # print("actor",self.actor)

    def forward(self, state, mode="train"):
        self.clear_buffer()
        # if mode == "inference":
        #     temp = self.batch_size
        #     self.batch_size = 1

        self.input_state = state.reshape(state.shape[0], self.num_limbs, -1).permute(
            1, 0, 2
        )
        self.action = self.actor(self.input_state, self.graph)
        
        self.action = self.max_action * torch.tanh(self.action)
        # print("self.action", self.action.shape)

        # because of the permutation of the states, we need to unpermute the actions now so that the actions are (batch,actions)
        self.action = self.action.permute(1, 0, 2)
        self.action = self.action.contiguous().view(self.action.shape[0], -1)
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
