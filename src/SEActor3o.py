import math
import copy
import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, MultiheadAttention
from ModularActor import ActorGraphPolicy
from subequivariant_attentions import multi_head_attention_forward
from common import util
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np


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
        self.q_proj = nn.Linear(embed_dim+embed_dim, embed_dim*2)
        self.k_proj = nn.Linear(embed_dim+embed_dim, embed_dim*2)
        self.v_proj = nn.Linear(embed_dim+embed_dim, embed_dim*2)
        self.vg_proj = nn.Linear(embed_dim, embed_dim*2-2*num_heads,bias=False)
        self.ng_out = nn.Linear(embed_dim*2, embed_dim)
        self.g_out = nn.Linear(embed_dim*2, embed_dim,bias=False)
        self.g_proj = nn.Linear(embed_dim, 32-2,bias=False)
        self.linear_g1 = torch.nn.Linear(32*32,embed_dim*2)
        self.linear_g2 = torch.nn.Linear(embed_dim*2,embed_dim)
        
    
    def forward(self, g_src, ng_src, gdir, key_padding_mask=None, need_weights=True, attn_mask=None):
        return multi_head_attention_forward(
                g_src, ng_src, gdir, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask,
                q_proj=self.q_proj,    
                k_proj=self.k_proj, 
                v_proj=self.v_proj, 
                vg_proj=self.vg_proj, 
                g_proj=self.g_proj,
                ng_out=self.ng_out,
                g_out = self.g_out,
                linear_g1 = self.linear_g1,
                linear_g2 = self.linear_g2,
                )


class MyTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0., activation="relu"):
        super(MyTransformerEncoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout, activation)
        self.self_attn = MyMultiheadAttention(d_model, nhead, dropout=dropout)
        self.g_proj2 = nn.Linear(d_model, 32-2,bias=False)
        self.g_proj3 = nn.Linear(d_model, 32-2,bias=False)
        self.linear_g1 = torch.nn.Linear(32*32,dim_feedforward)
        self.linear_g2 = torch.nn.Linear(dim_feedforward,d_model)
        self.linear1 = torch.nn.Linear(d_model+d_model,dim_feedforward)
        # self.linear2 = torch.nn.Linear(dim_feedforward, d_model)
        self.linear3 = torch.nn.Linear(d_model+d_model,dim_feedforward)
        self.linear4 = torch.nn.Linear(dim_feedforward,32*32)
        self.linear5 = torch.nn.Linear(32,d_model,bias=False)

    def forward(self, g_src, ng_src, gdir, src_mask=None, src_key_padding_mask=None):
        
        # attn weight
        # print(g_src.shape,ng_src.shape)
        tgt_len, bsz, embed_dim = ng_src.size()
        g_src1, ng_src1 = self.self_attn(g_src, ng_src, gdir, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        g_src = g_src + g_src1
        ng_src = ng_src + ng_src1
        ng_src = self.norm1(ng_src)

        g_src2 = g_src1.contiguous().view(tgt_len*bsz, 3,-1)
        g_src2 = self.g_proj2(g_src2)
        g_src2 = torch.cat([g_src2,gdir],-1)
        g_src2 = torch.bmm(g_src2.transpose(-2, -1), g_src2)
        F_norm = torch.norm(g_src2,dim=(-2,-1),keepdim=True)+1.0
        F_norm = F_norm.contiguous().view(tgt_len,bsz,-1)
        # print("F_norm",F_norm.shape)
        g_src2 = g_src2.contiguous().view(tgt_len, bsz, -1)
        g_src2 = self.linear_g2(F.relu(self.linear_g1(g_src2)))
        ng_src2 = torch.cat([g_src2, ng_src],dim=-1)  #node*1*(128+1024)


        mat3 = self.linear4(F.relu(self.linear3(ng_src2)))/ F_norm
        # print(mat3.shape)#  #node*1*C*C
        mat3 = mat3.contiguous().view(tgt_len*bsz, 32,32)  #node*C*C
        g_src3 = g_src1.contiguous().view(tgt_len*bsz, 3, -1)  
        g_src3 = self.g_proj3(g_src3)
        g_src3 = torch.cat([g_src3,gdir],-1)
        g_src3 = torch.bmm(g_src3, mat3)  # node*3*C *  node*C*C        node*3*C
        # print(g_src3.shape)
        g_src3 = g_src3.contiguous().view(tgt_len,bsz, 3,-1)
        g_src = g_src + self.linear5(g_src3)  #node*3*C
        # print("g_src",g_src.shape)
        

        # if hasattr(self, "activation"):
        #     ng_src3 = self.linear2(self.activation(self.linear1(ng_src2)))/F_norm
        # else:  # for backward compatibility
        ng_src3 = self.linear2(F.relu(self.linear1(ng_src2)))/F_norm
        ng_src = ng_src + ng_src3
        ng_src = self.norm2(ng_src)
        # print(g_src.shape,ng_src.shape)
        return g_src, ng_src

class RepeatTransformerEncoder(TransformerEncoder):
    def __init__(self, encoder_layer, num_layers, ninp, nhead,
                norm=None, d_rel=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.nhead = nhead
        self.pos_scaling = float(ninp / nhead * 2.) ** -0.5
        self.rel_encoder = nn.Linear(d_rel, self.nhead)

    def forward(self, g_src, ng_src, gdir, pos=None, rel=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layers in turn.
        Args:
            src: the sequnce to the encoder (required).
            pos: position embedding (N, d)
            rel (mask): the mask for the src sequence (optional). (N, N, d_rel)
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        # output = ng_src

        N, B, d = ng_src.shape
        
        # process position embedding
        ng_src = ng_src + pos.unsqueeze(1)  # (N, B, d) + (N, 1, d)

        # process relation embedding
        rel = self.rel_encoder(rel).unsqueeze(0) # (1, N, N, H)
        rel = torch.cat([rel] * B, dim=0)   # (B, N, N, H)
        rel = rel.permute(2, 1, 0, 3).contiguous().view(N, N, B * self.nhead).transpose(0, 2) # (BH, N, N)
            
        # attn_weight
        for i in range(self.num_layers):
            g_src, ng_src = self.layers[i](g_src, ng_src, gdir, src_mask=rel if i == 0 else None,
                                    src_key_padding_mask=src_key_padding_mask)
        if self.norm:
            ng_src = self.norm(ng_src)

        return g_src, ng_src


class TransformerModel(nn.Module):
    def __init__(
        self,
        feature_size,
        output_size,
        ninp,
        nhead,
        nhid,
        nlayers,
        dropout=0.0,
        condition_decoder=False,
        transformer_norm=False,
        num_positions=0,
        rel_size=1,
    ):
        dropout = 0
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
        self.g_num = 8
        ng_feature_size = feature_size - self.g_num*3
        self.g_encoder = nn.Linear(self.g_num, ninp, bias = False)
        self.encoder = nn.Linear(ng_feature_size, ninp)
        self.ninp = ninp
        self.ninp_att = ninp_att
        self.condition_decoder = condition_decoder

        self.gg_proj = nn.Linear(ninp_att+self.g_num, 32-2,bias=False)
        self.linear1_g = torch.nn.Linear(32*32,ninp_att)
        self.linear2_g = torch.nn.Linear(ninp_att,ninp_att)
        self.linear1_ng = torch.nn.Linear(ninp_att+ng_feature_size,ninp_att)
        self.linear2_ng = torch.nn.Linear(ninp_att,ninp_att)

        ninp_dec = ninp_att+ninp_att
        # print(ninp_dec, output_size)
        self.output_size = output_size
        if self.output_size == 1:
            self.decoder_ng = nn.Linear(ninp_dec, output_size)
        else:
        #     output_size = output_size // 2
        #     self.output_g = (self.g_num-1)*3
        #     self.output_ng = output_size - self.output_g
        #     self.decoder_ng = nn.Linear(ninp_dec, self.output_ng*2)
            self.decoder_g = torch.nn.Linear(32,self.output_size,bias=False) # 4: no g
            self.linear1_m = torch.nn.Linear(ninp_dec,ninp_dec)
            self.linear2_m = torch.nn.Linear(ninp_dec,32*32)
            self.g_proj = nn.Linear(ninp_att+self.g_num, 32-2,bias=False)
            
        self.init_weights()


    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.g_encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, graph=None):
        # theta = np.pi#math.pi*rad

        tgt_len, bsz, embed_dim = src.size()
        g_src0 = src[:,:,:self.g_num*3]
        g_src0 = g_src0.contiguous().view(tgt_len,bsz, -1,3).transpose(-2, -1)
        ng_src0 = src[:,:,self.g_num*3:]

        g_src = src[:,:,:self.g_num*3]
        g_src = g_src.contiguous().view(tgt_len,bsz, -1,3).transpose(-2, -1)

        # print(g_src.shape,gdir.shape,gdir)
        
        # g_src = g_src.contiguous().view(tgt_len*bsz, 3,-1)
        # g_src0 = g_src0.contiguous().view(tgt_len*bsz, 3,-1)
        # O = torch.tensor([[math.cos(theta), -math.sin(theta), 0],
        #     [math.sin(theta), math.cos(theta), 0],
        #     [0, 0, 1]]).unsqueeze(0).to(g_src.device)
        # O = O.repeat(g_src.shape[0],1,1)
        # # print(O.shape)
        # g_src = torch.bmm(O, g_src)
        # g_src0 = torch.bmm(O, g_src0)
        # g_src = g_src.contiguous().view(tgt_len,bsz, 3,-1)
        # g_src0 = g_src0.contiguous().view(tgt_len,bsz, 3,-1)
        # print("input g_src0",g_src0.transpose(-2, -1))

        gdir = g_src[:,:,:,1:3]
        gdir = gdir.contiguous().view(tgt_len*bsz, 3,-1)

        g_src = self.g_encoder(g_src) * math.sqrt(self.ninp)
        ng_src = src[:,:,self.g_num*3:]
        ng_src = self.encoder(ng_src) * math.sqrt(self.ninp)
        pos = self.pos_encoder(graph['traversals']) # (N, d)        
        rel = graph['relation']   # (N, N, d_rel)
        g_src, ng_src = self.transformer_encoder(g_src, ng_src, gdir, pos, rel) 
        output_ng = torch.cat([ng_src0, ng_src],dim=-1)  
        output_g = torch.cat([g_src0, g_src],dim=-1)

        output_gg = output_g.contiguous().view(tgt_len*bsz, 3,-1)
        output_gg = self.gg_proj(output_gg)
        output_gg = torch.cat([output_gg,gdir],-1)
        output_gg = torch.bmm(output_gg.transpose(-2, -1), output_gg)
        F_norm = torch.norm(output_gg,dim=(-2,-1),keepdim=True)+1.0
        F_norm = F_norm.contiguous().view(tgt_len,bsz,-1)
        output_gg = output_gg.contiguous().view(tgt_len, bsz, -1)
        output_gg = self.linear2_g(F.relu(self.linear1_g(output_gg)))

        output_ng = self.linear2_ng(F.relu(self.linear1_ng(output_ng)))
        output_ng = torch.cat([output_gg, output_ng],dim=-1)  
        if self.output_size == 1:
            output = self.decoder_ng(output_ng)/F_norm
            # print("Before: value",output)
        else:
            mat = self.linear2_m(F.relu(self.linear1_m(output_ng)))/ F_norm
            mat = mat.contiguous().view(tgt_len*bsz, 32,32)  #node*C*C
            output_g = output_g.contiguous().view(tgt_len*bsz, 3, -1)  
            output_g = self.g_proj(output_g)
            output_g = torch.cat([output_g,gdir],-1)
            output_g = torch.bmm(output_g, mat)  # node*3*C *  node*C*C        node*3*C
            output_g = output_g.contiguous().view(tgt_len,bsz, 3,-1)
            g_src = self.decoder_g(output_g) #tgt*bz*3*C
            g_src = g_src.contiguous().view(tgt_len*bsz, 3,-1)
            g_src0 = g_src0[:,:,:,5:8].contiguous().view(tgt_len*bsz, 3,-1)
            # print("output",g_src.transpose(-2, -1),g_src0.transpose(-2, -1))

            output0 = torch.bmm(g_src0[:,:,0:1].transpose(-2, -1), g_src[:,:,0:1]).contiguous().view(tgt_len*bsz, -1)
            output1 = torch.bmm(g_src0[:,:,1:2].transpose(-2, -1), g_src[:,:,1:2]).contiguous().view(tgt_len*bsz, -1)
            output2 = torch.bmm(g_src0[:,:,2:].transpose(-2, -1), g_src[:,:,2:]).contiguous().view(tgt_len*bsz, -1)

            # print("Before: g_src",g_src)
            output = torch.cat([output0,output1,output2],-1).contiguous().view(tgt_len,bsz,-1)
            # print("output",output)



        # #set test
        # rad = np.random.rand()
        # theta = math.pi*rad
        # O = torch.tensor([[math.cos(theta), -math.sin(theta), 0],
        #             [math.sin(theta), math.cos(theta), 0],
        #             [0, 0, 1]]).unsqueeze(0).to(g_src.device)

        # g_src0 = src[:,:,:self.g_num*3]
        # g_src0 = g_src0.contiguous().view(tgt_len*bsz, -1,3).transpose(-2, -1)
        # ng_src0 = src[:,:,self.g_num*3:]

        # g_src = src[:,:,:self.g_num*3]
        # g_src = g_src.contiguous().view(tgt_len*bsz, -1,3).transpose(-2, -1)

        # O = O.repeat(g_src.shape[0],1,1)
        # # print(O.shape)
        # g_src = torch.bmm(O, g_src)
        # g_src0 = torch.bmm(O, g_src0)
        # g_src = g_src.contiguous().view(tgt_len,bsz, 3,-1)
        # g_src0 = g_src0.contiguous().view(tgt_len,bsz, 3,-1)

        # g_src = self.g_encoder(g_src) * math.sqrt(self.ninp)
        # ng_src = src[:,:,self.g_num*3:]
        # ng_src = self.encoder(ng_src) * math.sqrt(self.ninp)
        # pos = self.pos_encoder(graph['traversals']) # (N, d)        
        # rel = graph['relation']   # (N, N, d_rel)
        # g_src, ng_src = self.transformer_encoder(g_src, ng_src, gdir, pos, rel) 
        # output_ng = torch.cat([ng_src0, ng_src],dim=-1)  
        # output_g = torch.cat([g_src0, g_src],dim=-1)

        # output_gg = output_g.contiguous().view(tgt_len*bsz, 3,-1)
        # output_gg = self.gg_proj(output_gg)
        # output_gg = torch.cat([output_gg,gdir],-1)
        # output_gg = torch.bmm(output_gg.transpose(-2, -1), output_gg)
        # F_norm = torch.norm(output_gg,dim=(-2,-1),keepdim=True)+1.0
        # F_norm = F_norm.contiguous().view(tgt_len,bsz,-1)
        # output_gg = output_gg.contiguous().view(tgt_len, bsz, -1)
        # output_gg = self.linear2_g(F.relu(self.linear1_g(output_gg)))

        # output_ng = self.linear2_ng(F.relu(self.linear1_ng(output_ng)))
        # output_ng = torch.cat([output_gg, output_ng],dim=-1)  
        
        # if self.output_size == 1:
        #     output = self.decoder_ng(output_ng)/F_norm
        #     print("After: value",output)
        # else:
        #     mat = self.linear2_m(F.relu(self.linear1_m(output_ng)))/ F_norm
        #     mat = mat.contiguous().view(tgt_len*bsz, 32,32)  #node*C*C
        #     output_g = output_g.contiguous().view(tgt_len*bsz, 3, -1)  
        #     output_g = self.g_proj(output_g)
        #     output_g = torch.cat([output_g,gdir],-1)
        #     output_g = torch.bmm(output_g, mat)  # node*3*C *  node*C*C        node*3*C
        #     output_g = output_g.contiguous().view(tgt_len,bsz, 3,-1)
        #     g_src = self.decoder_g(output_g) #tgt*bz*3*C
        #     g_src = g_src.contiguous().view(tgt_len*bsz, 3,-1)
        #     g_src0 = g_src0[:,:,:,5:8].contiguous().view(tgt_len*bsz, 3,-1)

        #     output0 = torch.bmm(g_src0[:,:,0:1].transpose(-2, -1), g_src[:,:,0:1]).contiguous().view(tgt_len*bsz, -1)
        #     output1 = torch.bmm(g_src0[:,:,1:2].transpose(-2, -1), g_src[:,:,1:2]).contiguous().view(tgt_len*bsz, -1)
        #     output2 = torch.bmm(g_src0[:,:,2:].transpose(-2, -1), g_src[:,:,2:]).contiguous().view(tgt_len*bsz, -1)

        #     print("After: g_src",g_src)
        #     output = torch.cat([output0,output1,output2],-1).contiguous().view(tgt_len,bsz,-1)
        #     print("output",output)    

        #     theta = -math.pi*rad
        #     O = torch.tensor([[math.cos(theta), -math.sin(theta), 0],
        #             [math.sin(theta), math.cos(theta), 0],
        #             [0, 0, 1]]).unsqueeze(0).to(g_src.device)
        #     O = O.repeat(g_src.shape[0],1,1)
        #     g_src = torch.bmm(O, g_src)
        #     print("Rotate: g_src",g_src)

        return output


class SEPolicy(ActorGraphPolicy):
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
        # print(self.actor)

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

        # because of the permutation of the states, we need to unpermute the actions now so that the actions are (batch,actions)
        self.action = self.action.permute(1, 0, 2)
        self.action = self.action.contiguous().view(self.action.shape[0], -1)

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
