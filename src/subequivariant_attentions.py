import torch
import torch.nn.functional as F

def multi_head_attention_forward(
                                 g_src,
                                 ng_src,
                                 gdir,
                                 embed_dim_to_check,              # type: int
                                 num_heads,                       # type: int
                                 in_proj_weight,                  # type: Tensor
                                 in_proj_bias,                    # type: Tensor
                                 bias_k,                          # type: Optional[Tensor]
                                 bias_v,                          # type: Optional[Tensor]
                                 add_zero_attn,                   # type: bool
                                 dropout_p,                       # type: float
                                 out_proj_weight,                 # type: Tensor
                                 out_proj_bias,                   # type: Tensor
                                 training=True,                   # type: bool
                                 key_padding_mask=None,           # type: Optional[Tensor]
                                 need_weights=True,               # type: bool
                                 attn_mask=None,                  # type: Optional[Tensor]
                                 use_separate_proj_weight=False,  # type: bool
                                 q_proj=None,              # type: Optional[Tensor]
                                 k_proj=None,              # type: Optional[Tensor]
                                 v_proj=None,              # type: Optional[Tensor]
                                 vg_proj=None,              # type: Optional[Tensor]
                                 g_proj=None,
                                 ng_out=None,
                                 g_out = None,
                                 linear_g1 = None,
                                 linear_g2 = None,
                                 static_k=None,                   # type: Optional[Tensor]
                                 static_v=None                    # type: Optional[Tensor]
                                 ):
    # type: (...) -> Tuple[Tensor, Optional[Tensor]]
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer).
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in differnt forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
        - attn_mask: :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """

    # print(g_src.shape,ng_src.shape)
    tgt_len, bsz, embed_dim = ng_src.size()
    assert embed_dim == embed_dim_to_check
    # assert key.size() == value.size()

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim*2) ** -0.5

    g_src2 = g_src.contiguous().view(tgt_len*bsz, 3,-1)
    g_src2 = g_proj(g_src2)
    g_src2 = torch.cat([g_src2,gdir],-1)
    g_src2 = torch.bmm(g_src2.transpose(-2, -1), g_src2)
    F_norm = torch.norm(g_src2,dim=(-2,-1),keepdim=True)+1.0
    F_norm = F_norm.contiguous().view(tgt_len,bsz,-1)
    g_src2 = g_src2.contiguous().view(tgt_len, bsz, -1)
    g_src2 = linear_g2(F.relu(linear_g1(g_src2)))
    ng_src2 = torch.cat([g_src2, ng_src],dim=-1)
    # self-attention
    # assert not use_separate_proj_weight and torch.equal(query, key) and torch.equal(key, value)
    q = q_proj(ng_src2)/F_norm 
    k = k_proj(ng_src2)/F_norm 
    v = v_proj(ng_src2)/F_norm 
    vg = vg_proj(g_src)

    # print(v.shape,vg.shape)
    # print(q.shape,k.shape,v.shape)

    q = q * scaling



    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim*2).transpose(0, 1)
    k = k.contiguous().view(tgt_len, bsz * num_heads, head_dim*2).transpose(0, 1)
    v = v.contiguous().view(tgt_len, bsz * num_heads, head_dim*2).transpose(0, 1)

    gdir = gdir.contiguous().view(tgt_len,bsz, 3,1,-1)
    gdir = gdir.repeat(1,1,1,2,1)
    vg = vg.contiguous().view(tgt_len, bsz , 3,num_heads,-1)
    vg = torch.cat([vg,gdir],-1)
    vg = vg.permute(2,1,3,0,4)
    vg = vg.contiguous().view(3*bsz*num_heads,tgt_len,head_dim*2)


    src_len = k.size(1)


    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * (num_heads), tgt_len, src_len]

    if attn_mask is not None:
        assert attn_mask.shape == attn_output_weights.shape
        attn_output_weights = attn_output_weights + attn_mask


    attn_output_weights = F.softmax(attn_output_weights, dim=-1)

    attn_output = torch.bmm(attn_output_weights, v) #bszhead*node*node   bszhead*node*head_dim  = bszhead *node *head_dim
    assert list(attn_output.size()) == [bsz * (num_heads), tgt_len, 2*head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, 2*embed_dim)
    ng_src = ng_out(attn_output)


    attn_output_weights = attn_output_weights.repeat(3,1,1)
    # print(attn_output_weights.shape)
    attn_output = torch.bmm(attn_output_weights, vg)#3bszhead*node*node    3bszhead*node*head_dim
    # print("attn_output",attn_output.shape) # 3bszhead*node*head_dim
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, 3, bsz,  -1) 
    attn_output = attn_output.transpose(1, 2)
    # print("attn_output",attn_output.shape)
    g_src = g_out(attn_output)

    
    return g_src, ng_src


