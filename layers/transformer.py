import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple


def attention(query, key, value, dropout=None):
    d_k = query.size(-1)
    # (b, n_head, l_q, d_per_head) * (b, n_head, d_per_head, l_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        attn = dropout(attn)  # (b, n_head, l_q, l_k)
    return torch.matmul(attn, value)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout, q_learnable):
        super(MultiHeadAttention, self).__init__()
        self.q_learnable = q_learnable
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head  # default: 32
        self.linears = nn.ModuleList(
            [
                nn.Linear(self.d_model, self.d_model),
                nn.Linear(self.d_model, self.d_model),
            ]
        )
        if self.q_learnable:
            self.linears.append(nn.Identity())
        else:
            self.linears.append(nn.Linear(self.d_model, self.d_model))

        self.proj = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        batch_size = query.size(0)

        key, value, query = [
            l(x).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (key, value, query))
        ]
        x = attention(query, key, value, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head * self.d_k)
        return self.proj(x)


# Different Attention Blocks, All Based on MultiHeadAttention
class SelfAttentionBlock(nn.Module):
    def __init__(self, config):
        super(SelfAttentionBlock, self).__init__()
        self.norm = nn.LayerNorm(config.graph_d_model)
        self.attn = MultiHeadAttention(
            config.graph_d_model, config.graph_n_head, config.dropout, q_learnable=False
        )
        self.dropout = nn.Dropout(p=config.dropout)
        self.drop_path = (
            DropPath(config.drop_path_rate)
            if config.drop_path_rate > 0.0
            else nn.Identity()
        )

    def forward(self, x):
        x_ = self.norm(x)
        x_ = self.attn(x_, x_, x_)
        return x + self.drop_path(x_)  # L_v4.2.2


class CrossAttentionBlock(nn.Module):
    def __init__(self, config):
        super(CrossAttentionBlock, self).__init__()
        self.norm = nn.LayerNorm(config.graph_d_model)
        self.attn = MultiHeadAttention(
            config.graph_d_model, config.graph_n_head, config.dropout, q_learnable=True
        )
        self.dropout = nn.Dropout(p=config.dropout)
        self.drop_path = (
            DropPath(config.drop_path_rate)
            if config.drop_path_rate > 0.0
            else nn.Identity()
        )

    def forward(self, x, learnt_q):
        x_ = self.norm(x)
        x_ = self.attn(learnt_q, x_, x_)
        # In multi_stage' attention, no residual connection is used because of the change in output shape
        return self.drop_path(x_)


# Blocks Used in Encoder
class FuseFeatureBlock(nn.Module):
    def __init__(self, config):
        super(FuseFeatureBlock, self).__init__()
        self.norm_kv = nn.LayerNorm(config.graph_d_model)
        self.norm_q = nn.LayerNorm(config.graph_d_model)
        self.fuse_attn = MultiHeadAttention(
            config.graph_d_model, config.graph_n_head, config.dropout, q_learnable=False
        )
        self.feed_forward = FeedForwardBlock(config)

    def forward(self, memory, q):
        x_ = self.norm_kv(memory)
        q_ = self.norm_q(q)
        x = self.fuse_attn(q_, x_, x_)
        x = self.feed_forward(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4,
        out_features=None,
        act_layer="relu",
        drop=0.0,
        bias=True,
    ):
        super(Mlp, self).__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        if act_layer.lower() == "relu":
            self.act = nn.ReLU()
        elif act_layer.lower() == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {act_layer}")
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class FeedForwardBlock(nn.Module):
    def __init__(self, config):
        super(FeedForwardBlock, self).__init__()
        dim = config.graph_d_model
        mlp_ratio = config.graph_d_ff / dim
        act_layer = config.act_function
        drop = config.dropout
        drop_path = config.drop_path_rate

        self.norm = nn.LayerNorm(dim)
        self.feed_forward = Mlp(dim, mlp_ratio, act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        return x + self.drop_path(self.feed_forward(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self, config):
        super(EncoderBlock, self).__init__()
        self.self_attn = SelfAttentionBlock(config)
        self.feed_forward = FeedForwardBlock(config)

    def forward(self, x):
        x = self.self_attn(x)
        x = self.feed_forward(x)
        return x


class FuseStageBlock(nn.Module):
    def __init__(self, config, stg_id, dp_rates):
        super(FuseStageBlock, self).__init__()
        self.n_self_attn = config.depths[stg_id] - 1
        self.self_attns = nn.ModuleList()
        self.feed_forwards = nn.ModuleList()
        for i, r in enumerate(dp_rates):
            config.drop_path_rate = r
            self.feed_forwards.append(FeedForwardBlock(config))
            if i == 0:
                self.cross_attn = CrossAttentionBlock(config)
            else:
                self.self_attns.append(SelfAttentionBlock(config))

    def forward(self, kv, q):
        x = self.cross_attn(kv, q)
        x = self.feed_forwards[0](x)
        for i in range(self.n_self_attn):
            x = self.self_attns[i](x)
            x = self.feed_forwards[i + 1](x)
        return x


# Main class
class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.num_stage = len(config.depths)
        self.num_layers = sum(config.depths)
        self.norm = nn.LayerNorm(config.graph_d_model)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, config.drop_path_rate, self.num_layers)
        ]

        # 1st stage: Encoder
        self.layers = nn.ModuleList()
        for i in range(config.depths[0]):
            config.drop_path_rate = dpr[i]
            self.layers.append(EncoderBlock(config))

        if self.num_stage > 1:
            # Rest stage: information fusion
            self.fuseUnit = nn.ModuleList()
            self.fuseStages = nn.ModuleList()
            self.fuseStages.append(
                FuseStageBlock(
                    config,
                    stg_id=1,
                    dp_rates=dpr[sum(config.depths[:1]) : sum(config.depths[:2])],
                )
            )
            for i in range(2, self.num_stage):
                self.fuseUnit.append(FuseFeatureBlock(config))
                self.fuseStages.append(
                    FuseStageBlock(
                        config,
                        stg_id=i,
                        dp_rates=dpr[
                            sum(config.depths[:i]) : sum(config.depths[: i + 1])
                        ],
                    )
                )

            self.learnt_q = nn.ParameterList(
                [
                    nn.Parameter(torch.randn(1, 2 ** (3 - s), config.graph_d_model))
                    for s in range(1, self.num_stage)
                ]
            )

            num_in_degree = 10
            num_out_degree = 10
            self.in_degree_encoder = nn.Embedding(num_in_degree, config.graph_d_model)
            self.out_degree_encoder = nn.Embedding(num_out_degree, config.graph_d_model)

    def forward(self, x, graph_indices):
        B, _, _ = x.shape

        adjs, in_degree, out_degree = graph_indices
        x = x + self.in_degree_encoder(in_degree) + self.out_degree_encoder(out_degree)

        # 1st stage: Encoder
        for i, layer in enumerate(self.layers):
            x = layer(x)  # EncoderBlock()
        x_ = x
        # Rest stage: information fusion
        if self.num_stage > 1:
            memory = x
            q = self.fuseStages[0](
                memory, self.learnt_q[0].repeat(B, 1, 1, 1)
            )  # q(b,4,d)
            for i in range(self.num_stage - 2):
                kv = self.fuseUnit[i](memory, q)
                q = self.fuseStages[i + 1](
                    kv, self.learnt_q[i + 1].repeat(B, 1, 1, 1)
                )  # q(b,2,d), q(b,1,d)
            x_ = q
        output = self.norm(x_)
        return output


class RegHead(nn.Module):
    def __init__(self, config):
        super(RegHead, self).__init__()
        self.config = config
        if self.config.avg_tokens:
            self.pool = nn.AdaptiveAvgPool1d(1)
        self.layer = nn.Linear(config.d_model, 1)
        self.dataset = config.dataset
        if self.dataset == "nnlqp":
            mlp_hiddens = [config.d_model // (2 ** i) for i in range(4)]
            self.mlp = []
            dim = config.d_model
            for hidden_size in mlp_hiddens:
                self.mlp.append(
                    nn.Sequential(
                        nn.Linear(dim, hidden_size),
                        nn.ReLU(inplace=False),
                        nn.Dropout(p=config.dropout),
                    )
                )
                dim = hidden_size
            self.mlp.append(nn.Linear(dim, 1))
            self.mlp = nn.Sequential(*self.mlp)

    def forward(self, x, sf):  # x(b/n_gpu, l, d)
        if self.config.avg_tokens:
            x_ = self.pool(x.permute(0, 2, 1)).permute(0, 2, 1).squeeze(1)
        else:
            x_ = x[:, -1, :]  # (b,d)

        if self.dataset == "nnlqp":
            x_ = torch.cat([x_, sf], dim=-1)

        res = self.mlp(x_) if self.dataset == "nnlqp" else torch.sigmoid(self.layer(x_))
        return res
