import torch
import torch.nn as nn
import torch.nn.functional as F
from data_process.position_encoding import get_embedder
from layers.transformer import Encoder, RegHead


def tokenizer(ops, matrix, dim_x, dim_p, embed_type):
    # encode operation
    fn, _ = get_embedder(dim_x, embed_type=embed_type)
    code_ops_tmp = [fn(torch.tensor([-1], dtype=torch.float32))]
    code_ops_tmp += [fn(torch.tensor([op], dtype=torch.float32)) for op in ops]
    code_ops_tmp.append(fn(torch.tensor([1e5], dtype=torch.float32)))
    # code_ops = torch.stack(code_ops_tmp, dim=0)  # (len, dim_x)

    # encode self position
    fn, _ = get_embedder(dim_p, embed_type=embed_type)
    code_pos_tmp = [fn(torch.tensor([-1], dtype=torch.float32))]
    code_pos_tmp += [
        fn(torch.tensor([i], dtype=torch.float32)) for i in range(len(ops))
    ]
    code_pos_tmp.append(fn(torch.tensor([1e5], dtype=torch.float32)))
    # code_pos = torch.stack(code_pos_tmp, dim=0)  # (len, dim_p)

    adj = torch.tensor(matrix)
    start_idx = torch.argmin(adj.sum(0)) + 1
    end_idx = torch.argmin(adj.sum(1)) + 1

    code = [
        torch.cat(
            [
                code_ops_tmp[0],
                code_pos_tmp[0],
                code_ops_tmp[start_idx],
                code_pos_tmp[start_idx],
            ],
            dim=0,
        )
    ]

    for i in range(1, len(ops)):
        for j in range(i):
            if matrix[j][i] == 1:
                code.append(
                    torch.cat(
                        [
                            code_ops_tmp[j],
                            code_pos_tmp[j],
                            code_ops_tmp[i],
                            code_pos_tmp[i],
                        ],
                        dim=0,
                    )
                )
    code.append(
        torch.cat(
            [code_ops_tmp[end_idx], code_pos_tmp[end_idx], code_ops_tmp[-1], code_pos_tmp[-1],],
            dim=0,
        )
    )

    code = torch.stack(code)
    return code


class NetEncoder(nn.Module):
    def __init__(self, config):
        super(NetEncoder, self).__init__()
        self.config = config
        self.dim_x = self.config.multires_x
        self.dim_p = self.config.multires_p
        self.embed_type = self.config.embed_type
        # self.linear = nn.Linear((self.dim_x + self.dim_p) * 4, config.graph_d_model)
        # self.norm = nn.LayerNorm(config.graph_d_model)
        self.transformer = Encoder(config)
        self.mlp = RegHead(config)
        if config.use_extra_token:
            self.dep_map = nn.Linear(config.graph_d_model, config.graph_d_model)
            # self.dep_map = nn.Linear((self.dim_x + self.dim_p) * 4, (self.dim_x + self.dim_p) * 4)

    def forward(self, X, R, embeddings, static_feats):
        # Get embedding
        seqcode = embeddings  # (b, l+1(end), d)

        # Depth token
        if self.config.use_extra_token:
            if "nasbench" in self.config.dataset.lower():
                # depth = L
                depth = torch.full(
                    (seqcode.shape[0], 1, 1), fill_value=seqcode.shape[1] - 1
                ).to(seqcode.device)
                depth_fn, _ = get_embedder(
                    # (self.dim_x + self.dim_p) * 2, self.embed_type
                    self.config.graph_d_model // 2, self.embed_type
                )
                code_depth = F.relu(self.dep_map(depth_fn(depth)))  # (b, 1, d)
                seqcode = torch.cat([seqcode, code_depth], dim=1)
            elif "nnlqp" in self.config.dataset.lower():
                code_rest, code_depth = torch.split(
                    seqcode, [seqcode.shape[1] - 1, 1], dim=1
                )
                code_depth = F.relu(self.dep_map(code_depth))
                seqcode = torch.cat([code_rest, code_depth], dim=1)

        # seqcode = self.linear(seqcode)
        # seqcode = self.norm(seqcode)
        aev = self.transformer(seqcode)  # multi_stage:aev(b, 1, d)
        predict = self.mlp(aev, static_feats)
        return predict
