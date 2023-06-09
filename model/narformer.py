import torch
import torch.nn as nn
import torch.nn.functional as F
from data_process.position_encoding import get_embedder
from layers.transformer import Encoder, RegHead


# original
# def tokenizer(ops, matrix, dim_x, dim_p, embed_type):
#     # encode operation
#     fn, _ = get_embedder(dim_x, embed_type=embed_type)
#     code_ops_tmp = [fn(torch.tensor([op], dtype=torch.float32)) for op in ops]
#     # code_ops_tmp.append(fn(torch.tensor([1e5], dtype=torch.float32)))
#     code_ops = torch.stack(code_ops_tmp, dim=0)  # (len, dim_x)

#     # encode self position
#     fn, _ = get_embedder(dim_p, embed_type=embed_type)
#     code_pos_tmp = [fn(torch.tensor([i], dtype=torch.float32)) for i in range(len(ops))]
#     # code_pos_tmp.append(fn(torch.tensor([1e5], dtype=torch.float32)))
#     code_pos = torch.stack(code_pos_tmp, dim=0)  # (len, dim_p)

#     # adj = torch.tensor(matrix, dtype=torch.float32)
#     # evals, evecs = torch.linalg.eig(adj) # evecs @ (evals.reshape(-1, 1) * evecs.T) == adj
#     # code_pos[:len(ops), :2 * adj.shape[0]] = torch.cat([evecs.T.real, evecs.T.imag], dim=1)
#     # evals, evecs = torch.linalg.eig(adj.T)
#     # code_pos[:len(ops), 2 * adj.shape[0]:2 * adj.shape[0] + 2 * adj.shape[1]] = torch.cat([evecs.T.real, evecs.T.imag], dim=1)

#     # encode data source of each node
#     fn, _ = get_embedder(dim_p, embed_type=embed_type)
#     code_sour_tmp = [fn(torch.tensor([-1], dtype=torch.float32))]
#     for i in range(1, len(ops)):
#         i_sour = 0
#         for j in range(i):
#             if matrix[j][i] == 1:
#                 # i_sour += fn(torch.tensor([j], dtype=torch.float32))
#                 i_sour += code_pos[j]
#         code_sour_tmp.append(i_sour)
#     # code_sour_tmp.append(fn(torch.tensor([1e5], dtype=torch.float32)))
#     # code_sour_tmp.append(code_pos[-1])
#     code_sour = torch.stack(code_sour_tmp, dim=0)

#     code = torch.cat([code_ops, code_pos, code_sour], dim=-1)
#     return code


# Feature position embedding
def tokenizer(ops, matrix, dim_x, dim_p, embed_type):
    # encode operation
    fn, _ = get_embedder(dim_x, embed_type=embed_type)
    code_ops_tmp = [fn(torch.tensor([op], dtype=torch.float32)) for op in ops]
    code_ops = torch.stack(code_ops_tmp, dim=0)  # (len, dim_x)

    # encode operation position
    fn, _ = get_embedder(dim_p, embed_type=embed_type)
    code_pos_tmp = [fn(torch.tensor([i], dtype=torch.float32)) for i in range(len(ops))]

    # encode source and target feature position embedding of each node
    code_sour_tmp = [fn(torch.tensor([-1], dtype=torch.float32))]
    for i in range(1, len(ops)):
        i_sour = 0
        for j in range(i):
            if matrix[j][i] == 1:
                i_sour += code_pos_tmp[j]
        code_sour_tmp.append(i_sour)

    code_targ_tmp = []
    for i in range(len(ops) - 1):
        i_targ = code_sour_tmp[matrix[i].index(1)]
        code_targ_tmp.append(i_targ)
        # for j in range(i + 1, len(ops)):
        #     if matrix[i][j] == 1:
        #         code_targ_tmp.append(code_sour_tmp[j])
        #         break
        # assert code_targ_tmp[-1].allclose(i_targ)
    code_targ_tmp.append(fn(torch.tensor([1e5], dtype=torch.float32)))

    code_sour = torch.stack(code_sour_tmp, dim=0)
    code_targ = torch.stack(code_targ_tmp, dim=0)

    code = torch.cat([code_ops, code_sour, code_targ], dim=-1)
    return code


# x3
# def tokenizer(ops, matrix, dim_x, dim_p, embed_type):
#     # encode operation
#     fn, _ = get_embedder(dim_x, embed_type=embed_type)
#     code_ops_tmp = [fn(torch.tensor([op], dtype=torch.float32)) for op in ops]
#     code_ops = torch.stack(code_ops_tmp, dim=0)  # (len, dim_x)

#     # encode operation position
#     fn, _ = get_embedder(dim_p, embed_type=embed_type)
#     code_pos_tmp = [fn(torch.tensor([i], dtype=torch.float32)) for i in range(len(ops))]
#     code_pos = torch.stack(code_pos_tmp, dim=0)

#     # encode source and target feature position embedding of each node
#     code_sour_tmp = []
#     code_targ_tmp = []
#     for i in range(len(ops)):
#         i_sour = 0
#         i_targ = 0
#         for j in range(i):
#             if matrix[j][i] == 1:
#                 i_sour += code_pos_tmp[j]
#         for j in range(i + 1, len(ops)):
#             if matrix[i][j] == 1:
#                 i_targ += code_pos_tmp[j]
#         if isinstance(i_sour, int):
#             i_sour = fn(torch.tensor([-1], dtype=torch.float32))
#         if isinstance(i_targ, int):
#             i_targ = fn(torch.tensor([1e5], dtype=torch.float32))
#         code_sour_tmp.append(i_sour)
#         code_targ_tmp.append(i_targ)
#     code_sour = torch.stack(code_sour_tmp, dim=0)
#     code_targ = torch.stack(code_targ_tmp, dim=0)

#     code = torch.cat([code_ops, code_sour, code_pos, code_targ], dim=-1)
#     return code


class NetEncoder(nn.Module):
    def __init__(self, config):
        super(NetEncoder, self).__init__()
        self.config = config
        self.dim_x = self.config.multires_x
        self.dim_p = self.config.multires_p
        self.embed_type = self.config.embed_type
        # self.linear = nn.Linear((self.dim_x + self.dim_p) * 4, config.graph_d_model)
        self.linear = nn.Linear(config.graph_d_model, config.graph_d_model)
        self.transformer = Encoder(config)
        self.mlp = RegHead(config)
        if config.use_extra_token:
            self.dep_map = nn.Linear(config.graph_d_model, config.graph_d_model)
            # self.dep_map = nn.Linear((self.dim_x + self.dim_p) * 4, (self.dim_x + self.dim_p) * 4)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        return
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, X, adjs, embeddings, static_feats):
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
                    self.config.graph_d_model // 2,
                    self.embed_type,
                )
                code_depth = F.relu(self.dep_map(depth_fn(depth)))  # (b, 1, d)
                seqcode = torch.cat([seqcode, code_depth], dim=1)
            elif "nnlqp" in self.config.dataset.lower():
                code_rest, code_depth = torch.split(
                    seqcode, [seqcode.shape[1] - 1, 1], dim=1
                )
                code_depth = F.relu(self.dep_map(code_depth))
                seqcode = torch.cat([code_rest, code_depth], dim=1)

        in_degree, out_degree = X
        dtype, device = in_degree.dtype, in_degree.device
        padding = torch.ones((adjs.shape[0], 1), dtype=dtype, device=device) * 9
        in_degree = torch.cat([in_degree, padding], dim=1)
        out_degree = torch.cat([out_degree, padding], dim=1)
        seqcode = self.linear(seqcode)
        aev = self.transformer(seqcode, [adjs, in_degree, out_degree])
        # multi_stage:aev(b, 1, d)
        predict = self.mlp(aev, static_feats)
        return predict
