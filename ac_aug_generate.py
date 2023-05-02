import os
import argparse
import itertools
from tqdm import tqdm, trange

import numpy as np
import torch
from model.narformer import tokenizer


def args_loader():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="nasbench101", help="nasbench101/nasbench301/oo"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../all.pt",
        help="train data needed to be augmented",
    )
    parser.add_argument(
        "--n_percent", type=float, default=8000, help="train proportion or numbers"
    )
    parser.add_argument(
        "--multires_x", type=int, default=32, help="dim of operation encoding"
    )
    parser.add_argument(
        "--multires_p", type=int, default=32, help="dim of position encoding"
    )
    parser.add_argument(
        "--embed_type",
        type=str,
        default="nerf",
        help="Type of position embedding: nerf|trans",
    )
    args = parser.parse_args()
    return args


def upper_tri_matrix(matrix):
    flag = True
    for i in range(len(matrix)):
        for j in range(0, i):
            if matrix[i][j] != 0:
                flag = False
                break
    return flag


def ac_aug_generate(adj, ops):
    num_vertices = len(ops)
    ordered_index = list(range(1, num_vertices - 1))
    all_perms = itertools.permutations(ordered_index)
    auged_adjs = [adj]
    auged_opss = [ops]

    for id, label in enumerate(all_perms):
        if id == 2:  # 这里是不是不太合理？意味着所有sample至多只有一个augment
            break
        label = [0] + list(label) + [num_vertices - 1]
        P = np.zeros((num_vertices, num_vertices))
        for i, j in enumerate(label):
            P[i, j] = 1
        P_inv = np.linalg.inv(P)
        adj_aug = (P @ adj @ P_inv).astype(int).tolist()
        ops_aug = (ops @ P_inv).astype(int).tolist()
        if (
            (adj_aug not in auged_adjs) or (ops_aug not in auged_opss)
        ) and upper_tri_matrix(adj_aug):
            auged_adjs.append(adj_aug)
            auged_opss.append(ops_aug)
    # print('Original:')
    # print(adj, ops)
    # print('Augmented arch:')
    # rint(auged_adjs[1:], auged_opss[1:], '\n\n')
    return auged_adjs[1:], auged_opss[1:]


if __name__ == "__main__":
    args = args_loader()
    save_dir = args.data_path.split()[0]

    dx, dp = args.multires_x, args.multires_p

    train_data = torch.load(args.data_path)

    data_num = (
        int(args.n_percent)
        if args.n_percent > 1
        else int(len(train_data) * args.n_percent)
    )
    auged_data = {}
    for key in trange(data_num):
        ops = train_data[key]["ops"]
        adj = train_data[key]["adj"]
        auged_adjs, auged_opss = ac_aug_generate(adj, ops)
        netcodes = [
            tokenizer(auged_ops, auged_adj, dx, dp, args.embed_type)
            for auged_ops, auged_adj in zip(auged_opss, auged_adjs)
        ]
        if len(auged_opss) > 0:
            auged_data[key] = {}
            if args.dataset == "nasbench101":
                for i in range(len(auged_opss)):
                    auged_data[key][i + 1] = {  # auged data's key : 1 ~ num_auged
                        "index": i,
                        "adj": auged_adjs[i],
                        "ops": auged_opss[i],
                        "netcode": netcodes[i],
                        "params": train_data[key]["params"],
                        "training_time": train_data[key]["training_time"],
                        "validation_accuracy": train_data[key]["validation_accuracy"],
                        "test_accuracy": train_data[key]["test_accuracy"],
                    }
            elif args.dataset == "nasbench201":
                for i in range(len(auged_opss)):
                    auged_data[key][i + 1] = {
                        "index": i,
                        "adj": auged_adjs[i],
                        "ops": auged_opss[i],
                        "training_time": train_data[key]["training_time"],
                        "test_accuracy": train_data[key]["test_accuracy"],
                        "test_accuracy_avg": train_data[key]["test_accuracy_avg"],
                        "valid_accuracy": train_data[key]["valid_accuracy"],
                        "valid_accuracy_avg": train_data[key]["valid_accuracy_avg"],
                        "netcode": netcodes[i],
                    }
    torch.save(auged_data, os.path.join(save_dir, f"OurAug.pt"))
