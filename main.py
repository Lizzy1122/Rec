import os
import random

import torch
import numpy as np

from time import time
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path
from prettytable import PrettyTable

from common.test import test_v2
from common.utils import early_stopping, print_dict
from common.config import parse_args
from common.dataset import CKGData
from common.dataset.build import build_loader

from modules.sampler import KGPolicy
from modules.recommender import MF

from sklearn.metrics.pairwise import cosine_similarity


def train_one_epoch(
    recommender,
    sampler,
    train_loader,
    recommender_optim,
    sampler_optim,
    adj_matrix,
    edge_matrix,
    train_data,
    cur_epoch,
    avg_reward,
):

    loss, base_loss, reg_loss = 0, 0, 0
    epoch_reward = 0

    """Train one epoch"""
    tbar = tqdm(train_loader, ascii=True)
    num_batch = len(train_loader)
    print("num_batch:")
    print(num_batch)  #
    for batch_data in tbar:

        tbar.set_description("Epoch {}".format(cur_epoch))

        if torch.cuda.is_available():
            batch_data = {k: v.cuda(non_blocking=True) for k, v in batch_data.items()}

        """Train recommender using negtive item provided by sampler"""
        recommender_optim.zero_grad()  # 优化器梯度清0

        neg = batch_data["neg_i_id"]
        pos = batch_data["pos_i_id"]
        users = batch_data["u_id"]
        # print(batch_data)
        # print("neg:")
        # print(neg)
        # print("pos:")
        # print(pos)
        # print("users:")
        # print(users)

        selected_neg_items_list, _ = sampler(batch_data, adj_matrix, edge_matrix)
        selected_neg_items = selected_neg_items_list[-1, :]

        train_set = train_data[users]
        in_train = torch.sum(
            selected_neg_items.unsqueeze(1) == train_set.long(), dim=1
        ).bool()  # 修改了这里，把.byte()改成.bool()    # 每个选定的负样本是否在对应用户的训练集中？
        selected_neg_items[in_train] = neg[in_train].long()  # 添加了.long()

        base_loss_batch, reg_loss_batch = recommender(users, pos, selected_neg_items)
        loss_batch = base_loss_batch + reg_loss_batch

        loss_batch.backward()
        recommender_optim.step()

        """Train sampler network"""
        sampler_optim.zero_grad()
        selected_neg_items_list, selected_neg_prob_list = sampler(
            batch_data, adj_matrix, edge_matrix
        )

        with torch.no_grad():
            reward_batch = recommender.get_reward(users, pos, selected_neg_items_list)

        epoch_reward += torch.sum(reward_batch)
        reward_batch -= avg_reward

        batch_size = reward_batch.size(1)
        n = reward_batch.size(0) - 1
        R = torch.zeros(batch_size, device=reward_batch.device)
        reward = torch.zeros(reward_batch.size(), device=reward_batch.device)

        gamma = args_config.gamma

        for i, r in enumerate(reward_batch.flip(0)):
            R = r + gamma * R
            reward[n - i] = R

        reinforce_loss = -1 * torch.sum(reward_batch * selected_neg_prob_list)
        reinforce_loss.backward()
        sampler_optim.step()

        """record loss in an epoch"""
        loss += loss_batch
        reg_loss += reg_loss_batch
        base_loss += base_loss_batch

    avg_reward = epoch_reward / num_batch
    train_res = PrettyTable()
    train_res.field_names = ["Epoch", "Loss", "BPR-Loss", "Regulation", "AVG-Reward"]
    train_res.add_row(
        [cur_epoch, loss.item(), base_loss.item(), reg_loss.item(), avg_reward.item()]
    )
    print(train_res)

    return loss, base_loss, reg_loss, avg_reward


def save_model(file_name, model, config):
    if not os.path.isdir(config.out_dir):
        os.mkdir(config.out_dir)

    model_file = Path(config.out_dir + file_name)
    model_file.touch(exist_ok=True)

    print("Saving model...")
    torch.save(model.state_dict(), model_file)


def build_sampler_graph(n_nodes, edge_threshold, graph):
    adj_matrix = torch.zeros(n_nodes, edge_threshold * 2)
    edge_matrix = torch.zeros(n_nodes, edge_threshold)

    """sample neighbors for each node"""
    for node in tqdm(graph.nodes, ascii=True, desc="Build sampler matrix"):
        neighbors = list(graph.neighbors(node))
        if len(neighbors) >= edge_threshold:
            sampled_edge = random.sample(neighbors, edge_threshold)
            edges = deepcopy(sampled_edge)
        else:
            neg_id = random.sample(
                range(CKG.item_range[0], CKG.item_range[1] + 1),
                edge_threshold - len(neighbors),
            )
            node_id = [node] * (edge_threshold - len(neighbors))
            sampled_edge = neighbors + neg_id
            edges = neighbors + node_id

        """concatenate sampled edge with random edge"""
        sampled_edge += random.sample(
            range(CKG.item_range[0], CKG.item_range[1] + 1), edge_threshold
        )

        adj_matrix[node] = torch.tensor(sampled_edge, dtype=torch.long)
        edge_matrix[node] = torch.tensor(edges, dtype=torch.long)

    if torch.cuda.is_available():
        adj_matrix = adj_matrix.cuda().long()
        edge_matrix = edge_matrix.cuda().long()

    return adj_matrix, edge_matrix


def build_train_data(train_mat):
    num_user = max(train_mat.keys()) + 1
    num_true = max([len(i) for i in train_mat.values()])

    train_data = torch.zeros(num_user, num_true)

    for i in train_mat.keys():
        true_list = train_mat[i]
        true_list += [-1] * (num_true - len(true_list))
        train_data[i] = torch.tensor(true_list, dtype=torch.long)

    return train_data


# def calculate_item_popularity(interactions):
#     item_popularity = {}
#     for user, items in interactions.items():
#         for item in items:
#             if item not in item_popularity:
#                 item_popularity[item] = 0
#             item_popularity[item] += 1
#     # 归一化流行度
#     max_popularity = max(item_popularity.values())
#     for item in item_popularity:
#         item_popularity[item] /= max_popularity
#     return item_popularity
#
#
# def calculate_item_similarity(interactions, n_items):
#     # 创建项目共现矩阵
#     cooccurrence_matrix = np.zeros((n_items, n_items))
#     for items in interactions.values():
#         for i in items:
#             for j in items:
#                 if i != j:
#                     cooccurrence_matrix[i, j] += 1
#                     cooccurrence_matrix[j, i] += 1  # 因为是无向的
#
#     # 计算余弦相似度
#     item_similarity = cosine_similarity(cooccurrence_matrix)
#     return item_similarity


# 计算 item_popularity
def calculate_item_popularity(train_user_dict, n_items):
    item_popularity = {i: 0 for i in range(n_items)}
    for user, pos_items in train_user_dict.items():
        for item in pos_items:
            # 跳过超出范围的项目 ID
            if item < 0 or item >= n_items:
                continue
            item_popularity[item] += 1
    # 计算流行度的反数，以便用于新颖性计算
    total_items = sum(item_popularity.values())
    item_popularity = {item: popularity / total_items for item, popularity in item_popularity.items()}
    return item_popularity


# 计算 item_similarity
def calculate_item_similarity(model, n_users, n_items):
    # 获取项目嵌入向量
    u_e, i_e = torch.split(model.all_embed, [n_users, n_items])
    i_e = i_e.cpu().detach().numpy()  # 使用 .detach() 方法将张量与计算图分离

    item_similarity = {}
    for i in range(n_items):
        for j in range(i + 1, n_items):
            # 计算余弦相似度
            dot_product = np.dot(i_e[i], i_e[j])
            norm_i = np.linalg.norm(i_e[i])
            norm_j = np.linalg.norm(i_e[j])
            cosine_similarity = dot_product / (norm_i * norm_j)
            # 存储项目相似度
            item_similarity[(i, j)] = cosine_similarity
            item_similarity[(j, i)] = cosine_similarity

    return item_similarity


def train(train_loader, test_loader, graph, data_config, args_config):
    """build padded training set"""
    train_mat = graph.train_user_dict
    # print(train_mat) # 带-1填充的字典，可是这个时候还没填充呢，是哪里设置的？回去看看
    train_data = build_train_data(train_mat)

    # 加载预训练模型
    if args_config.pretrain_r:
        print(
            "\nLoad model from {}".format(
                args_config.data_path + args_config.model_path
            )
        )
        paras = torch.load(args_config.data_path + args_config.model_path)
        all_embed = torch.cat((paras["user_para"], paras["item_para"]))
        data_config["all_embed"] = all_embed

    recommender = MF(data_config=data_config, args_config=args_config)
    sampler = KGPolicy(recommender, data_config, args_config)

    if torch.cuda.is_available():
        train_data = train_data.long().cuda()
        sampler = sampler.cuda()
        recommender = recommender.cuda()

        print("\nSet sampler as: {}".format(str(sampler)))
        print("Set recommender as: {}\n".format(str(recommender)))

    recommender_optimer = torch.optim.Adam(recommender.parameters(), lr=args_config.rlr)
    sampler_optimer = torch.optim.Adam(sampler.parameters(), lr=args_config.slr)

    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger, mmr_loger, auc_loger, novelty_loger, diversity_loger = [], [], [], [], [], [], [], [], []
    stopping_step, cur_best_pre_0, avg_reward = 0, 0.0, 0
    t0 = time()

    for epoch in range(args_config.epoch):
        if epoch % args_config.adj_epoch == 0:
            """sample adjacency matrix"""
            adj_matrix, edge_matrix = build_sampler_graph(
                data_config["n_nodes"], args_config.edge_threshold, graph.ckg_graph
            )

        cur_epoch = epoch + 1
        loss, base_loss, reg_loss, avg_reward = train_one_epoch(
            recommender,
            sampler,
            train_loader,
            recommender_optimer,
            sampler_optimer,
            adj_matrix,
            edge_matrix,
            train_data,
            cur_epoch,
            avg_reward,
        )

        n_users = graph.n_users
        n_items = graph.n_items
        item_popularity = calculate_item_popularity(train_mat, n_items)
        item_similarity = calculate_item_similarity(recommender, n_users, n_items)

        """Test"""
        if cur_epoch % args_config.show_step == 0:
            with torch.no_grad():
                ret = test_v2(recommender, args_config.Ks, graph, item_popularity=item_popularity, item_similarity=item_similarity)

            loss_loger.append(loss)
            rec_loger.append(ret["recall"])
            pre_loger.append(ret["precision"])
            ndcg_loger.append(ret["ndcg"])
            hit_loger.append(ret["hit_ratio"])
            mmr_loger.append(ret["mmr"])
            auc_loger.append(ret["auc"])
            novelty_loger.append(ret["novelty"])
            diversity_loger.append(ret["diversity"])

            print_dict(ret)

            cur_best_pre_0, stopping_step, should_stop = early_stopping(
                ret["recall"][0],
                cur_best_pre_0,
                stopping_step,
                expected_order="acc",
                flag_step=args_config.flag_step,
            )

            if should_stop:
                break

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)
    auc = np.array(auc_loger)
    novelty = np.array(novelty_loger)
    diversity = np.array(diversity_loger)

    best_rec_0 = max(recs[:, 0])
    # best_rec_0 = max(recs)
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = (
        "Best Iter=[%d]@[%.1f]\n recall=[%s] \n precision=[%s] \n hit=[%s] \n ndcg=[%s] \n auc=[%s] \n novelty=[%s] \n diversity=[%s]"
        % (
            idx,
            time() - t0,
            "\t".join(["%.5f" % r for r in recs[idx]]),
            "\t".join(["%.5f" % r for r in pres[idx]]),
            "\t".join(["%.5f" % r for r in hit[idx]]),
            "\t".join(["%.5f" % r for r in ndcgs[idx]]),
            "\t".join(["%.5f" % r for r in auc[idx]]),
            "\t".join(["%.5f" % r for r in novelty[idx]]),
            "\t".join(["%.5f" % r for r in diversity[idx]]),
        )
    )
    print(final_perf)
    save_model('recommander', recommender, args_config)
    save_model('sampler', sampler, args_config)


if __name__ == "__main__":
    """fix the random seed"""
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    """initialize args and dataset"""
    args_config = parse_args()
    CKG = CKGData(args_config)

    """set the gpu id"""
    if torch.cuda.is_available():
        torch.cuda.set_device(args_config.gpu_id)

    data_config = {
        "n_users": CKG.n_users,
        "n_items": CKG.n_items,
        "n_relations": CKG.n_relations + 2,
        "n_entities": CKG.n_entities,
        "n_nodes": CKG.entity_range[1] + 1,
        "item_range": CKG.item_range,
    }

    print("\ncopying CKG graph for data_loader.. it might take a few minutes")
    graph = deepcopy(CKG)
    train_loader, test_loader = build_loader(args_config=args_config, graph=graph)

    train(
        train_loader=train_loader,
        test_loader=test_loader,
        graph=CKG,
        data_config=data_config,
        args_config=args_config,
    )
