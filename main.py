# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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
from scipy.sparse import lil_matrix
import networkx as nx


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

# 用最大值归一化，并没有处理重映射问题，舍弃
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


# 计算 item_popularity
# 注意：train_user_dict进行了重映射， item id from [0, #items) to [#users, #users + #items)
def calculate_item_popularity(train_user_dict, n_users, n_items):
    # 初始化字典来存储每个物品的交互次数
    item_popularity = {i: 0 for i in range(n_items)}
    # 遍历训练集中的每个用户以及他们的交互物品列表
    for user, pos_items in train_user_dict.items(): # 取每行字典
        # 对用户交互的每个物品，增加其交互次数
        for item in pos_items:
            # 跳过超出范围的项目 ID
            if item - n_users < 0 or item - n_users >= n_items:
                # print(f"用户 {user} 的交互列表中存在超出范围的 item ID: {item}")
                continue
            item_popularity[item - n_users] += 1


    total_items = sum(item_popularity.values())
    # 如果总交互次数为零，打印警告信息并返回初始流行度字典
    # if total_items == 0:
    #     print("警告: 没有物品的交互记录或所有物品的 ID 超出了范围。")
    #     return item_popularity

    # 归一化
    item_popularity = {item: popularity / total_items for item, popularity in item_popularity.items()}

    return item_popularity


# 计算 item_similarity  时间太长，资源不够
# def calculate_item_similarity(model, n_users, n_items):
#     # 获取项目嵌入向量
#     u_e, i_e = torch.split(model.all_embed, [n_users, n_items])
#     i_e = i_e.cpu().detach().numpy()  # 使用 .detach() 方法将张量与计算图分离
#
#     item_similarity = {}
#     for i in range(n_items):
#         for j in range(i + 1, n_items):
#             # 计算余弦相似度
#             dot_product = np.dot(i_e[i], i_e[j])
#             norm_i = np.linalg.norm(i_e[i])
#             norm_j = np.linalg.norm(i_e[j])
#             cosine_similarity = dot_product / (norm_i * norm_j)
#             # 存储项目相似度
#             item_similarity[(i, j)] = cosine_similarity
#             item_similarity[(j, i)] = cosine_similarity
#
#     return item_similarity

# def calculate_item_similarity(recommender, n_users, n_items):
#     # 从模型中获取用户和物品的嵌入
#     u_e, i_e = np.split(recommender.all_embed.detach().cpu().numpy(), [n_users])
#
#     # 使用 cosine_similarity 函数计算物品之间的相似性
#     item_similarity = cosine_similarity(i_e)
#
#     # 返回计算得到的物品相似性矩阵
#     return item_similarity

# 需要分配的数组资源太大，需要重新写
# def calculate_item_similarity(train_user_dict, n_items):
#     # Initialize item-item similarity matrix
#     item_similarity = np.zeros((n_items, n_items))
#
#     # Calculate item-item similarity
#     for item1 in range(n_items):
#         for item2 in range(n_items):
#             if item1 != item2:  # Exclude self-similarity
#                 # Compute similarity between item1 and item2 based on their co-occurrence in user interactions
#                 common_users = set(train_user_dict.get(item1, [])) & set(train_user_dict.get(item2, []))
#                 similarity = len(common_users) / np.sqrt(len(train_user_dict[item1]) * len(train_user_dict[item2]))
#                 item_similarity[item1, item2] = similarity
#                 # Check if both items have interactions
#                 if item1 in train_user_dict and item2 in train_user_dict:
#                     # Compute similarity between item1 and item2 based on their co-occurrence in user interactions
#                     common_users = set(train_user_dict[item1]) & set(train_user_dict[item2])
#                     similarity = len(common_users) / np.sqrt(len(train_user_dict[item1]) * len(train_user_dict[item2]))
#                 else:
#                     similarity = 0  # Set similarity to 0 if one or both items have no interactions
#                 item_similarity[item1, item2] = similarity
#
#     return item_similarity

# 利用稀疏矩阵，还是太复杂了，所需时间太长
# def calculate_item_similarity(train_user_dict, n_items, threshold=0.1):
#     # Initialize item-item similarity matrix as a LIL sparse matrix
#     item_similarity = lil_matrix((n_items, n_items), dtype=np.float32)
#
#     # Calculate item-item similarity
#     for item1 in range(n_items):
#         for item2 in range(item1 + 1, n_items):
#             if item1 in train_user_dict and item2 in train_user_dict:
#                 common_users = set(train_user_dict[item1]) & set(train_user_dict[item2])
#                 if common_users:
#                     similarity = len(common_users) / np.sqrt(len(train_user_dict[item1]) * len(train_user_dict[item2]))
#                     if similarity > threshold:
#                         item_similarity[item1, item2] = similarity
#                         item_similarity[item2, item1] = similarity
#
#     return item_similarity.tocsr()

# 无法对多重图使用
# def calculate_item_similarity(G):
#     # 使用 Jaccard 系数计算相似度
#     preds = nx.jaccard_coefficient(G)
#     similarity = {}
#     for u, v, p in preds:
#         if p > 0:  # 可以设置阈值过滤
#             similarity[(u, v)] = p
#     return similarity

# 多重图转简单图
def convert_multigraph_to_simple_graph(multigraph):
    simple_graph = nx.Graph()
    for u, v, data in multigraph.edges(data=True):
        # 每个节点对只添加一次边，无视多重边
        if not simple_graph.has_edge(u, v):
            simple_graph.add_edge(u, v)
    return simple_graph

# 计算杰卡德相似度
def calculate_jaccard_similarity(multigraph):
    # 转换为简单图
    simple_graph = convert_multigraph_to_simple_graph(multigraph)
    # 初始化进度条
    edges = list(simple_graph.edges())
    pbar = tqdm(total=len(edges), desc="Calculating Jaccard Similarity")

    # 计算Jaccard相似度
    similarities = []
    for u, v in edges:
        union_size = len(set(simple_graph.neighbors(u)) | set(simple_graph.neighbors(v)))
        intersection_size = len(set(simple_graph.neighbors(u)) & set(simple_graph.neighbors(v)))
        if union_size > 0:
            similarity = intersection_size / union_size
            similarities.append((u, v, similarity))
        pbar.update(1)
    pbar.close()
    return similarities


# 利用矩阵分解MF来计算相似度
def calculate_item_similarity(recommender, n_users, n_items):
    # 从模型中获取用户和物品的嵌入
    # _, i_e = torch.split(recommender.all_embed.detach().cpu().numpy(), [n_users])
    u_e, i_e = torch.split(recommender.all_embed, [n_users, n_items])

    # 将物品嵌入转换为 NumPy 数组
    i_e = i_e.detach().cpu().numpy()

    # 使用 cosine_similarity 函数计算物品之间的相似性
    item_similarity = cosine_similarity(i_e)

    # 返回计算得到的物品相似性矩阵
    return item_similarity


def train(train_loader, test_loader, graph, data_config, args_config):
    """build padded training set"""
    train_mat = graph.train_user_dict
    # print(train_mat) # 带-1填充的字典，可是这个时候还没填充呢，是哪里设置的？回去检查
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

    n_users = graph.n_users
    n_items = graph.n_items
    print("计算流行度...")
    item_popularity = calculate_item_popularity(train_mat, n_users, n_items)
    print("计算相似度...")
    # item_similarity = calculate_item_similarity(train_user_dict=train_mat, n_items=n_items)  #使用相似矩阵计算
    item_similarity = calculate_item_similarity(recommender, n_users, n_items)  # 使用MF计算
    # item_similarity = calculate_item_similarity(graph.ckg_graph)  # 使用图计算
    # 暂不处理
    # item_similarity = calculate_jaccard_similarity(graph.ckg_graph)
    # item_similarity_dict = {(u, v): s for u, v, s in item_similarity}


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

        """Test"""
        if cur_epoch % args_config.show_step == 0:
            with torch.no_grad():
                # ret = test_v2(recommender, args_config.Ks, graph, item_popularity=item_popularity)
                # 暂不处理
                # ret = test_v2(recommender, args_config.Ks, graph, item_popularity=item_popularity, item_similarity=item_similarity_dict)
                ret = test_v2(recommender, args_config.Ks, graph, item_popularity=item_popularity, item_similarity=item_similarity)

            loss_loger.append(loss)
            rec_loger.append(ret["recall"])
            pre_loger.append(ret["precision"])
            ndcg_loger.append(ret["ndcg"])
            hit_loger.append(ret["hit_ratio"])
            mmr_loger.append(ret["mmr"])
            auc_loger.append(ret["auc"])
            novelty_loger.append(ret["novelty"])
            # 暂不处理
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
    # 暂不处理


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
            # "\t".join(["%.5f" % r for r in novelty[idx]]),
    # 暂不处理
            "\t".join(["%.5f" % r for r in diversity[idx]]),
        )
    )
    print("\n" + final_perf)
    save_model('recommender_bridge', recommender, args_config)
    save_model('sampler_bridge', sampler, args_config)


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
