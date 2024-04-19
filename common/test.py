import torch

import numpy as np
from tqdm import tqdm


def get_score(model, n_users, n_items, train_user_dict, s, t):
    # print("n_users:" + str(n_users))
    # print("n_items:" + str(n_items))
    # print("all_embed.size:" + str(model.all_embed.size(0)))

    u_e, i_e = torch.split(model.all_embed, [n_users, n_items])

    # 选择当前批次的用户嵌入
    u_e = u_e[s:t, :]

    # 计算用户和物品的得分矩阵
    score_matrix = torch.matmul(u_e, i_e.t())
    # print(train_user_dict)

    # 调整得分矩阵中的训练集物品
    for u in range(s, t):
        # pos = train_user_dict[u]
        # # print(f"User {u} positive samples: {pos}")
        # idx = pos.index(-1) if -1 in pos else len(pos)
        # score_matrix[u-s][pos[:idx] - n_users] = -1e5

        # 获取当前用户的正样本列表
        pos = train_user_dict.get(u, [])
        if not pos:  # 如果正样本物品列表为空，则跳过当前用户的处理，进入下一个用户的处理
            # print(f"User {u} has no positive samples.")
            continue
        # else:  # 如果正样本物品列表不为空，则打印正样本列表
        #     print(f"User {u} positive samples: {pos}")
        # print(f"User {u} positive samples: {pos}")

        idx = pos.index(-1) if -1 in pos else len(pos)
        score_matrix[u-s][pos[:idx] - n_users] = -1e5

    return score_matrix


def cal_ndcg(topk, test_set, num_pos, k):
    # 限制 k 的范围
    n = min(num_pos, k)
    nrange = np.arange(n) + 2
    idcg = np.sum(1 / np.log2(nrange))

    dcg = 0
    for i, s in enumerate(topk):
        if s in test_set:
            dcg += 1 / np.log2(i + 2)

    # ndcg = dcg / idcg
    ndcg = dcg / idcg if idcg > 0 else 0

    return ndcg


def cal_mrr(topk, test_set):
    for rank, item in enumerate(topk, 1):
        if item in test_set:
            return 1 / rank
    return 0


def cal_auc(topk, test_set, all_items):
    pos_ranks = [idx + 1 for idx, item in enumerate(topk) if item in test_set]
    if not pos_ranks:
        return 0.5  # 如果没有正样本，AUC 默认为0.5
    M = len(test_set)  # 正样本数量
    N = len(all_items)  # 所有可能的项
    sum_ranks = sum(pos_ranks)
    auc = (sum_ranks - (M * (M + 1) / 2)) / (M * (N - M))
    return auc


def cal_novelty(topk, item_popularity):
    novelty = 0
    for item in topk:
        novelty += 1 - item_popularity.get(item, 0)  # 使用项目流行度的倒数，如果项目未知则假设最大新颖性
    return novelty / len(topk)


def cal_diversity(topk, item_similarity):
    diversity = 0
    num_pairs = 0
    for i in range(len(topk)):
        for j in range(i + 1, len(topk)):
            diversity += 1 - item_similarity.get((topk[i], topk[j]), 0)  # 如果项目间无相似度数据，则假设不相似
            num_pairs += 1
    if num_pairs == 0:
        return 0
    return diversity / num_pairs


def test_v2(model, ks, ckg, n_batchs=4, item_popularity=None, item_similarity=None):
    ks = eval(ks)
    train_user_dict, test_user_dict = ckg.train_user_dict, ckg.test_user_dict

    all_items = set(range(ckg.n_items))  # 假设所有项目的集合

    n_users = ckg.n_users
    n_items = ckg.n_items
    n_test_users = len(test_user_dict)

    n_k = len(ks)
    result = {
        "precision": np.zeros(n_k),
        "recall": np.zeros(n_k),
        "ndcg": np.zeros(n_k),
        "hit_ratio": np.zeros(n_k),
        "mmr": np.zeros(n_k),
        "auc": np.zeros(n_k),
        "novelty": np.zeros(n_k),
        "diversity": np.zeros(n_k)
    }

    n_users = model.n_users
    batch_size = n_users // n_batchs
    for batch_id in tqdm(range(n_batchs), ascii=True, desc="Evaluate"):
        s = batch_size * batch_id
        t = batch_size * (batch_id + 1)
        if t > n_users:
            t = n_users
        if s == t:
            break

        score_matrix = get_score(model, n_users, n_items, train_user_dict, s, t)
        for i, k in enumerate(ks):
            # precision, recall, ndcg, hr, mrr = 0, 0, 0, 0, 0
            metrics = {key: 0 for key in result.keys()}

            _, topk_index = torch.topk(score_matrix, k)
            topk_index = topk_index.cpu().numpy() + n_users

            for u in range(s, t):
                # gt_pos = test_user_dict[u]
                # print(f"User {u} positive samples: {gt_pos}")
                # topk = topk_index[u - s]  # 获取当前用户的top-k推荐列表
                # num_pos = len(gt_pos)  # 获取当前用户的正样本数量
                # print(f"获取当前用户的正样本数量{num_pos}")

                gt_pos = test_user_dict.get(u, [])
                if not gt_pos:  # 如果正样本物品列表为空，则跳过当前用户的处理，进入下一个用户的处理
                    # print(f"User {u} has no positive samples.")
                    continue
                # else:  # 如果正样本物品列表不为空，则打印正样本列表
                #     print(f"User {u} positive samples: {gt_pos}")
                topk = topk_index[u - s]  # 获取当前用户的top-k推荐列表
                num_pos = len(gt_pos)  # 获取当前用户的正样本数量
                # print(f"获取当前用户的正样本数量{num_pos}")

                topk_set = set(topk)  # 转换为集合方便计算
                test_set = set(gt_pos)  # 转换为集合方便计算
                num_hit = len(topk_set & test_set)  # 计算推荐列表和测试集正样本的交集数量
                # print(f"计算推荐列表和测试集正样本的交集数量{num_hit}")

                # precision += num_hit / k
                # recall += num_hit / num_pos
                # hr += 1 if num_hit > 0 else 0
                #
                # ndcg += cal_ndcg(topk, test_set, num_pos, k)
                # mrr += cal_mrr(topk, test_set)

                # result["precision"][i] += precision / n_test_users
                # result["recall"][i] += recall / n_test_users
                # result["ndcg"][i] += ndcg / n_test_users
                # result["hit_ratio"][i] += hr / n_test_users
                # result["mrr"][i] += mrr / n_test_users  # 累加每个批次的 MRR


                metrics["precision"] += num_hit / k
                metrics["recall"] += num_hit / num_pos
                metrics["hit_ratio"] += 1 if num_hit > 0 else 0
                metrics["ndcg"] += cal_ndcg(topk, test_set, num_pos, k)
                metrics["mmr"] += cal_mrr(topk, test_set)
                metrics["auc"] += cal_auc(topk, test_set, all_items)
                metrics["novelty"] += cal_novelty(topk, item_popularity)
                metrics["diversity"] += cal_diversity(topk, item_similarity)

            for key in metrics:
                result[key][i] += metrics[key] / n_test_users



    return result
