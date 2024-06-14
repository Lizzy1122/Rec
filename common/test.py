import torch

import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


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
    n = min(num_pos, k)  # min(测试集正样本数量，推荐列表长度)
    nrange = np.arange(n) + 2  # 返回一个步长为1，起点为0，终点为n，步长为1的排列，每个值加2而不是加1，是因为排名从1开始
    idcg = np.sum(1 / np.log2(nrange))  # idcg的计算公式

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
    pos_ranks = [idx + 1 for idx, item in enumerate(topk) if item in test_set]  # 正样本在推荐列表中的排名（即正样本物品的位置）
    if not pos_ranks:
        return 0.5  # 如果没有正样本，AUC 默认为0.5
    M = len(test_set)  # 正样本数量
    N = len(all_items)  # 所有可能的项
    sum_ranks = sum(pos_ranks)
    auc = (sum_ranks - (M * (M + 1) / 2)) / (M * (N - M))
    return auc





# def cal_diversity(topk, item_similarity):
#     diversity = 0
#     num_pairs = 0
#     for i in range(len(topk)):
#         for j in range(i + 1, len(topk)):
#             diversity += 1 - item_similarity.get((topk[i], topk[j]), 0)  # 如果项目间无相似度数据，则假设不相似
#             num_pairs += 1
#     if num_pairs == 0:
#         return 0
#     return diversity / num_pairs


# # 用相似矩阵时的计算方法
# def cal_diversity(topk, item_similarity):
#     diversity = 0
#     num_pairs = 0
#     n_items = item_similarity.shape[0]  # 获取物品的数量
#     for i in range(len(topk)):
#         for j in range(i + 1, len(topk)):
#             if topk[i] < n_items and topk[j] < n_items:  # 检查索引是否在物品相似度矩阵范围内
#                 similarity = item_similarity[topk[i], topk[j]]
#                 diversity += 1 - similarity
#                 num_pairs += 1
#             # # 获取物品 i 和物品 j 的相似度
#             # similarity = item_similarity.get((topk[i], topk[j]), 0)
#             # # 更新多样性值
#             # diversity += 1 - similarity
#             # num_pairs += 1
#     # # 计算分母
#     # denominator = 0.5 * len(topk) * (len(topk) - 1)
#     # # 计算多样性
#     # diversity /= denominator
#     #
#     # return diversity
#     if num_pairs == 0:
#         return 0
#     return diversity / num_pairs



# 使用相似字典计算
# def cal_diversity(topk, item_similarity):
#     diversity = 0
#     num_pairs = 0
#     # 遍历 topk 中的所有物品组合
#     for i in range(len(topk)):
#         for j in range(i + 1, len(topk)):
#             # 从字典中获取两个物品之间的相似度，如果没有找到，默认为 0
#             similarity = item_similarity.get((topk[i], topk[j]), item_similarity.get((topk[j], topk[i]), 0))
#             diversity += 1 - similarity
#             num_pairs += 1
#
#     if num_pairs == 0:
#         return 0
#     return diversity / num_pairs

# 使用嵌入向量计算
# def cal_diversity(topk, item_embeddings):
#     # 确保topk索引不超出范围，避免越界报错
#     topk = np.clip(topk, 0, len(item_embeddings) - 1)
#     # 计算推荐列表中物品嵌入向量的相似度矩阵（余弦相似度）
#     pairwise_sim = cosine_similarity(item_embeddings[topk])
#     # 提取上三角矩阵（不包括对角线）中相似度值，并计算其均值
#     diversity = 1 - np.mean(pairwise_sim[np.triu_indices_from(pairwise_sim, k=1)])
#     return diversity

# 更正相似索引问题
# def cal_diversity(topk, item_similarity, n_users):
#     if len(topk) <= 1:
#         return 0
#     diversity = 0
#     num_pairs = 0
#     # 遍历 topk 中的所有物品组合
#     for i in range(len(topk)):
#         for j in range(i + 1, len(topk)):
#             # 从相似性矩阵中获取两个物品之间的相似度
#             similarity = item_similarity[topk[i]-n_users, topk[j]-n_users]
#             diversity += 1 - similarity
#             num_pairs += 1
#
#     if num_pairs == 0:
#         return 0
#     return diversity / num_pairs
#

# 第-1种计算方法：GPT给出，利用流行度计算
# def cal_novelty(topk, item_popularity):
#     novelty = 0
#     for item in topk:
#         novelty += 1 - item_popularity.get(item, 0)  # 使用项目流行度的倒数，如果项目未知则假设最大新颖性
#     return novelty / (len(topk) - 1)

# 第0种计算方法：修正索引
# def cal_novelty(topk, item_similarity, n_users):
#     if len(topk) <= 1:
#         return 0
#     novelty = 0
#     for i in range(len(topk)):
#         for j in range(0, len(item_similarity)-1):
#             novelty += 1 - item_similarity[topk[i]-n_users, topk[j]-n_users]
#     return novelty / (len(topk) - 1)

# 第1种计算方法：按照网上文章公式计算，不合理，大于1
# def cal_novelty(topk, item_similarity, n_users):
#     novelty = 0
#     for item_j in topk:
#         for item_i in topk:
#             if item_i != item_j:
#                 # 确保item_i和item_j在相似度矩阵范围内
#                 if item_i-n_users < len(item_similarity) and item_j-n_users < len(item_similarity):
#                     similarity_ij = item_similarity[item_i-n_users, item_j-n_users]
#                     novelty += 1 - similarity_ij
#     # 计算 novelty
#     R_u = len(topk)
#     if R_u > 1:
#         novelty /= (R_u - 1)
#         print("R_u>1")
#     return novelty

# 第2中计算方法：Z_u 不在用户历史列表，R_u表示系统给用户的推荐物品列表。 topk中的元素与在topk以外的元素
# def cal_novelty(topk, item_similarity, n_users):
#     novelty = 0
#     U = set(topk)  # 全集
#     R_u = set()    # 已选取
#     Z_u = set()    # 未选取
#
#     if len(topk) <= 1:
#         print("Topk中没有元素")
#         return 0
#
#     for item_i in topk:
#         # item_i -= n_users
#         if item_i in R_u:  # 若i已被选取，则跳过
#             continue
#         R_u.add(item_i)  # 已选取集合
#         for item_j in topk:
#             # item_j -= n_users
#             Z_u = U - R_u  # 更新补集
#             if item_i != item_j and item_j in Z_u:  # 自己不和自己比，topk推荐列表中i不和已经交互过的j比
#                 if item_i - n_users < len(item_similarity) and item_j - n_users < len(item_similarity):  # 确保item_i和item_j在相似度矩阵范围内
#                     similarity_ij = item_similarity[item_i-n_users, item_j-n_users]
#                     # novelty += 1 - similarity_ij
#                     novelty += similarity_ij
#                     R_u.add(item_j)  # 添加已选取集合
#                     # print(str(item_j) + "已被选取")
#
#     novelty /= (len(topk)-1)  # 可能会出现小于0的情况
#     # print(novelty)
#     return novelty

# 第3种计算方法
def cal_novelty(topk, item_similarity, n_users, train_user_dict):
    novelty = 0
    R_u = set(topk)  # 推荐列表
    Z_u = set(train_user_dict)  # 已经交互过

    if len(topk) <= 1:
        print("Topk中没有元素")
        return 0
    #
    for item_i in topk:
        num = 0
        novelty_temp = 0
        for item_j in Z_u:
            if item_i != item_j:
                if item_i - n_users < len(item_similarity) and item_j < len(item_similarity):  # 确保item_i和item_j在相似度矩阵范围内
                    similarity_ij = item_similarity[item_i-n_users, item_j]
                    novelty_temp += 1 - similarity_ij
                    num += 1
        novelty = novelty_temp / num

    novelty /= (len(topk)-1)
    return novelty

# 根据计算公式、相似矩阵实现，逻辑正确
def cal_diversity(topk, item_similarity, n_users):
    diversity_sum = 0
    R_u = len(topk)
    for i in range(R_u):
        for j in range(i + 1, R_u):
            # 确保item_i和item_j在相似度矩阵范围内
            if topk[i]-n_users < len(item_similarity) and topk[j]-n_users < len(item_similarity):
                similarity_ij = item_similarity[topk[i]-n_users, topk[j]-n_users]
                diversity_sum += 1 - similarity_ij

    if R_u > 1:
        # 计算 diversity
        diversity = diversity_sum / (0.5 * R_u * (R_u - 1))
    else:
        diversity = 0

    return diversity


# def test_v2(model, ks, ckg, n_batchs=4, item_popularity=None, item_similarity=None):
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
    # 暂不处理
        "diversity": np.zeros(n_k)
    }

    n_users = model.n_users
    batch_size = n_users // n_batchs
    # item_embeddings = model.all_embed[n_users:].cpu().numpy() # 使用嵌入向量计算diversity


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

            jump = 0
            for u in range(s, t):
                # gt_pos = test_user_dict[u]
                # print(f"User {u} positive samples: {gt_pos}")
                # topk = topk_index[u - s]  # 获取当前用户的top-k推荐列表
                # num_pos = len(gt_pos)  # 获取当前用户的正样本数量
                # print(f"获取当前用户的正样本数量{num_pos}")

                # 获取 测试集 正样品列表
                gt_pos = test_user_dict.get(u, [])
                if not gt_pos:  # 如果正样本物品列表为空，则跳过当前用户的处理，进入下一个用户的处理
                    # print(f"User {u} has no positive samples.")
                    jump = jump + 1
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
                # metrics["novelty"] += cal_novelty(topk, item_popularity) # 第-1种
                # metrics["novelty"] += cal_novelty(topk, item_similarity, n_users) # 第0-2种
                metrics["novelty"] += cal_novelty(topk, item_similarity, n_users, ckg.train_user_dict) # 第3种
                metrics["diversity"] += cal_diversity(topk, item_similarity, n_users)
                # metrics["diversity"] += cal_diversity(topk, item_embeddings)

            for key in metrics:
                result[key][i] += metrics[key] / n_test_users



    return result
