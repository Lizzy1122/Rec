import torch
from tqdm import tqdm

import heapq
import multiprocessing
from .metrics import *
from utility import metrics

from utility.parser import parse_args
from dataloader.data_processor import CKG_Data

# initialize the sources --- args_config and CKG --- shared with all programs.
args_config = parse_args()
CKG = CKG_Data(args_config=args_config)


_Ks = eval(args_config.Ks)
_train_user_dict, _test_user_dict = CKG.train_user_dict, CKG.test_user_dict
_n_test_users = len(_test_user_dict.keys())
_item_range = CKG.item_range

def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc

def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    return r

def get_performance(user_pos_test, r, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(metrics.precision_at_k(r, K))
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K))
        hit_ratio.append(metrics.hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio)}

def test_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    # uid
    u = x[1]
    # user u's items in the training set
    try:
        training_items = _train_user_dict[u]
    except Exception:
        training_items = []
    # user u's items in the test set
    user_pos_test = _test_user_dict[u]

    all_items = set(range(_item_range[0], _item_range[1] + 1))

    test_items = list(all_items - set(training_items))


    r = ranklist_by_heapq(user_pos_test, test_items, rating, _Ks)


    return get_performance(user_pos_test, r, _Ks)


def test(model, test_loader):


    result = {'precision': np.zeros(len(_Ks)), 'recall': np.zeros(len(_Ks)), 'ndcg': np.zeros(len(_Ks)),
              'hit_ratio': np.zeros(len(_Ks))}

    cores = multiprocessing.cpu_count() // 2
    pool = multiprocessing.Pool(cores)

    for batch_data in tqdm(test_loader, ascii=True, desc='Evaluate'):
        batch_u_id = batch_data['u_id']
        all_i_id = torch.arange(start=_item_range[0], end=_item_range[1] + 1, dtype=torch.long)

        if torch.cuda.is_available():
            all_i_id = all_i_id.cuda()

        batch_pred = model.inference(batch_u_id, all_i_id)

        user_batch_rating_uid = zip(batch_pred.cpu, batch_u_id.cpu)
        batch_result = pool.map(test_one_user, user_batch_rating_uid)

        for re in batch_result:
            result['precision'] += re['precision']/_n_test_users
            result['recall'] += re['recall']/_n_test_users
            result['ndcg'] += re['ndcg']/_n_test_users
            result['hit_ratio'] += re['hit_ratio']/_n_test_users

    pool.close()
    return result
