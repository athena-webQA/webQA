from multiprocess import Pool
import pytrec_eval
from tqdm.auto import tqdm
from rank_bm25 import BM25Okapi
from collections import defaultdict
from beir.retrieval.custom_metrics import mrr
from typing import Type, List, Dict, Union, Tuple
from beir.datasets.data_loader import GenericDataLoader

def evaluate(qrels: Dict[str, Dict[str, int]],
             results: Dict[str, Dict[str, float]],
             k_values: List[int]) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:

    ndcg = {}
    _map = {}
    recall = {}
    precision = {}

    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        precision[f"P@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {map_string, ndcg_string, recall_string, precision_string})
    scores = evaluator.evaluate(results)

    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
            precision[f"P@{k}"] += scores[query_id]["P_" + str(k)]

    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"]/len(scores), 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"]/len(scores), 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"]/len(scores), 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"]/len(scores), 5)


    for eval in [ndcg, _map, recall, precision]:
        for k in eval.keys():
            print("{}: {:.4f}".format(k, eval[k]))

    return ndcg, _map, recall, precision



def get_bm25(passages):
    passages_tokenized = [passage.strip().lower().split() for passage in passages]
    bm25 = BM25Okapi(passages_tokenized)
    return bm25


if __name__ == "__main__":
    corpus, queries, qrels = GenericDataLoader("/home/adityasv/webqa/text_only_retrieval_beir").load(split="text-dev")
    
    # passage_ids = [i for i in corpus.keys()]
    # passages = [f"{i['text']} {i['title']}" for i in corpus.values()]

    # query_ids = [i for i in queries.keys()]
    # query_list = [i for i in queries.values()]

    # bm25 = get_bm25(passages)

    retrieved_results = defaultdict(dict)

    k = [2,3,5,10,100]
    max_k = max(k)

    # for qid, query in tqdm(zip(query_ids, query_list), total=len(query_list)):
    #     scores_q = bm25.get_scores(query.strip().lower().split())
    #     retr_psgs = [(i, score) for score,i in sorted(zip(scores_q, passage_ids), reverse=True)]
    #     retr_psgs = retr_psgs[:max_k]
    #     for pid, score in retr_psgs:
    #         retrieved_results[qid][pid] = score


    with open('/home/adityasv/webqa/bm25-retrieved_results.tsv', 'r') as fi:
        fi.readline()
        for line in fi:
            qid, pid, score = line.strip().split('\t')
            score = float(score)
            retrieved_results[qid][pid] = score

    ndcg, _map, recall, precision = evaluate(qrels, retrieved_results, [2, 3, 5, 10, 100])
    print(ndcg, _map, recall, precision)


    # with open(f'bm25-retrieved_results.tsv', 'w') as fo:
    #     fo.write(f"query_id\tdoc_id\tscore\n")
    #     for query_id in retrieved_results.keys():
    #         for doc_id, score in retrieved_results[query_id].items():
    #             fo.write(f"{query_id}\t{doc_id}\t{score}\n")
        