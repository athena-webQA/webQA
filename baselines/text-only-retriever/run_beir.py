from collections import defaultdict
import sys
import pytrec_eval
from beir import util
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from typing import Type, List, Dict, Union, Tuple
from models import Splade, BEIRSpladeModel, BEIRDPR
from beir.retrieval.custom_metrics import mrr
from transformers import AutoTokenizer, DPRQuestionEncoder, DPRContextEncoder
from beir.retrieval.search.lexical import BM25Search as BM25

import argparse

import pdb

def evaluate(qrels: Dict[str, Dict[str, int]],
             results: Dict[str, Dict[str, float]],
             k_values: List[int]) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:

    ndcg = {}
    _map = {}
    recall = {}
    precision = {}
    _mrr = {}

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

    _mrr = mrr(qrels, results, k_values)

    for eval in [ndcg, _map, recall, precision, _mrr]:
        for k in eval.keys():
            print("{}: {:.4f}".format(k, eval[k]))

    return ndcg, _map, recall, precision, _mrr

def recall_at_k(qrels, results, k=10):
    not_retreived = []
    recall = 0
    counts = 0

    for query_id in qrels.keys():
        results_at_k = [doc_id for (doc_id, score) in sorted(results[query_id].items(), key=lambda x: x[1], reverse=True)][:k]
        results_at_k = set(results_at_k)
        for doc_id in qrels[query_id].keys():
            if doc_id in results_at_k:
                recall += 1
            else:
                if qrels[query_id][doc_id] > 0:
                    not_retreived.append((query_id, doc_id))
            counts += 1

    return recall/counts, not_retreived

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, choices=['bm25', 'dpr-ft', 'dpr-nq', 'tas-b', 'distilsplade', 'splade', 'splade-ft', 'distilsplade-ft', 'tas-b'])
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--data_dir", default="./text_only_retrieval_beir")
    parser.add_argument("--split", default="dev", choices=["dev", "image-dev", "text-dev"])
    parser.add_argument("--dump_prefix", type=str)

    args = parser.parse_args()
    return args

args = get_args()

corpus, queries, qrels = GenericDataLoader(
    args.data_dir).load(split=args.split)

# corpus, queries, qrels = GenericDataLoader(
#     "/home/adityasv/webqa/OFA_retrieval_beir").load(split=args.split)

print("lengths of queries, corpus, qrels:", len(queries), len(corpus), len(qrels))

model_name = args.model_name
model_path = args.model_path

print(model_name, model_path)

if model_name == "dpr-nq":
    if model_path is None:
        model = DRES(models.SentenceBERT(
            "sentence-transformers/facebook-dpr-question_encoder-multiset-base"), batch_size=128)
    else:
        model = DRES(models.SentenceBERT(
            model_path), batch_size=128)

elif model_name == "tas-b":
    if model_path is None:
        model = DRES(models.SentenceBERT(
            "sentence-transformers/msmarco-distilbert-base-tas-b"), batch_size=128)
    else:
        model = DRES(models.SentenceBERT(
            model_path), batch_size=128)

elif model_name == "splade":
    if model_path is None:
        model_type_or_dir = "/home/adityasv/COILv2/splade/weights/splade_max"
        model = Splade(model_type_or_dir)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
        beir_splade = BEIRSpladeModel(model, tokenizer)
        model = DRES(beir_splade)

elif model_name == "distilsplade":
    if model_path is None:
        model_type_or_dir = "/home/adityasv/COILv2/splade/weights/distilsplade_max"
        model = Splade(model_type_or_dir)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
        beir_splade = BEIRSpladeModel(model, tokenizer)
        model = DRES(beir_splade)

elif model_name == "bm25":
    hostname = "localhost" #localhost
    index_name = "webqa" # scifact
    initialize = True # False
    number_of_shards = 1

    model = BM25(index_name=index_name, hostname=hostname, initialize=initialize, number_of_shards=number_of_shards)

retriever = EvaluateRetrieval(model, score_function="dot") # retriever retrieves topk +1 for some reason
results = retriever.retrieve(corpus, queries)

assert len(results) == len(queries)
for query_id in results.keys():
    assert query_id in qrels
    # assert len(results[query_id]) == 10, f"{len(results[query_id])}"
    for doc_id,score in results[query_id].items():
        assert doc_id in corpus

_mrr = mrr(qrels, results, [10])
recall, not_retrieved = recall_at_k(qrels, results)

ndcg, _map, recall, precision, _mrr = evaluate(qrels, results, [2, 3, 5, 10, 100])
print(ndcg, _map, recall, precision, _mrr)

with open(f'{args.dump_prefix}-{model_name}-{args.split}.tsv', 'w') as fo:
    fo.write(f"query_id\tdoc_id\tscore\n")
    for query_id in results.keys():
        for doc_id, score in results[query_id].items():
            fo.write(f"{query_id}\t{doc_id}\t{score}\n")

with open(f'{model_name}-{args.split}-not_retrieved.tsv', 'w') as fo:
    fo.write('query_id\tdoc_id\n')
    for qid, docid in not_retrieved:
        fo.write(f'{qid}\t{docid}\n')

# python run_beir.py splade
# python run_beir.py dpr
# python run_beir.py tas-b
# python run_beir.py distilsplade


