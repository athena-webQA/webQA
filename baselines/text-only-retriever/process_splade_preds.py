from tqdm import tqdm
from collections import defaultdict
from beir.datasets.data_loader import GenericDataLoader


# corpus, queries, qrels = GenericDataLoader(
#     "/home/adityasv/webqa/OFA_retrieval_beir").load(split='dev')

# corpus, queries, qrels = GenericDataLoader(
#     "/home/adityasv/webqa/OFA_large_retrieval_beir").load(split='dev')

corpus, queries, qrels = GenericDataLoader(
    "/home/adityasv/webqa/text_only_retrieval_beir/").load(split='dev')

results = defaultdict(list)
with open("/home/adityasv/webqa/bm25-retrieved_results.tsv", 'r') as fi:
    for line in tqdm(fi.readlines()[1:]):
        query_id, doc_id, score = line.strip().split('\t')
        results[query_id].append((doc_id,score))

results = {k:sorted(v, key=lambda x: x[1], reverse=True)[:10] for k,v in results.items()}


with open('/home/adityasv/webqa/text_only_retrieval_beir/reader/bm25-test.idx', 'w') as fidx:
    with open('/home/adityasv/webqa/text_only_retrieval_beir/reader/bm25-test.source', 'w') as fsource:
        for qid, res in tqdm(results.items()):
            res = [f"{corpus[i[0]]['title']} {corpus[i[0]]['text']} " for i in res]
            res = ' </s> '.join(res)
            res = res.replace("\n", " ")
            question = queries[qid]
            question = question.lstrip("\"").rstrip("\"")
            out_line = question + " </s> " + res

            out_line = out_line.replace("\n", "").replace("\r","")
            fsource.write(out_line + '\n')
            fidx.write(str(qid) + '\n')