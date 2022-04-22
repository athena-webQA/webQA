python run_eval.py  \
     /home/adityasv/webqa/seq2seq/bart-webqa \
     /home/adityasv/webqa/text_only_retrieval_beir/reader/bm25-test.source \
     /home/adityasv/webqa/text_only_retrieval_beir/reader/bm25-test.preds \
     --task summarization \
     --bs 4 \
     --fp16



# /home/adityasv/webqa/seq2seq/webqa_dataset/preds/val.preds