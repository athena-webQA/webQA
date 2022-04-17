import logging
from typing import List, Dict, Union

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
from tqdm.auto import trange
from transformers import AutoModelForMaskedLM

try:
    import sentence_transformers
    from sentence_transformers.util import batch_to_device
except ImportError:
    print("Import Error: could not load sentence_transformers... proceeding")
logger = logging.getLogger(__name__)


class BEIRSpladeModel:
    def __init__(self, model, tokenizer, max_length=256):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.model = model

    # Write your own encoding query function (Returns: Query embeddings as numpy array)
    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        X = self.model.encode_sentence_bert(self.tokenizer, queries, is_q=True, maxlen=self.max_length)
        return X

    # Write your own encoding corpus function (Returns: Document embeddings as numpy array)  
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        sentences = [(doc["title"] + ' ' + doc["text"]).strip() for doc in corpus]
        return self.model.encode_sentence_bert(self.tokenizer, sentences, maxlen=self.max_length)


class Splade(torch.nn.Module):

    def __init__(self, model_type_or_dir):
        super().__init__()
        self.transformer = AutoModelForMaskedLM.from_pretrained(model_type_or_dir)

    def forward(self, **kwargs):
        out = self.transformer(**kwargs)["logits"]  # output (logits) of MLM head, shape (bs, pad_len, voc_size)
        return torch.max(torch.log(1 + torch.relu(out)) * kwargs["attention_mask"].unsqueeze(-1), dim=1).values

    def _text_length(self, text: Union[List[int], List[List[int]]]):
        """helper function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):  # {key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, '__len__'):  # Object has no len() method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):  # Empty string or list of ints
            return len(text)
        else:
            return sum([len(t) for t in text])  # Sum of length of individual strings

    def encode_sentence_bert(self, tokenizer, sentences: Union[str, List[str], List[int]],
                             batch_size: int = 32,
                             show_progress_bar: bool = None,
                             output_value: str = 'sentence_embedding',
                             convert_to_numpy: bool = True,
                             convert_to_tensor: bool = False,
                             device: str = None,
                             normalize_embeddings: bool = False,
                             maxlen: int = 512,
                             is_q: bool = False) -> Union[List[Tensor], ndarray, Tensor]:
        """
        Computes sentence embeddings
        :param sentences: the sentences to embed
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Output a progress bar when encode sentences
        :param output_value:  Default sentence_embedding, to get sentence embeddings. Can be set to token_embeddings to get wordpiece token embeddings.
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
        :param device: Which torch.device to use for the computation
        :param normalize_embeddings: If set to true, returned vectors will have length 1. In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.
        :return:
           By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
        """
        self.eval()
        if show_progress_bar is None:
            show_progress_bar = True

        if convert_to_tensor:
            convert_to_numpy = False

        if output_value == 'token_embeddings':
            convert_to_tensor = False
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences, '__len__'):
            # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.to(device)

        all_embeddings = []
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index:start_index + batch_size]
            # features = tokenizer(sentences_batch)
            # print(sentences_batch)
            features = tokenizer(sentences_batch,
                                 add_special_tokens=True,
                                 padding="longest",  # pad to max sequence length in batch
                                 truncation="only_first",  # truncates to self.max_length
                                 max_length=maxlen,
                                 return_attention_mask=True,
                                 return_tensors="pt")
            # print(features)
            features = batch_to_device(features, device)

            with torch.no_grad():
                out_features = self.forward(**features)
                if output_value == 'token_embeddings':
                    embeddings = []
                    for token_emb, attention in zip(out_features[output_value], out_features['attention_mask']):
                        last_mask_id = len(attention) - 1
                        while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                            last_mask_id -= 1
                        embeddings.append(token_emb[0:last_mask_id + 1])
                else:  # Sentence embeddings
                    # embeddings = out_features[output_value]
                    embeddings = out_features
                    embeddings = embeddings.detach()
                    if normalize_embeddings:
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                    if convert_to_numpy:
                        embeddings = embeddings.cpu()
                all_embeddings.extend(embeddings)
        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        if input_was_string:
            all_embeddings = all_embeddings[0]
        return all_embeddings

class BEIRDPR:
    def __init__(self, query_encoder, doc_encoder, query_tokenizer, doc_tokenizer, device=None) -> None:
        self.query_encoder = query_encoder
        self.doc_encoder = doc_encoder
        self.query_tokenizer = query_tokenizer
        self.doc_tokenizer = doc_tokenizer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Use pytorch device: {}".format(device))

        self._target_device = torch.device(device)

    def _text_length(self, text: Union[List[int], List[List[int]]]):
        """helper function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):  # {key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, '__len__'):  # Object has no len() method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):  # Empty string or list of ints
            return len(text)
        else:
            return sum([len(t) for t in text])  # Sum of length of individual strings

    def encode(self, sentences: Union[str, List[str]],
               encoder,
               tokenizer,
               batch_size: int = 32,
               show_progress_bar: bool = True,
               output_value: str = 'pooler_output',
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False,
               device: str = None,
               normalize_embeddings: bool = False) -> Union[List[Tensor], ndarray, Tensor]:
        """
        Computes sentence embeddings

        :param sentences: the sentences to embed
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Output a progress bar when encode sentences
        :param output_value:  Default sentence_embedding, to get sentence embeddings. Can be set to token_embeddings to get wordpiece token embeddings. Set to None, to get all output values
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
        :param device: Which torch.device to use for the computation
        :param normalize_embeddings: If set to true, returned vectors will have length 1. In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.

        :return:
           By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
        """
        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel()==logging.INFO or logger.getEffectiveLevel()==logging.DEBUG)

        if convert_to_tensor:
            convert_to_numpy = False

        if output_value != 'pooler_output':
            convert_to_tensor = False
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences, '__len__'): #Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        if device is None:
            device = self._target_device

        encoder.to(device)


        all_embeddings = []
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index:start_index+batch_size]
            features = tokenizer(sentences_batch, padding=True, truncation='longest_first', return_tensors="pt")
            features = batch_to_device(features, device)

            with torch.no_grad():
                out_features = encoder(**features)

                if output_value == 'token_embeddings':
                    embeddings = []
                    for token_emb, attention in zip(out_features[output_value], out_features['attention_mask']):
                        last_mask_id = len(attention)-1
                        while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                            last_mask_id -= 1

                        embeddings.append(token_emb[0:last_mask_id+1])
                elif output_value is None:  #Return all outputs
                    embeddings = []
                    for sent_idx in range(len(out_features['pooler_output'])):
                        row =  {name: out_features[name][sent_idx] for name in out_features}
                        embeddings.append(row)
                else:   #Sentence embeddings
                    embeddings = out_features[output_value]
                    embeddings = embeddings.detach()
                    if normalize_embeddings:
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                    # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                    if convert_to_numpy:
                        embeddings = embeddings.cpu()

                all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings


    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        # encodings = self.query_encoder.encode_queries(queries, batch_size, **kwargs)
        encodings = self.encode(queries, self.query_encoder, self.query_tokenizer, batch_size=batch_size)
        return encodings
    
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        # encodings = self.doc_encoder.encode_corpus(corpus, batch_size, **kwargs)
        sentences = [(doc["title"] + ' ' + doc["text"]).strip() for doc in corpus]
        encodings = self.encode(sentences, self.doc_encoder, self.doc_tokenizer, batch_size=batch_size)
        return encodings