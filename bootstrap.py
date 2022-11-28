from typing import Any
import numpy as np
from numpy.typing import NDArray
import torch
import datasets
from datasets import load_dataset
from similarity import get_approx_knn_matrix
from transformers import BertModel, BertTokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model: BertModel = BertModel.from_pretrained('bert-base-uncased')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

mnli_ds_name = "multi_nli"
snli_ds_name = "snli"

def bert_cls_vector(sample: datasets.arrow_dataset.Example) -> NDArray[np.int32]:
    """
    Get the [CLS] embedding from the output of the base BERT model
    """
    encoding = bert_tokenizer.encode(
        text = f"[CLS] {sample['premise']} [SEP] {sample['hypothesis']} [SEP]",
        add_special_tokens = True,  # Add [CLS] and [SEP]
        max_length = 128,
        padding = "max_length",
        return_attention_mask = True,
        return_tensors = "pt"
    )

    outputs = bert_model.forward(encoding, output_hidden_states=True)
    last_layer_outputs = outputs[0]  # [batch_size, tokens, dimension]
    cls_vector = last_layer_outputs[0, 0, :]
    return cls_vector.detach().numpy()


def nlp_experiment(seed: int, verbose: bool = False):
    if verbose:
        print("Downloading datasets...")

    mnli_train = load_dataset(mnli_ds_name, split="train")
    snli_train = load_dataset(snli_ds_name, split="train")
    snli_test = load_dataset(snli_ds_name, split="validation")

    # Get kNN matrix
    if verbose:
        print("Calculating kNN matrix...")
    nearest_indices, distances = get_approx_knn_matrix(
        train = mnli_train,
        test = snli_train,
        encoder = bert_nli_cls_vector,
        k = 10,
        seed = seed,
        verbose = verbose
    )

    



if __name__ == '__main__':
    nlp_experiment(seed = 0, verbose = True)
