from typing import Any, Generator
import random
import numpy as np
from numpy.typing import NDArray
import torch
import datasets
from datasets import load_dataset
from similarity import get_approx_knn_matrix
from transformers import AutoModel, AutoTokenizer

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert_model = AutoModel.from_pretrained('bert-base-uncased').to(device)

mnli_ds_name = "multi_nli"
snli_ds_name = "snli"

def bert_cls_vector_batched(batch: datasets.arrow_dataset.Batch) -> NDArray[np.int32]:
    """
    Get the [CLS] embedding from the output of the base BERT model
    """
    encoding = bert_tokenizer.batch_encode_plus(
        [f"[CLS] {premise} [SEP] {hypothesis} [SEP]"
            for premise, hypothesis in zip(batch["premise"], batch["hypothesis"])],
        add_special_tokens = True,  # Add [CLS] and [SEP]
        max_length = 128,
        padding = "max_length",
        truncation = True,
        return_attention_mask = True,
        return_tensors = "pt"
    ).to(device)

    outputs = bert_model.forward(encoding["input_ids"], encoding["attention_mask"], output_hidden_states=True)
    last_layer_outputs = outputs[0]  # [batch_size, tokens, dimension]
    cls_vector = last_layer_outputs[:, 0, :]
    return cls_vector.cpu().detach().numpy()


def bert_cls_vector(sample: dict[str, torch.Tensor]) -> NDArray[np.int32]:
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


def bayesian_generator(labeled_ds: datasets.arrow_dataset.Dataset,
        nearest_indices: NDArray[np.int32], distances: NDArray[np.number],
        batch_size: int = 32, shuffle_data: bool = True
    ) -> Generator[tuple[NDArray, NDArray], None, None]:
    """
    Generate samples by performing weighted sampling

    :param labeled_ds: the labeled dataset
    :param nearest_indices: The ith row contains the indices of the k closest
        unlabeled samples to the ith labeled sample
    :param distances: The entry at indices (i, j) contains the distance between
        labeled sample i and unlabeled sample nearest_indices[i, j]
    :returns: a tuple consisting of a batch of train samples and their
        associated labels
    """
    # Convert distances to probabilities
    def get_probabilities(distances: NDArray[np.number]) -> NDArray[np.number]:
        """
        Convert a list of distances into probabilities, weighted by inverse of
        distance
        """
        inv_distances = distances ** -1
        normalized = inv_distances / sum(inv_distances)
        return normalized
    probabilities = np.apply_along_axis(get_probabilities, axis=1, arr=distances)

    unlabeled_ds_size = len(nearest_indices)
    unlabeled_ds_indices = list(range(unlabeled_ds_size))

    # Loop forever
    while True:
        if shuffle_data:
            random.shuffle(unlabeled_ds_indices)

        # Iterate through batches
        for batch_start in range(0, unlabeled_ds_size - batch_size, batch_size):
            # Crete X and Y arrays
            X_train = []
            Y_train = []

            # Iterate through batch
            unlabeled_batch = unlabeled_ds_indices[batch_start:batch_start+batch_size]
            for unlabeled_indx in unlabeled_batch:
                labeled_indx = np.random.choice(nearest_indices[unlabeled_indx], p=probabilities[unlabeled_indx])
                labeled_sample = labeled_ds[labeled_indx]
                X_train.append(labeled_sample["encoding"])
                Y_train.append(labeled_sample["label"])
            
            yield np.array(X_train), np.array(Y_train)


def nlp_experiment(seed: int, verbose: bool = False):
    if verbose:
        print("Downloading datasets...")

    mnli_train: datasets.arrow_dataset.Dataset = load_dataset(mnli_ds_name, split="train").with_format("np")  # type: ignore
    snli_train: datasets.arrow_dataset.Dataset = load_dataset(snli_ds_name, split="train").with_format("np")  # type: ignore
    snli_test: datasets.arrow_dataset.Dataset = load_dataset(snli_ds_name, split="validation").with_format("np")  # type: ignore

    # Get kNN matrix
    if verbose:
        print("Calculating kNN matrix...")
    nearest_indices, distances = get_approx_knn_matrix(
        unlabeled_ds = snli_train,
        labeled_ds = mnli_train,
        encoder = bert_cls_vector_batched,
        k = 10,
        seed = seed,
        verbose = verbose
    )

    generator = bayesian_generator(snli_train, nearest_indices, distances)

    # TODO: fine tune the BERT model to the generator

if __name__ == '__main__':
    nlp_experiment(seed = 0, verbose = True)
