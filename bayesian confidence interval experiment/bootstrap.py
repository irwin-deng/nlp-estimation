import math
import numpy as np
from numpy.typing import NDArray
import torch
import datasets
from datasets import load_dataset
from similarity import get_approx_knn_matrix
import transformers
from tqdm import tqdm
from bert_classifier import BertClassifier, preprocess_nli_dataset
from utils import get_batch_eval_metrics


bert_base = transformers.AutoModel.from_pretrained('bert-base-uncased')
bert_classifier: torch.nn.Module
mnli_test: datasets.arrow_dataset.Dataset
snli_test: datasets.arrow_dataset.Dataset


def bert_cls_vector_batched(batch: dict[str, NDArray]) -> NDArray[np.int32]:
    """
    Get the [CLS] embedding from the output of the base BERT model
    """

    outputs = bert_base.forward(batch["input_ids"].to(device), batch["attention_mask"].to(device), output_hidden_states=True)
    last_layer_outputs = outputs[0]  # [batch_size, tokens, dimension]
    cls_vector = last_layer_outputs[:, 0, :]
    return cls_vector.cpu().detach().numpy()


def weighted_sampler(labeled_ds: datasets.arrow_dataset.Dataset,
        labeled_indices: NDArray[np.int32], unlabeled_indices: NDArray[np.int32],
        nn_matrix: NDArray[np.int32], weights: NDArray[np.number]
    ) -> dict[str, torch.tensor]:
    """
    Generate samples for a batch of unlabeled samples by performing weighted
    sampling from the labeled dataset. For each unlabeled sample provided,
    a labeled sample is selected.

    :param labeled_ds: the labeled dataset
    :param labeled_indices: an array consisting of the subset of the labeled
        dataset's indices that can be sampled from
    :param unlabeled_indices: an array consisting of the subset of the unlabeled
        dataset's indices for which to draw samples for
    :param nn_matrix: The ith row contains the indices of the k closest
        unlabeled samples to the ith labeled sample
    :param weights: The entry at indices (i, j) contains the weight
        of sampling the jth index in the nn_matrix array given the
        unlabeled sample with index i in the unlabeled dataset
    :returns: a dict in which the keys are the column names,
        and the values are the values of that column for the batch
    """

    column_names = labeled_ds.format["columns"]

    # For each sample in the unlabeled batch, get a sample from the labeled batch
    def get_labeled_sample(unlabeled_indx: np.int32) -> np.int32:
        subset_indices = np.nonzero(np.isin(nn_matrix[unlabeled_indx], labeled_indices))[0]
        nn_subset = nn_matrix[unlabeled_indx][subset_indices]
        weights_subset = weights[unlabeled_indx][subset_indices]
        probabilities = weights_subset / sum(weights_subset)
        labeled_indx = np.random.choice(nn_subset, p=probabilities)
        assert labeled_indx in labeled_indices
        return labeled_indx
    labeled_indices = np.vectorize(get_labeled_sample)(unlabeled_indices)

    # Convert to a dict of tensors
    sampled_batch = {}
    for column in column_names:
        sampled_batch[column] = torch.stack([labeled_ds[labeled_indx.item()][column]
            for labeled_indx in labeled_indices])
    return sampled_batch


def unweighted_sampler(labeled_ds: datasets.arrow_dataset.Dataset,
        labeled_indices: NDArray[np.int32], batch_size: int) -> dict[str, torch.tensor]:
    """
    Generate a batch of size batch_size consisting of randomly sampled
    observations (with replacement) from the labeled dataset

    :param labeled_ds: the labeled dataset
    :param labeled_indices: an array consisting of the subset of the labeled
        dataset's indices that can be sampled from
    :param batch_size: the number of samples to get for this batch
    :returns: a dict in which the keys are the column names,
        and the values are the values of that column for the batch
    """

    column_names = labeled_ds.format["columns"]

    labeled_indices = np.random.choice(labeled_indices, size=batch_size)

    # Convert to a dict of tensors
    sampled_batch = {}
    for column in column_names:
        sampled_batch[column] = torch.stack([labeled_ds[labeled_indx.item()][column]
            for labeled_indx in labeled_indices])
    return sampled_batch


def get_accuracy(classifier: torch.nn.Module, dataset: datasets.arrow_dataset.Dataset,
        dataset_indices: NDArray[np.int32]) -> float:
    """
    Compute the accuracy of the classifier on the provided dataset
    """
    batch_size = 128

    subset = torch.utils.data.Subset(dataset, dataset_indices.tolist())
    test_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size)
    n_correct, n_samples = 0, 0
    for batch in test_loader:
        target = batch["label"].to(device)
        output = classifier.forward(batch["input_ids"].to(device),
            batch["attention_mask"].to(device))
        metrics = get_batch_eval_metrics(target, output)

        n_samples += metrics["batch_size"]
        n_correct += metrics["accuracy"] * metrics["batch_size"]
    return n_correct / n_samples


def get_confidence_interval(classifier: torch.nn.Module,
        labeled_ds: datasets.arrow_dataset.Dataset, labeled_ds_indices: NDArray[np.int32],
        unlabeled_ds_indices: NDArray[np.int32], weighted: bool = False,
        nn_matrix: NDArray = None, weights: NDArray = None, verbose: bool = False
    ) -> tuple[float, float]:
    """
    Compute a confidence interval by sampling with repetition from the labeled
    dataset. Returns a tuple consisting of the lower and upper bounds of the
    confidence interval
    """
    if weighted:
        assert nn_matrix is not None and weights is not None

    n_bootstrap_iterations = 10
    batch_size = 128

    n_batches = math.ceil(len(unlabeled_ds_indices) / batch_size)
    sampled_accuracies = []  # Sampled accuracies from each bootstrap iteration

    tqdm_ci = tqdm(range(1, n_bootstrap_iterations + 1), disable = not verbose)
    for iteration in tqdm_ci:
        n_samples, n_correct = 0, 0
        np.random.shuffle(unlabeled_ds_indices)
        if verbose:
            tqdm_ci.set_description(f"Iteration {iteration} / {n_bootstrap_iterations}")
        for unlabeled_batch_indices in np.array_split(unlabeled_ds_indices, n_batches):
            if weighted:
                labeled_batch = weighted_sampler(labeled_ds = labeled_ds,
                    unlabeled_indices = unlabeled_batch_indices, labeled_indices = labeled_ds_indices,
                    nn_matrix = nn_matrix, weights = weights)
            else:
                labeled_batch = unweighted_sampler(labeled_ds = labeled_ds,
                    labeled_indices = labeled_ds_indices, batch_size = len(unlabeled_batch_indices))
            target = labeled_batch["label"].to(device)
            output = classifier.forward(labeled_batch["input_ids"].to(device),
                labeled_batch["attention_mask"].to(device))
            metrics = get_batch_eval_metrics(target, output)
            n_samples += metrics["batch_size"]
            n_correct += metrics["accuracy"] * metrics["batch_size"]
        sampled_accuracies.append(n_correct / n_samples)

    confidence_interval = (np.percentile(sampled_accuracies, 2.5), np.percentile(sampled_accuracies, 97.5))
    return confidence_interval


def nlp_experiment_1(seed: int = 0, verbose: bool = False):
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("Running experiment with labeled dataset MNLI, unlabeled dataset SNLI")

    bert_base.to(device)
    nearest_indices, distances = get_approx_knn_matrix(unlabeled_ds=snli_test,
        labeled_ds=mnli_test, encoder=bert_cls_vector_batched, k=10, seed=seed, verbose=verbose)
    bert_base.cpu()
    weights = distances ** -1

    bert_classifier.to(device)
    if verbose:
        print("\nCalculating accuracy on SNLI dataset...")
    accuracy = get_accuracy(bert_classifier, snli_test, np.arange(len(snli_test)))
    print(f"Accurcy on SNLI dataset: {accuracy}")

    if verbose:
        print("\nCalculating confidence interval with weighted sampling...")
    ci = get_confidence_interval(bert_classifier, labeled_ds=mnli_test,
        labeled_ds_indices=np.arange(len(mnli_test)),
        unlabeled_ds_indices=np.arange(len(snli_test)), weighted=True,
        nn_matrix=nearest_indices, weights=weights, verbose=verbose)
    print(f"Confidence interval from weighted sampling: [{ci[0]}, {ci[1]}]")

    if verbose:
        print("\nCalculating confidence interval with unweighted sampling...")
    ci = get_confidence_interval(bert_classifier, labeled_ds=mnli_test,
        labeled_ds_indices=np.arange(len(mnli_test)),
        unlabeled_ds_indices=np.arange(len(snli_test)), weighted=False,
        nn_matrix=nearest_indices, weights=weights, verbose=verbose)
    print(f"Confidence interval from unweighted sampling: [{ci[0]}, {ci[1]}]")

    bert_classifier.cpu()


if __name__ == '__main__':
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    print("Loading model...")
    bert_classifier = BertClassifier(n_labels=3)
    bert_classifier.load_state_dict(torch.load("finetuned_mnli.pt"))
    bert_classifier.to(device)
    torch.no_grad()
    bert_classifier.eval()

    print("Downloading datasets...")
    mnli_test: datasets.arrow_dataset.Dataset = load_dataset("multi_nli", split="validation_matched").with_format("pt")  # type: ignore
    snli_test: datasets.arrow_dataset.Dataset = load_dataset("snli", split="test").with_format("pt")  # type: ignore

    # Preprocess datasets by removing invalid data points and encoding inputs with BERT encoder
    print("Preprocessing datasets...")
    mnli_test = preprocess_nli_dataset(mnli_test)
    snli_test = preprocess_nli_dataset(snli_test)

    nlp_experiment_1(seed=0, verbose=True)
