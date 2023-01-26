import math
import numpy as np
import torch
import datasets
from datasets import load_dataset
from similarity import get_knn_matrix
from tqdm import tqdm
from bert_classifier import BertClassifier, preprocess_nli_dataset
from utils import get_batch_eval_metrics
from common import device

bert_classifier: torch.nn.Module

mnli_test: dict[str, torch.Tensor]
snli_test: dict[str, torch.Tensor]


def weighted_sampler(labeled_ds: dict[torch.Tensor],
        labeled_indices: torch.Tensor, unlabeled_indices: torch.Tensor,
        nn_matrix: torch.Tensor, weights: torch.Tensor, debug: bool = False
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

    batch_size = len(unlabeled_indices)

    # For each sample in the unlabeled batch, get a sample from the labeled batch
    def get_labeled_samples(unlabeled_indices: torch.Tensor) -> torch.Tensor:
        neighbors = nn_matrix[unlabeled_indices]
        subset_mask = torch.isin(neighbors, labeled_indices)
        weights_subset = weights[unlabeled_indices] * subset_mask
        sampled_neighbor_indices = torch.multinomial(weights_subset, num_samples=1).squeeze()
        sampled_labeled_indices = neighbors.gather(dim=1, index=sampled_neighbor_indices.unsqueeze(1)).squeeze()
        if debug:
            if not torch.isin(sampled_labeled_indices, labeled_indices).all().item():
                raise AssertionError()
            if sampled_labeled_indices.size() != unlabeled_indices.size():
                raise AssertionError(f"sampled_labeled_indices size: {sampled_labeled_indices.size()}"
                    f"unlabeled_indices size: {unlabeled_indices.size()}")
        return sampled_labeled_indices
    labeled_batch_indices = get_labeled_samples(unlabeled_indices)

    # Convert to a dict of tensors
    sampled_batch = {}
    for column in labeled_ds.keys():
        sampled_batch[column] = labeled_ds[column][labeled_batch_indices]
    return sampled_batch


def unweighted_sampler(labeled_ds: dict[torch.Tensor],
        labeled_indices: torch.Tensor, batch_size: int) -> dict[str, torch.tensor]:
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

    labeled_batch_indices = labeled_indices[torch.randint(batch_size, size=(batch_size,), device=device)]

    # Convert to a dict of tensors
    sampled_batch = {}
    for column in labeled_ds.keys():
        sampled_batch[column] = labeled_ds[column][labeled_batch_indices]
    return sampled_batch


def get_accuracy(classifier: torch.nn.Module, dataset: dict[torch.Tensor],
        dataset_indices: torch.Tensor) -> float:
    """
    Compute the accuracy of the classifier on the provided dataset
    """

    dataset_size = len(dataset["label"])
    batch_size = 128
    n_batches = math.ceil(dataset_size / batch_size)

    n_correct, n_samples = 0, 0
    for batch_indices in torch.split(torch.arange(dataset_size, device=device), batch_size):
        target = dataset["label"][batch_indices].to(device)
        output = classifier.forward(dataset["input_ids"][batch_indices].to(device),
            dataset["attention_mask"][batch_indices].to(device))
        metrics = get_batch_eval_metrics(target, output)

        n_samples += metrics["batch_size"]
        n_correct += metrics["accuracy"] * metrics["batch_size"]
    return n_correct / n_samples


def get_confidence_interval(classifier: torch.nn.Module,
        labeled_ds: dict[torch.Tensor], labeled_ds_indices: torch.Tensor,
        unlabeled_ds_indices: torch.Tensor, weighted: bool = False,
        nn_matrix: torch.Tensor = None, weights: torch.Tensor = None,
        verbose: bool = False, debug: bool = False
    ) -> tuple[float, float]:
    """
    Compute a confidence interval by sampling with repetition from the labeled
    dataset. Returns a tuple consisting of the lower and upper bounds of the
    confidence interval
    """
    if weighted:
        assert nn_matrix is not None and weights is not None

    n_bootstrap_iterations = 10000
    batch_size = 128

    n_batches = math.ceil(len(unlabeled_ds_indices) / batch_size)
    sampled_accuracies = []  # Sampled accuracies from each bootstrap iteration

    tqdm_ci = tqdm(range(1, n_bootstrap_iterations + 1), disable = not verbose)
    for iteration in tqdm_ci:
        n_samples, n_correct = 0, 0
        # Shuffle unlabeled dataset
        shuffled_indices = torch.randperm(unlabeled_ds_indices.shape[0], device=device)
        unlabeled_ds_indices = unlabeled_ds_indices[shuffled_indices]

        if verbose:
            tqdm_ci.set_description(f"Iteration {iteration} / {n_bootstrap_iterations}")
        for unlabeled_batch_indices in torch.tensor_split(unlabeled_ds_indices, n_batches):
            if weighted:
                labeled_batch = weighted_sampler(labeled_ds = labeled_ds,
                    unlabeled_indices = unlabeled_batch_indices, labeled_indices = labeled_ds_indices,
                    nn_matrix = nn_matrix, weights = weights, debug=debug)
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

    sampled_accuracies = torch.tensor(sampled_accuracies, device=device)
    confidence_interval = (torch.quantile(sampled_accuracies, 0.025).item(),
        torch.quantile(sampled_accuracies, 0.975).item())
    return confidence_interval


def nlp_experiment_1(seed: int = 0, verbose: bool = False, debug: bool = False):
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("Running experiment with labeled dataset MNLI, unlabeled dataset SNLI")

    # Calculate kNN matrix
    nearest_indices, distances = get_knn_matrix(unlabeled_ds=snli_test,
        labeled_ds=mnli_test, k=10, verbose=verbose, debug=debug)
    weights = distances ** -1

    bert_classifier.to(device)
    if verbose:
        print("\nCalculating accuracy on SNLI dataset...")
    accuracy = get_accuracy(bert_classifier, snli_test, torch.arange(len(snli_test["label"]), device=device))
    print(f"Accurcy on SNLI dataset: {accuracy}")

    if verbose:
        print("\nCalculating confidence interval with weighted sampling...")
    ci = get_confidence_interval(bert_classifier, labeled_ds=mnli_test,
        labeled_ds_indices=torch.arange(len(mnli_test["label"]), device=device),
        unlabeled_ds_indices=torch.arange(len(snli_test["label"]), device=device),
        weighted=True, nn_matrix=nearest_indices, weights=weights, verbose=verbose, debug=True)
    print(f"Confidence interval from weighted sampling: [{ci[0]}, {ci[1]}]")

    if verbose:
        print("\nCalculating confidence interval with unweighted sampling...")
    ci = get_confidence_interval(bert_classifier, labeled_ds=mnli_test,
        labeled_ds_indices=torch.arange(len(mnli_test["label"]), device=device),
        unlabeled_ds_indices=torch.arange(len(snli_test["label"]), device=device),
        weighted=False, verbose=verbose, debug=True)
    print(f"Confidence interval from unweighted sampling: [{ci[0]}, {ci[1]}]")

    bert_classifier.cpu()


if __name__ == '__main__':
    print("Loading model...")
    bert_classifier = BertClassifier(n_labels=3)
    bert_classifier.load_state_dict(torch.load("finetuned_mnli.pt"))
    bert_classifier.to(device)
    bert_classifier.eval()

    print("Downloading datasets...")
    mnli_test: datasets.arrow_dataset.Dataset = load_dataset("multi_nli", split="validation_matched").with_format("pt")  # type: ignore
    snli_test: datasets.arrow_dataset.Dataset = load_dataset("snli", split="test").with_format("pt")  # type: ignore

    with torch.no_grad():
        # Preprocess datasets by removing invalid data points and encoding inputs with BERT encoder
        print("Preprocessing datasets...")
        mnli_test = preprocess_nli_dataset(mnli_test)
        snli_test = preprocess_nli_dataset(snli_test)

        def convert_dataset_to_tensor_dict(dataset: datasets.arrow_dataset.Dataset):
            dataset = dataset.with_format("torch", device=device)
            return {column:dataset[column] for column in dataset.format["columns"]}
        mnli_test = convert_dataset_to_tensor_dict(mnli_test)
        snli_test = convert_dataset_to_tensor_dict(snli_test)

        nlp_experiment_1(seed=0, verbose=True, debug=False)
