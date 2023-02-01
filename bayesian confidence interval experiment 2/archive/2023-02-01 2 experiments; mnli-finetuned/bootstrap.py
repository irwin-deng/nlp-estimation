import numpy as np
import torch
import datasets
import csv
from datasets import load_dataset
from similarity import get_distance_matrix
from tqdm import tqdm
from bert_classifier import BertClassifier, preprocess_nli_dataset
from utils import shuffle_tensor
from common import device

bert_classifier: torch.nn.Module

mnli_test: dict[str, torch.Tensor]
snli_test: dict[str, torch.Tensor]



def weighted_sampler(weights: torch.Tensor, batch_size: int, debug: bool = False) -> torch.Tensor:
    """
    Generate samples for a batch of unlabeled samples by performing weighted
    sampling from the provided weights vector

    :param weights: The entry at index i contains the weight of sampling the ith
        example from the labeled dataset
    :returns: a tensor consisting of the sampled indices from the labeled dataset
    """

    # For each sample in the unlabeled batch, get a sample from the labeled batch
    sampled_labeled_indices = torch.multinomial(weights, num_samples=batch_size, replacement=True).squeeze()
    if debug:
        if sampled_labeled_indices.size() != (batch_size,):
            raise AssertionError(f"sampled_labeled_indices size: {sampled_labeled_indices.size()}"
                f"batch_size: {batch_size}")
    return sampled_labeled_indices



def unweighted_sampler(labeled_indices: torch.Tensor, batch_size: int, debug: bool = False) -> torch.Tensor:
    """
    Generate a batch of size batch_size consisting of randomly sampled
    observations (with replacement) from the labeled dataset

    :param labeled_indices: a tensor consisting of the subset of the labeled
        dataset's indices that can be sampled from
    :param batch_size: the number of samples to get for this batch
    :returns: a tensor consisting of the sampled indices from the labeled dataset
    """

    sampled_labeled_indices = labeled_indices[
        torch.randint(len(labeled_indices), size=(batch_size,), device=device)]
    if debug:
        if not torch.isin(sampled_labeled_indices, labeled_indices).all():
            raise AssertionError()
        if sampled_labeled_indices.size() != (batch_size,):
            raise AssertionError(f"sampled_labeled_indices size: {sampled_labeled_indices.size()}")
    return sampled_labeled_indices



def compute_correctly_classified(classifier: torch.nn.Module, dataset: dict[torch.Tensor], debug: bool = False):
    """
    Add a new boolean field 'is_correct' to the dataset indicating whether the classifier
    predicted the label correctly
    """
    dataset_size = len(dataset["label"])
    batch_size = 128
    dataset["is_correct"] = torch.empty_like(dataset["label"], dtype=torch.bool, device=device)

    for batch_indices in tqdm(torch.split(torch.arange(dataset_size, device=device), batch_size)):
        target = dataset["label"][batch_indices].to(device)
        output = classifier.forward(dataset["input_ids"][batch_indices].to(device),
            dataset["attention_mask"][batch_indices].to(device))
        predictions = output.argmax(dim=1, keepdim=True).squeeze()
        is_correct = torch.eq(target, predictions)
        if debug:
            if target.size() != (len(batch_indices),):
                raise AssertionError(f"target size: {target.size()}")
            if output.size() != (len(batch_indices), classifier.num_labels):
                raise AssertionError(f"output size: {output.size()}")
            if predictions.size() != (len(batch_indices),):
                raise AssertionError(f"predictions size: {predictions.size()}, ")
            if is_correct.size() != (len(batch_indices),):
                raise AssertionError(f"is_correct size: {is_correct.size()}, ")
        dataset["is_correct"][batch_indices] = is_correct



def get_accuracy(dataset: dict[str, torch.Tensor], dataset_indices: torch.Tensor = None
    ) -> torch.Tensor:
    """
    Compute the accuracy of the classifier on the provided dataset

    :param dataset_indices: the subset of the dataset to compute the accuracy
        for. May include duplicates, in which each row of dataset_indices is weighted
        equally.
    :returns: a singleton float tensor containing the accuracy
    """

    if dataset_indices is None:
        dataset_indices = torch.arange(len(dataset["label"]), device=device)

    if "is_correct" not in dataset.keys():
        raise AssertionError("Must call compute_correctly_classified() first to "
            "precompute predicted samples")

    return torch.mean(dataset["is_correct"][dataset_indices], dtype=torch.float)



def get_confidence_interval(labeled_ds: dict[torch.Tensor],
        labeled_ds_indices: torch.Tensor, unlabeled_ds_indices: torch.Tensor,
        weighted: bool = False, weights: torch.Tensor = None,
        verbose: bool = False, debug: bool = False
    ) -> tuple[float, float]:
    """
    Compute a confidence interval by sampling with repetition from the labeled
    dataset. Returns a tuple consisting of the lower and upper bounds of the
    confidence interval
    """
    if weighted:
        assert weights is not None

    n_bootstrap_iterations = 10000
    sampled_accuracies = []  # Sampled accuracies from each bootstrap iteration

    tqdm_ci = tqdm(range(1, n_bootstrap_iterations + 1), disable = not verbose)
    for iteration in tqdm_ci:
        if verbose:
            tqdm_ci.set_description(f"Iteration {iteration} / {n_bootstrap_iterations}")
        if weighted:
            labeled_samples = weighted_sampler(weights=weights,
                batch_size=len(labeled_ds_indices), debug=debug)
        else:
            labeled_samples = unweighted_sampler(labeled_indices=labeled_ds_indices,
                batch_size=len(labeled_ds_indices), debug=debug)
        sampled_accuracies.append(get_accuracy(dataset=labeled_ds,
            dataset_indices=labeled_samples))

    sampled_accuracies = torch.tensor(sampled_accuracies, device=device)
    confidence_interval = (torch.quantile(sampled_accuracies, 0.025).item(),
        torch.quantile(sampled_accuracies, 0.975).item())
    return confidence_interval



def ci_experiment_repeated(snli_labeled: bool, seed: int = 0,
        results_save_path: str = None, verbose: bool = False, debug: bool = False):
    """
    Compute the proportion of confidence intervals that contain the true value
    of the classifier's accuracy on the unlabeled dataset, comparing weighted
    vs unweighted sampling.

    :param snli_labeled: Whether to use examples from the SNLI dataset as
        the labeled dataset. If True, we will use the examples from SNLI not
        selected to be the unlabeled dataset. If False, we will use the entire
        MNLI dataset as the labeled dataset.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    if snli_labeled:
        print("\nRunning experiment with 100 unlabeled and rest labeled samples from SNLI")
        unlabeled_ds = snli_test
        labeled_ds = snli_test
    else:
        print("\nRunning experiment with 100 unlabeled from SNLI, labeled from MNLI")
        unlabeled_ds = snli_test
        labeled_ds = mnli_test

    # Precompute whether the classifier can correctly classify each element of the dataset(s)
    if verbose:
        print("\nPrecomputing predicted values on datasets...")
    bert_classifier.to(device)
    compute_correctly_classified(bert_classifier, snli_test, debug=debug)
    if not snli_labeled:
        compute_correctly_classified(bert_classifier, mnli_test, debug=debug)
    bert_classifier.cpu()

    n_iterations = 1000 if not debug else 10  # Number of different confidence intervals to generate
    unlabeled_size = 100  # Number of examples from SNLI to use as unlabeled dataset
    snli_indices = torch.arange(len(snli_test["label"]), device=device)

    # Calculate weights matrix
    distances = get_distance_matrix(unlabeled_ds=unlabeled_ds,
        labeled_ds=labeled_ds, verbose=verbose, debug=debug)
    weights = distances ** -1

    accuracies, weighted_cis, unweighted_cis = [], [], []  # Accuracies; confidence intervals produced by weighted, unweighte sampling
    weighted_in_ci, unweighted_in_ci = 0, 0  # The number of trials in which weighted and unweighted sampling produced a CI containing the true accuracy

    if verbose:
        print("Generating confidence intervals...")

    tqdm_iterator = tqdm(range(1, n_iterations + 1), disable = not verbose)
    for iteration in tqdm_iterator:
        tqdm_iterator.set_description(f"Iteration {iteration} / {n_iterations}")

        # Get indices of unlabeled samples from SNLI
        snli_indices = shuffle_tensor(snli_indices)
        unlabeled_indices = snli_indices[torch.arange(unlabeled_size, device=device)]
        # Get indices of labeled samples
        if snli_labeled:
            labeled_indices = snli_indices[torch.arange(unlabeled_size, len(snli_indices), device=device)]
        else:
            labeled_indices = torch.arange(len(mnli_test["label"]), device=device)

        # Get true accuracy
        accuracy = get_accuracy(dataset=unlabeled_ds, dataset_indices=unlabeled_indices)
        accuracies.append(accuracy)

        # Calculate weights given the subset of the labeled dataset
        weights_subset = weights[unlabeled_indices].clone().detach()
        if snli_labeled:
            weights_subset[:, unlabeled_indices] = 0
        if debug:
            if weights_subset.size() != (len(unlabeled_indices), len(labeled_ds["label"])):
                raise AssertionError(f"weights_subset size: {weights_subset.size()})")
        # Normalize sum of weights in each row to 1
        weights_subset = weights_subset / weights_subset.sum(dim=1, keepdim=True)
        # Combine into a single weight vector
        weights_subset = torch.sum(weights_subset, dim=0)
        if debug:
            if weights_subset.size() != (len(labeled_ds["label"]),):
                raise AssertionError(f"weights_subset size: {weights_subset.size()})")

        # Weighted sample
        weighted_ci = get_confidence_interval(labeled_ds=labeled_ds,
            labeled_ds_indices=labeled_indices,
            unlabeled_ds_indices=unlabeled_indices, weighted=True,
            weights=weights_subset, verbose=False, debug=debug)
        weighted_cis.append(weighted_ci)
        if weighted_ci[0] <= accuracy and accuracy <= weighted_ci[1]:
            weighted_in_ci += 1

        # Unweighted sample
        unweighted_ci = get_confidence_interval(labeled_ds=labeled_ds,
            labeled_ds_indices=labeled_indices,
            unlabeled_ds_indices=unlabeled_indices, weighted=False,
            verbose=False, debug=debug)
        unweighted_cis.append(unweighted_ci)
        if unweighted_ci[0] <= accuracy and accuracy <= unweighted_ci[1]:
            unweighted_in_ci += 1
        
        tqdm_iterator.set_postfix({"weighted proportion": weighted_in_ci / iteration,
            "unweighted proportion": unweighted_in_ci / iteration})
    
    print(f"\nWeighted proportion: {weighted_in_ci / n_iterations}")
    print(f"Unweighted proportion: {unweighted_in_ci / n_iterations}")

    # Write data to CSV
    if results_save_path:
        file = open(results_save_path, 'w')
        csv_writer = csv.writer(file)
        csv_writer.writerow(["True Accuracy", "Weighted CI Lower", "Weighted CI Upper", "Unweighted CI Lower", "Unweighted CI Upper"])
        for accuracy, (w0, w1), (u0, u1) in zip(accuracies, weighted_cis, unweighted_cis):
            csv_writer.writerow([accuracy.item(), w0, w1, u0, u1])
        file.close()
        if verbose:
            print(f"Saved results to {results_save_path}")


if __name__ == '__main__':
    print("Loading model...")
    bert_classifier = BertClassifier(n_labels=3)
    bert_classifier.load_state_dict(torch.load("finetuned_mnli.pt"))  # Path to fine-tuned BERT classifier
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

        ci_experiment_repeated(snli_labeled=True, results_save_path="results_snli-snli.csv",
            seed=0, verbose=True, debug=False)
        ci_experiment_repeated(snli_labeled=False, results_save_path="results_mnli-snli.csv",
            seed=0, verbose=True, debug=False)
