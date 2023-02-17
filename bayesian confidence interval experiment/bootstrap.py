import numpy as np
import torch
import datasets
import csv
from datasets.load import load_dataset
from similarity import get_distance_matrix
from tqdm import tqdm
from typing import Optional
from bert_classifier import BertClassifier, preprocess_nli_dataset
from utils import shuffle_tensor
from common import device

bert_classifier: BertClassifier

mnli_test: dict[str, torch.Tensor]
snli_test: dict[str, torch.Tensor]


def compute_correctly_classified(classifier: torch.nn.Module, dataset: dict[str, torch.Tensor], debug: bool = False):
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



def get_accuracy(dataset: dict[str, torch.Tensor], dataset_indices: Optional[torch.Tensor] = None
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



def get_confidence_interval(labeled_ds: dict[str, torch.Tensor],
        labeled_ds_indices: torch.Tensor, n_bootstrap_samples: int,
        weighted: bool = False, weights: Optional[torch.Tensor] = None, debug: bool = False
    ) -> tuple[float, float]:
    """
    Compute a 95% confidence interval by sampling with repetition from the labeled
    dataset. If weighted=True, sample from weights tensor. If weighted=False, sample
    with equal probability from labeled_ds_indices. Returns a tuple consisting
    of the lower and upper bounds of the confidence interval
    """

    n_bootstrap_iterations = 10000
    
    # Convert to probability of correct
    if weighted:
        assert weights is not None
        probabilities = weights / weights.sum()
        p_correct = torch.where(labeled_ds["is_correct"], probabilities, 0).sum()
    else:
        p_correct = torch.sum(labeled_ds["is_correct"][labeled_ds_indices], dtype=torch.float64) / len(labeled_ds_indices)
    if debug:
        if len(p_correct.size()) != 0:
            raise AssertionError(f"p_correct.size(): {p_correct.size()}")
        if torch.logical_or(p_correct<0, p_correct>1):
            raise AssertionError(f"Invalid probability: {p_correct}")
    # Sample 
    accuracy_distribution = torch.distributions.binomial.Binomial(total_count=n_bootstrap_samples,
        probs=torch.full(size=(n_bootstrap_iterations,), fill_value=p_correct, device=device))  # type: ignore
    sampled_accuracies = accuracy_distribution.sample() / n_bootstrap_samples
    if debug:
        if sampled_accuracies.size() != (n_bootstrap_iterations,):
            raise AssertionError(f"sampled_accuracies.size(): {sampled_accuracies.size()}")
        if torch.logical_or(torch.any(sampled_accuracies<0), torch.any(sampled_accuracies>1)):
            raise AssertionError(f"Invalid accuracies: {sampled_accuracies}")
    confidence_interval = (torch.quantile(sampled_accuracies, 0.025).item(),
        torch.quantile(sampled_accuracies, 0.975).item())
    return confidence_interval



def ci_experiment_repeated(snli_labeled: bool, bootstrap_size: str,
        overlapping_labeled_unlabeled: bool = False, seed: int = 0, results_save_path: Optional[str] = None,
        verbose: bool = False, debug: bool = False):
    """
    Compute the proportion of confidence intervals that contain the true value
    of the classifier's accuracy on the unlabeled dataset, comparing weighted
    vs unweighted sampling.

    :param snli_labeled: Whether to use examples from the SNLI dataset as
        the labeled dataset. If True, we will use the examples from SNLI not
        selected to be the unlabeled dataset. If False, we will use the entire
        MNLI dataset as the labeled dataset.
    :param overlapping_labeled_unlabeled: whether to also include the unlabeled
        examples in the labeled dataset (this just serves as a sanity check)
    """
    if overlapping_labeled_unlabeled:
        assert snli_labeled
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
    print(f"with len({bootstrap_size} dataset) samples per bootstrap iteration")

    # Precompute whether the classifier can correctly classify each element of the dataset(s)
    if verbose:
        print("\nPrecomputing predicted values on datasets...")
    bert_classifier.to(device)
    compute_correctly_classified(bert_classifier, snli_test, debug=debug)
    if not snli_labeled:
        compute_correctly_classified(bert_classifier, mnli_test, debug=debug)
    bert_classifier.cpu()

    n_iterations = 10000 if not debug else 10  # Number of different confidence intervals to generate
    unlabeled_size = 100  # Number of examples from SNLI to use as unlabeled dataset
    n_bootstrap_samples: int  # Number of samples per bootstrap iteration
    if bootstrap_size == "unlabeled":
        n_bootstrap_samples = unlabeled_size
    elif bootstrap_size == "labeled":
        n_bootstrap_samples = len(labeled_ds["label"]) - (unlabeled_size if unlabeled_ds is labeled_ds else 0)
    else:
        raise AssertionError
    snli_indices = torch.arange(len(snli_test["label"]), device=device)

    # Calculate weights matrix
    distances = get_distance_matrix(unlabeled_ds=unlabeled_ds,
        labeled_ds=labeled_ds, encoding_type="cls", bert_model=bert_classifier.bert,
        verbose=verbose, debug=debug)
    epsilon = 0.1  # to make sure that no weight is infinite
    weights = torch.exp((distances+epsilon) ** -1)

    accuracies, weighted_cis, unweighted_cis = [], [], []  # Accuracies; confidence intervals produced by weighted, unweighte sampling
    weighted_in_ci, unweighted_in_ci = 0, 0  # The number of trials in which weighted and unweighted sampling produced a CI containing the true accuracy
    weighted_too_high, unweighted_too_high, weighted_too_low, unweighted_too_low = 0, 0, 0, 0  # The number of confidence intervals that were higher/lower than the true accuracy
    weighted_length_sum, unweighted_length_sum = 0, 0  # The sum of the lengths of the intervals

    if verbose:
        print("Generating confidence intervals...")
    tqdm_iterator = tqdm(range(1, n_iterations + 1), disable = not verbose)
    for iteration in tqdm_iterator:
        tqdm_iterator.set_description(f"Iteration {iteration} / {n_iterations}")

        # Get indices of unlabeled samples from SNLI
        snli_indices = shuffle_tensor(snli_indices)
        unlabeled_indices = snli_indices[torch.arange(unlabeled_size, device=device)]
        # Get indices of labeled samples
        if overlapping_labeled_unlabeled:
            labeled_indices = snli_indices[torch.arange(unlabeled_size*5)]
        elif snli_labeled:
            labeled_indices = snli_indices[torch.arange(unlabeled_size, len(snli_indices), device=device)]
        else:
            labeled_indices = torch.arange(len(mnli_test["label"]), device=device)

        # Get true accuracy
        accuracy = get_accuracy(dataset=unlabeled_ds, dataset_indices=unlabeled_indices)
        accuracies.append(accuracy)

        # Calculate weights given the subset of the labeled dataset
        weights_subset = weights[unlabeled_indices].clone().detach()
        if snli_labeled and overlapping_labeled_unlabeled:
            weights_subset[:, ~labeled_indices] = 0
        if snli_labeled and not overlapping_labeled_unlabeled:
            weights_subset[:, unlabeled_indices] = 0
        if debug:
            if weights_subset.size() != (len(unlabeled_indices), len(labeled_ds["label"])):
                raise AssertionError(f"weights_subset size: {weights_subset.size()})")
        # Normalize sum of weights in each row to 1
        weights_subset = weights_subset / weights_subset.sum(dim=1, keepdim=True)
        # Combine into a single weight vector, where the entry at index i is the probability of sampling the ith labeled example
        weights_subset = torch.sum(weights_subset, dim=0)
        if debug:
            if weights_subset.size() != (len(labeled_ds["label"]),):
                raise AssertionError(f"weights_subset size: {weights_subset.size()})")

        # Weighted sample
        weighted_ci = get_confidence_interval(labeled_ds=labeled_ds,
            labeled_ds_indices=labeled_indices, n_bootstrap_samples=n_bootstrap_samples,
            weighted=True, weights=weights_subset, debug=debug)
        weighted_cis.append(weighted_ci)
        if weighted_ci[0] <= accuracy and accuracy <= weighted_ci[1]:
            weighted_in_ci += 1
        elif weighted_ci[0] > accuracy:
            weighted_too_high += 1
        elif weighted_ci[1] < accuracy:
            weighted_too_low += 1
        weighted_length_sum += weighted_ci[1] - weighted_ci[0]

        # Unweighted sample
        unweighted_ci = get_confidence_interval(labeled_ds=labeled_ds,
            labeled_ds_indices=labeled_indices, n_bootstrap_samples=n_bootstrap_samples,
            weighted=False, debug=debug)
        unweighted_cis.append(unweighted_ci)
        if unweighted_ci[0] <= accuracy and accuracy <= unweighted_ci[1]:
            unweighted_in_ci += 1
        elif unweighted_ci[0] > accuracy:
            unweighted_too_high += 1
        elif unweighted_ci[1] < accuracy:
            unweighted_too_low += 1
        unweighted_length_sum += unweighted_ci[1] - unweighted_ci[0]
        
        tqdm_iterator.set_postfix({"weighted proportion": weighted_in_ci / iteration,
            "unweighted proportion": unweighted_in_ci / iteration})
    
    print(f"\nWeighted proportion: {weighted_in_ci / n_iterations} "
          f"({weighted_too_low / n_iterations} too low, {weighted_too_high / n_iterations} too high), "
          f"average length: {weighted_length_sum / n_iterations}")
    print(f"Unweighted proportion: {unweighted_in_ci / n_iterations} "
          f"({unweighted_too_low / n_iterations} too low, {unweighted_too_high / n_iterations} too high), "
          f"average length: {unweighted_length_sum / n_iterations}")

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
    bert_classifier.eval()

    print("Downloading datasets...")
    mnli_test_ds: datasets.arrow_dataset.Dataset = load_dataset("multi_nli", split="validation_matched").with_format("pt")  # type: ignore
    snli_test_ds: datasets.arrow_dataset.Dataset = load_dataset("snli", split="test").with_format("pt")  # type: ignore

    with torch.no_grad():
        # Preprocess datasets by removing invalid data points and encoding inputs with BERT encoder
        print("Preprocessing datasets...")
        mnli_test_ds = preprocess_nli_dataset(mnli_test_ds)
        snli_test_ds = preprocess_nli_dataset(snli_test_ds)

        def convert_dataset_to_tensor_dict(dataset: datasets.arrow_dataset.Dataset):
            dataset = dataset.with_format("torch", device=device)
            return {column:dataset[column] for column in dataset.format["columns"]}
        mnli_test = convert_dataset_to_tensor_dict(mnli_test_ds)
        snli_test = convert_dataset_to_tensor_dict(snli_test_ds)

        bert_classifier.load_state_dict(torch.load("finetuned_mnli.pt"))
        ci_experiment_repeated(snli_labeled=True, bootstrap_size="unlabeled",
            overlapping_labeled_unlabeled=True,
            results_save_path="results_overlapping-labeled-unlabeled_size-unlabeled_mnli-finetuned.csv",
            seed=0, verbose=True, debug=False)

        ci_experiment_repeated(snli_labeled=True, bootstrap_size="labeled",
            overlapping_labeled_unlabeled=True,
            results_save_path="results_overlapping-labeled-unlabeled_size-labeled_mnli-finetuned.csv",
            seed=0, verbose=True, debug=False)

        bert_classifier.load_state_dict(torch.load("finetuned_snli.pt"))
        ci_experiment_repeated(snli_labeled=True, bootstrap_size="unlabeled",
            overlapping_labeled_unlabeled=True,
            results_save_path="results_overlapping-labeled-unlabeled_size-unlabeled_snli-finetuned.csv",
            seed=0, verbose=True, debug=False)

        ci_experiment_repeated(snli_labeled=True, bootstrap_size="labeled",
            overlapping_labeled_unlabeled=True,
            results_save_path="results_overlapping-labeled-unlabeled_size-labeled_snli-finetuned.csv",
            seed=0, verbose=True, debug=False)
