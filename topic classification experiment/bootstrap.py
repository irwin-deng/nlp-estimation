import sys
import random
import numpy as np
import torch
import csv
from similarity import get_distance_matrix
from tqdm import tqdm
from typing import Optional
from transformers import BertForSequenceClassification, BertTokenizer
import gc
from utils import TensorDictDataset, shuffle_tensor
from common import device
from eval_TE import (DataProcessor, convert_examples_to_features, get_hypothesis)

gc.collect()
torch.cuda.empty_cache()

def compute_correctly_classified(classifier: BertForSequenceClassification, dataset: TensorDictDataset, debug: bool = False):
    """
    Add a new boolean field 'is_correct' to the dataset indicating whether the classifier
    predicted the label correctly
    """
    dataset["is_correct"] = torch.empty(len(dataset), dtype=torch.bool, device=device)
    classifier.to(device)
    features_to_move = ["labels", "input_ids", "input_mask", "segment_ids"]
    for feature in features_to_move:
        dataset[feature] = dataset[feature].to(device)

    batch_size = 64
    for batch_indices in tqdm(torch.split(torch.arange(len(dataset), device=device), batch_size)):
        target = dataset["labels"][batch_indices]
        output = classifier.forward(input_ids=dataset["input_ids"][batch_indices],
            attention_mask=dataset["input_mask"][batch_indices],
            token_type_ids=dataset["segment_ids"][batch_indices])[0]
        assert isinstance(output, torch.Tensor)
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

    classifier.cpu()
    for feature in features_to_move:
        dataset[feature] = dataset[feature].cpu()


def get_accuracy(dataset: TensorDictDataset, dataset_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
    """
    Compute the accuracy of the classifier on the provided dataset

    :param dataset_indices: the subset of the dataset to compute the accuracy
        for. May include duplicates, in which each row of dataset_indices is weighted
        equally.
    :returns: a singleton float tensor containing the accuracy
    """

    if dataset_indices is None:
        dataset_indices = torch.arange(len(dataset), device=device)

    if "is_correct" not in dataset.keys():
        raise AssertionError("Must call compute_correctly_classified() first to "
            "precompute predicted samples")

    return torch.mean(dataset["is_correct"][dataset_indices], dtype=torch.float)



def get_confidence_interval(labeled_ds: TensorDictDataset,
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



def ci_experiment_repeated(classifier: BertForSequenceClassification,
        labeled_ds: TensorDictDataset, unlabeled_ds: TensorDictDataset,
        unlabeled_prop = 0.01, overlapping_labeled_unlabeled: bool = False,
        seed: int = 0, results_save_path: Optional[str] = None,
        verbose: bool = False, debug: bool = False):
    """
    Compute the proportion of confidence intervals that contain the true value
    of the classifier's accuracy on the unlabeled dataset, comparing weighted
    vs unweighted sampling.

    :param labeled_ds: The dataset to draw labeled samples from
    :param unlabeled_ds: The dataset to draw unlabeled samples from. If this is
        the same object as labeled_ds, then this dataset will be partitioned
        into a labeled and an unlabeled set each bootstrap iteration.
    :param unlabeled_prop: The proportion of unlabeled_ds to use each bootstrap
        iteration
    :param overlapping_labeled_unlabeled: whether to also include the unlabeled
        examples in the labeled dataset (this just serves as a sanity check)
    """
    if overlapping_labeled_unlabeled:
        assert unlabeled_ds is labeled_ds
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_unlabeled = int(unlabeled_prop * len(unlabeled_ds))  # Number of samples from unlabeled_ds to use as the unlabeled dataset each iteration
    if unlabeled_ds is labeled_ds:
        print(f"\nRunning experiment partitioning {unlabeled_ds.name} dataset into unlabeled and labeled")
        n_labeled = len(unlabeled_ds) - n_unlabeled
    else:
        print(f"\nRunning experiment with {labeled_ds.name} as labeled and {unlabeled_ds.name} as unlabeled")
        n_labeled = len(labeled_ds)
    if overlapping_labeled_unlabeled:
        print("Unlabeled data is a subset of labeled data")
        n_labeled = min(n_unlabeled * 10, len(labeled_ds))
    else:
        print("Unlabeled data and labeled data are disjoint")
    if verbose:
        print(f"labeled subset size: {n_labeled}/{len(labeled_ds)}, unlabeled subset size: {n_unlabeled}/{len(unlabeled_ds)}")

    # Precompute whether the classifier can correctly classify each element of the dataset(s)
    if verbose:
        print("\nPrecomputing predicted values on datasets...")
    compute_correctly_classified(classifier, labeled_ds, debug=debug)
    if unlabeled_ds is not labeled_ds:
        compute_correctly_classified(classifier, unlabeled_ds, debug=debug)

    n_iterations = 10000 if not debug else 10  # Number of different confidence intervals to generate
    n_bootstrap_samples = n_labeled

    # Calculate weights matrix
    distances = get_distance_matrix(unlabeled_ds=unlabeled_ds,
        labeled_ds=labeled_ds, encoding_type="sbert", verbose=verbose, debug=debug)
    if verbose:
        print("pairwise distances matrix: ")
        print(distances)
    epsilon = 1e-8  # to make sure that no weight is infinite
    distances += epsilon
    weights = distances ** -1
    # exp = torch.exp(weights - torch.amax(weights, 1, keepdim=True))  # Subtracting by row-wise max prevents overflow
    # weights = exp / torch.sum(exp, 1, keepdim=True)
    # weights -= torch.amax(weights, 1, keepdim=True)
    # weights.exp_()
    if verbose:
        print("full weights matrix: ")
        print(weights)

    accuracies, weighted_cis, unweighted_cis = [], [], []  # Accuracies; confidence intervals produced by weighted, unweighte sampling
    weighted_in_ci, unweighted_in_ci = 0, 0  # The number of trials in which weighted and unweighted sampling produced a CI containing the true accuracy
    weighted_too_high, unweighted_too_high, weighted_too_low, unweighted_too_low = 0, 0, 0, 0  # The number of confidence intervals that were higher/lower than the true accuracy
    weighted_length_sum, unweighted_length_sum = 0, 0  # The sum of the lengths of the intervals

    if verbose:
        print("Generating confidence intervals...")
    all_unlabeled_indices = torch.arange(len(unlabeled_ds), device=device)
    tqdm_iterator = tqdm(range(1, n_iterations + 1), disable = not verbose)
    for iteration in tqdm_iterator:
        tqdm_iterator.set_description(f"Iteration {iteration} / {n_iterations}")

        # Get indices of unlabeled subset
        all_unlabeled_indices = shuffle_tensor(all_unlabeled_indices)
        unlabeled_indices = all_unlabeled_indices[torch.arange(n_unlabeled, device=device)]
        # Get indices of labeled subset
        if labeled_ds is unlabeled_ds and overlapping_labeled_unlabeled:
            labeled_indices = all_unlabeled_indices[torch.arange(n_labeled, device=device)]
        elif labeled_ds is unlabeled_ds and not overlapping_labeled_unlabeled:
            labeled_indices = all_unlabeled_indices[torch.arange(n_unlabeled, len(unlabeled_ds), device=device)]
        else:
            labeled_indices = torch.arange(len(labeled_ds), device=device)

        # Get true accuracy
        accuracy = get_accuracy(dataset=unlabeled_ds, dataset_indices=unlabeled_indices)
        accuracies.append(accuracy)

        # Calculate weights given the subset of the labeled dataset
        weights_subset = weights[unlabeled_indices].clone().detach()
        if labeled_ds is unlabeled_ds and overlapping_labeled_unlabeled:
            weights_subset[:, ~labeled_indices] = 0
        if labeled_ds is unlabeled_ds and not overlapping_labeled_unlabeled:
            weights_subset[:, unlabeled_indices] = 0
        if debug:
            if weights_subset.size() != (len(unlabeled_indices), len(labeled_ds)):
                raise AssertionError(f"weights_subset size: {weights_subset.size()})")
        # Normalize sum of weights in each row to 1
        weights_subset /= weights_subset.sum(dim=1, keepdim=True)
        # Combine into a single weight vector, where the entry at index i is the probability of sampling the ith labeled example
        weights_subset = torch.sum(weights_subset, dim=0)
        weights_subset /= weights_subset.sum()  # Normalize sum to 1
        if debug:
            if weights_subset.size() != (len(labeled_ds),):
                raise AssertionError(f"weights_subset size: {weights_subset.size()})")
        if verbose and iteration <= 2:
            print(f"iteration {iteration} weights (sorted descending): ")
            print(weights_subset.sort(descending=True)[0])
            print(f"iteration {iteration} weights (ordered by labeled index): ")
            print(weights_subset)

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
    debug = False
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_printoptions(threshold=10)

    print("Loading model...")
    processor = DataProcessor()
    label_list = processor.get_labels()  # [0, 1]
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    classifier: BertForSequenceClassification = BertForSequenceClassification.from_pretrained("./model/TE_WikiCate", num_labels=num_labels)  # type: ignore
    classifier.cpu()
    classifier.eval()

    print("Downloading datasets...")
    labeling = "single"
    max_seq_length = 128
    batch_size = 32
    def get_dataset(name: str, data_path: str, label_path: str, included_types: Optional[set[int]] = None) -> TensorDictDataset:
        labeling = "single"
        type2hypothesis = get_hypothesis(label_path, True)
        examples, _ = processor.get_examples_Yahoo_test(data_path, type2hypothesis, labeling,
                                                                 included_types=included_types, limit=200 if debug else None)
        features = convert_examples_to_features(examples, ["entailment", "not_entailment"], max_seq_length, tokenizer)
        dataset = TensorDictDataset(name, {"input_ids": torch.tensor([f.input_ids for f in features], dtype=torch.long),
                                    "input_mask": torch.tensor([f.input_mask for f in features], dtype=torch.long),
                                    "segment_ids": torch.tensor([f.segment_ids for f in features], dtype=torch.long),
                                    "premises": [example.text_a for example in examples],
                                    "hypotheses": [example.text_b for example in examples],
                                    "correct_hypotheses": [example.text_b_true for example in examples],
                                    "labels": torch.tensor([f.label_id for f in features], dtype=torch.long)})
        return dataset
    yahoo_dataset = get_dataset("yahoo", "./data/yahoo/test.txt", "./data/yahoo/label_names.txt", included_types=set((4,5,6,9)))
    print(f"Length of yahoo dataset: {len(yahoo_dataset)}")
    if debug:
        print(yahoo_dataset["hypotheses"][0:9])
        print(yahoo_dataset["labels"][0:9])
        for feature, values in yahoo_dataset.items():
            try:
                print(f"'{feature}' size: {sys.getsizeof(values.storage())} bytes")
            except:
                pass
    agnews_dataset = get_dataset("agnews", "./data/agnews/test.txt", "./data/agnews/label_names.txt")
    print(f"Length of agnews dataset: {len(agnews_dataset)}")
    if debug:
        print(agnews_dataset["hypotheses"][0:9])
        print(agnews_dataset["labels"][0:9])
        for feature, values in agnews_dataset.items():
            try:
                print(f"'{feature}' size: {sys.getsizeof(values.storage())} bytes")
            except:
                pass

    with torch.no_grad():
        ci_experiment_repeated(classifier=classifier, labeled_ds=yahoo_dataset,
            unlabeled_ds=yahoo_dataset, overlapping_labeled_unlabeled=True,
            results_save_path="results_overlapping-yahoo-yahoo.csv",
            seed=0, verbose=True, debug=debug)

        ci_experiment_repeated(classifier=classifier, labeled_ds=agnews_dataset,
            unlabeled_ds=agnews_dataset, overlapping_labeled_unlabeled=True,
            results_save_path="results_overlapping-agnews-agnews.csv",
            seed=0, verbose=True, debug=debug)

        ci_experiment_repeated(classifier=classifier, labeled_ds=yahoo_dataset,
            unlabeled_ds=yahoo_dataset,
            results_save_path="results_L-yahoo-U-yahoo.csv",
            seed=0, verbose=True, debug=debug)

        ci_experiment_repeated(classifier=classifier, labeled_ds=agnews_dataset,
            unlabeled_ds=agnews_dataset,
            results_save_path="results_L-agnews-U-agnews.csv",
            seed=0, verbose=True, debug=debug)

        ci_experiment_repeated(classifier=classifier, labeled_ds=agnews_dataset,
            unlabeled_ds=yahoo_dataset,
            results_save_path="results_L-agnews-U-yahoo.csv",
            seed=0, verbose=True, debug=debug)

        ci_experiment_repeated(classifier=classifier, labeled_ds=yahoo_dataset,
            unlabeled_ds=agnews_dataset,
            results_save_path="results_L-yahoo-U-agnews.csv",
            seed=0, verbose=True, debug=debug)
