import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Callable
import torch
import datasets
import pynndescent


from rouge_score import rouge_scorer as rouge_scorer_module
rouge_scorer = rouge_scorer_module.RougeScorer(['rougeL'], use_stemmer=True)

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def get_knn(sample: str, unlabeled: list[str],
            similarity: Callable[[str, str], float], k: int) -> np.ndarray:
    """
    Finds the (exact) k nearest neighbors in the unlabeled dataset for the given
    sample
    """
    distances = [similarity(sample, unlabeled[i]) for i in range(len(unlabeled))]
    indices = np.argpartition(distances, k)
    return indices[:k]


def get_knn_matrix(unlabeled: list[str], labeled: list[str],
                   similarity: Callable[[str, str], float],
                   k: int) -> dict[int, np.ndarray]:
    """
    Finds the (exact) k nearest neighbors in the unlabeled dataset for each sample
    in the labeled dataset
    """
    n_labeled = len(labeled)
    knn_dict = {}
    for indx in range(n_labeled):
        knn_dict[indx] = get_knn(labeled[indx], unlabeled, similarity, k)
    return knn_dict


def get_approx_knn_matrix(unlabeled_ds: datasets.arrow_dataset.Dataset,
                          labeled_ds: datasets.arrow_dataset.Dataset,
                          encoder: Callable[[dict[str, torch.Tensor]], ArrayLike],
                          k: int, seed: int = 0, verbose: bool = False
                         ) -> tuple[NDArray[np.int32], NDArray[np.number]]:
    """
    Finds the approximate k nearest neighbors in the labeled dataset for each
    sample in the unlabeled dataset using the PyNNDescent library:
    https://pynndescent.readthedocs.io/

    :param unlabeled_ds: the unlabeled dataset
    :param labeled_ds: the labeled dataset
    :param encoder: a function that converts a string into an embedding
    :returns:
        - array 'nearest_indices' in which the ith row contains the indices of
        the k closest labeled samples to the ith unlabeled sample
        - array 'distances' in which the entry at indices (i, j) contains the
        distance between unlabeled sample i and labeled sample nearest_indices[i, j]
    """
    def normalized_encoder(sample: dict[str, torch.Tensor]) -> NDArray[np.number]:
        """
        Normalize embedding to be a 1 dimensional ndarray
        """
        return np.array(encoder(sample)).flatten()

    # Encode unlabeled and labeled data
    if verbose:
        print("Encoding data with BERT encoder...")
    unlabeled_ds.map(lambda sample: {"encoding": normalized_encoder(sample)})
    labeled_ds.map(lambda sample: {"encoding": normalized_encoder(sample)})

    print(unlabeled_ds.select(range(5))["encoding"])

    # Index labeled data
    if verbose:
        print("Indexing labeled data...")
    nn_graph = pynndescent.NNDescent(
        labeled_ds["encoding"],
        metric = "euclidean",
        n_neighbors = 3 * k,
        random_state = seed,
        verbose = verbose
    )

    # Get nearest neighbours for each sample in the unlabeled dataset
    if verbose:
        print("Finding nearest neighbors...")
    nearest_indices, distances = nn_graph.query(unlabeled_ds["encoding"], k)

    print(nearest_indices[:5])

    return nearest_indices, distances


def distance_rouge_score(text1: str, text2: str) -> float:
    return 1 - rouge_scorer.score(text1, text2)['rougeL']


def distance_bleu_score(text1: str, text2: str) -> float:
    return 1 - sentence_bleu([text1], text2,
        smoothing_function=SmoothingFunction().method1)  # type: ignore
