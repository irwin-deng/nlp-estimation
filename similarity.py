import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Callable
import torch
import datasets
import pynndescent


from rouge_score import rouge_scorer as rouge_scorer_module
rouge_scorer = rouge_scorer_module.RougeScorer(['rougeL'], use_stemmer=True)

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def get_knn(sample: str, train: list[str],
            similarity: Callable[[str, str], float], k: int) -> np.ndarray:
    """
    Finds the (exact) k nearest neighbors in the train dataset for the given
    sample
    """
    distances = [similarity(sample, train[i]) for i in range(len(train))]
    indices = np.argpartition(distances, k)
    return indices[:k]


def get_knn_matrix(train: list[str], test: list[str],
                   similarity: Callable[[str, str], float],
                   k: int) -> dict[int, np.ndarray]:
    """
    Finds the (exact) k nearest neighbors in the train dataset for each sample
    in the test dataset
    """
    n_test = len(test)
    knn_dict = {}
    for indx in range(n_test):
        knn_dict[indx] = get_knn(test[indx], train, similarity, k)
    return knn_dict


def get_approx_knn_matrix(train: datasets.arrow_dataset.Dataset,
                          test: datasets.arrow_dataset.Dataset,
                          encoder: Callable[[dict[str, torch.Tensor]], ArrayLike],
                          k: int, seed: int = 0, verbose: bool = False
                         ) -> tuple[NDArray[np.int32], NDArray[np.number]]:
    """
    Finds the approximate k nearest neighbors in the train dataset for each
    sample in the test dataset using the PyNNDescent library:
    https://pynndescent.readthedocs.io/

    :param train: a list of training samples
    :param test: a list of test samples
    :param encoder: a function that converts a string into an embedding
    :returns:
        - array 'nearest_indices' in which the ith row contains the indices of
        the k closest train samples to the ith test sample
        - array 'distances' in which the entry at indices (i, j) contains the
        distance between test sample i and train sample nearest_indices[i, j]
    """
    def normalized_encoder(sample: dict[str, torch.Tensor]) -> NDArray[np.number]:
        """
        Normalize embedding to be a 1 dimensional ndarray
        """
        return np.array(encoder(sample)).flatten()

    # Encode train and test data
    if verbose:
        print("Encoding data with BERT encoder...")
    train.map(lambda sample: {"encoding": normalized_encoder(sample)})
    test.map(lambda sample: {"encoding": normalized_encoder(sample)})

    print(train.select(range(5))["encoding"])

    # Index training data
    if verbose:
        print("Indexing train data...")
    nn_graph = pynndescent.NNDescent(
        train["encoding"],
        metric = "euclidean",
        n_neighbors = 3 * k,
        random_state = seed,
        verbose = verbose
    )

    # Get nearest neighbours
    if verbose:
        print("Finding nearest neighbors...")
    nearest_indices, distances = nn_graph.query(test["encoding"], k)

    print(nearest_indices[:5])

    return nearest_indices, distances


def distance_rouge_score(text1: str, text2: str) -> float:
    return 1 - rouge_scorer.score(text1, text2)['rougeL']


def distance_bleu_score(text1: str, text2: str) -> float:
    return 1 - sentence_bleu([text1], text2,
        smoothing_function=SmoothingFunction().method1)  # type: ignore