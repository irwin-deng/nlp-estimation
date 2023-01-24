import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Callable
import datasets
import pynndescent


def get_approx_knn_matrix(unlabeled_ds: datasets.arrow_dataset.Dataset,
                          labeled_ds: datasets.arrow_dataset.Dataset,
                          encoder: Callable[[dict[str, NDArray]], ArrayLike],
                          k: int, seed: int = 0, verbose: bool = False
                         ) -> tuple[NDArray[np.int32], NDArray[np.number]]:
    """
    Finds the approximate k nearest neighbors in the labeled dataset for each
    sample in the unlabeled dataset using the PyNNDescent library:
    https://pynndescent.readthedocs.io/

    :param unlabeled_ds: the unlabeled dataset
    :param labeled_ds: the labeled dataset
    :param encoder: a function that takes in a batch of samples and outputs
        each sample's respective encoding as a vector
    :returns:
        - array 'nearest_indices' in which the ith row contains the indices of
        the k closest labeled samples to the ith unlabeled sample
        - array 'distances' in which the entry at indices (i, j) contains the
        distance between unlabeled sample i and labeled sample nearest_indices[i, j]
    """

    def normalized_encoder(batch: dict[str, NDArray]) -> NDArray[np.number]:
        """
        Normalize embedding to be a 1 dimensional ndarray
        """
        encoded = np.array(encoder(batch))
        return encoded.reshape(encoded.shape[0], -1)

    batch_size = 128

    # Encode unlabeled and labeled data
    if verbose:
        print("Encoding unlabeled data as vectors...")
    unlabeled_ds = unlabeled_ds.map(lambda batch:
        {"vector_encoding": normalized_encoder(batch)}, batched = True, batch_size = batch_size)
    if verbose:
        print("Encoding labeled data as vectors...")
    labeled_ds = labeled_ds.map(lambda batch:
        {"vector_encoding": normalized_encoder(batch)}, batched = True, batch_size = batch_size)

    # Index labeled data
    if verbose:
        print("Indexing labeled data...")
    nn_graph = pynndescent.NNDescent(
        labeled_ds["vector_encoding"].cpu(),
        metric = "euclidean",
        n_neighbors = 3 * k,
        random_state = seed,
        verbose = verbose
    )

    # Get nearest neighbours for each sample in the unlabeled dataset
    if verbose:
        print("Finding nearest neighbors...")
    nearest_indices, distances = nn_graph.query(unlabeled_ds["vector_encoding"].cpu(), k)

    return nearest_indices, distances
