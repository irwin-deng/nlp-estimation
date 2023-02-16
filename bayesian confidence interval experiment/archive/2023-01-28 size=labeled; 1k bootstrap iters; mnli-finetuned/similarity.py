import pynndescent
import torch
from tqdm import tqdm
from common import bert_base, device


def bert_cls_vector_batched(dataset: dict[str, torch.Tensor], batch_indices: torch.Tensor) -> torch.Tensor:
    """
    Get the [CLS] embedding from the output of the base BERT model
    """
    batch_size = len(batch_indices)
    input_ids = dataset["input_ids"][batch_indices]
    attention_mask = dataset["attention_mask"][batch_indices]

    outputs = bert_base.forward(input_ids.to(device), attention_mask.to(device), output_hidden_states=True)
    last_layer_outputs = outputs[0]  # [batch_size, tokens, dimension]
    cls_vector = last_layer_outputs[:, 0, :]
    return torch.reshape(cls_vector, (batch_size, -1))


def get_knn_matrix(unlabeled_ds: dict[str, torch.Tensor],
                   labeled_ds: dict[str, torch.Tensor],
                   k: int, verbose: bool = False, debug: bool = False
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Finds the k nearest neighbors in the labeled dataset (based on Euclidean
    distance between base BERT [CLS] vectors) for each sample in the
    unlabeled dataset

    :param unlabeled_ds: the unlabeled dataset
    :param labeled_ds: the labeled dataset
    :returns:
        - 2D tensor 'nearest_indices' in which the ith row contains the indices of
        the k closest labeled samples to the ith unlabeled sample
        - 2D tensor 'distances' in which the entry at indices (i, j) contains the
        distance between unlabeled sample i and labeled sample nearest_indices[i, j]
    """

    if verbose:
        print("Generating kNN matrix...")

    batch_size = 128
    unlabeled_ds_size = len(unlabeled_ds["label"])
    labeled_ds_size = len(labeled_ds["label"])

    # Encode unlabeled and labeled data
    bert_base.to(device)
    if verbose:
        print("Calculating [CLS] vectors of unlabeled dataset...")
    unlabeled_ds["vector_encoding"] = torch.cat([
        bert_cls_vector_batched(unlabeled_ds, batch_indices)
        for batch_indices in tqdm(torch.split(
            torch.arange(unlabeled_ds_size, device=device), batch_size), disable=not verbose)])

    if labeled_ds != unlabeled_ds:
        if verbose:
            print("Calculating [CLS] vectors of labeled dataset...")
        labeled_ds["vector_encoding"] = torch.cat([
            bert_cls_vector_batched(labeled_ds, batch_indices)
            for batch_indices in tqdm(torch.split(
                torch.arange(labeled_ds_size, device=device), batch_size), disable=not verbose)])
    bert_base.cpu()

    # Get Euclidean distance between each sample in the labeled dataset and each sample in the unlabeled dataset
    if verbose: 
        print("Calculating distances...")
    distances = torch.cdist(unlabeled_ds["vector_encoding"].unsqueeze(dim=0),
        labeled_ds["vector_encoding"].unsqueeze(dim=0), p=2).squeeze(dim=0)
    if debug:
        if distances.size() != (unlabeled_ds_size, labeled_ds_size):
            raise AssertionError(f"size of distances matrix {distances.size()} != ({unlabeled_ds_size}, {labeled_ds_size})")

    # Get nearest neighbors
    if verbose: 
        print("Finding nearest neighbors...")
    distances, nearest_indices = torch.topk(distances, k=k, dim=1, largest=False)
    if debug:
        if nearest_indices.size() != (unlabeled_ds_size, k):
            raise AssertionError(f"size of nearest_indices {nearest_indices.size()} != ({unlabeled_ds_size}, {k})")
        if distances.size() != (unlabeled_ds_size, k):
            raise AssertionError(f"size of distances {distances.size()} != ({unlabeled_ds_size}, {k})")
        if not (torch.all(nearest_indices >= 0).item() and torch.all(nearest_indices < labeled_ds_size).item()):
            raise AssertionError(f"index out of bound in\n{nearest_indices}")

    # Remove unneeded columns
    del labeled_ds["vector_encoding"]
    if labeled_ds != unlabeled_ds:
        del unlabeled_ds["vector_encoding"]

    if verbose: 
        print("Done generating kNN matrix")
    return nearest_indices, distances


def get_approx_knn_matrix(unlabeled_ds: dict[str, torch.Tensor],
                          labeled_ds: dict[str, torch.Tensor],
                          k: int, seed: int = 0, verbose: bool = False
                         ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Finds the approximate k nearest neighbors in the labeled dataset (based on
    Euclidean distance between base BERT [CLS] vectors) for each sample in the
    unlabeled dataset using the PyNNDescent library:
    https://pynndescent.readthedocs.io/

    :param unlabeled_ds: the unlabeled dataset
    :param labeled_ds: the labeled dataset
    :returns:
        - 2D tensor 'nearest_indices' in which the ith row contains the indices of
        the k closest labeled samples to the ith unlabeled sample
        - 2D tensor 'distances' in which the entry at indices (i, j) contains the
        distance between unlabeled sample i and labeled sample nearest_indices[i, j]
    """

    if verbose:
        print("Generating kNN matrix...")

    batch_size = 128

    # Encode unlabeled and labeled data
    if verbose:
        print("Calculating [CLS] vectors of unlabeled dataset...")
    unlabeled_ds_size = len(unlabeled_ds["label"])
    unlabeled_ds["vector_encoding"] = torch.cat([
        bert_cls_vector_batched(unlabeled_ds, batch_indices)
        for batch_indices in torch.split(torch.arange(unlabeled_ds_size, device=device), batch_size)])
    if verbose:
        print("Calculating [CLS] vectors of labeled dataset...")
    labeled_ds_size = len(labeled_ds["label"])
    labeled_ds["vector_encoding"] = torch.cat([
        bert_cls_vector_batched(labeled_ds, batch_indices)
        for batch_indices in torch.split(torch.arange(labeled_ds_size, device=device), batch_size)])

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

    # Remove unneeded columns
    del labeled_ds["vector_encoding"]
    del unlabeled_ds["vector_encoding"]

    return (torch.tensor(nearest_indices, dtype=torch.int64, device=device),
        torch.tensor(distances, dtype=torch.float64, device=device))
