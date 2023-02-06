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


def get_distance_matrix(unlabeled_ds: dict[str, torch.Tensor],
                        labeled_ds: dict[str, torch.Tensor],
                        verbose: bool = False, debug: bool = False
                       ) -> torch.Tensor:
    """
    Finds the distances between each example in the labeled dataset and the
    unlabeled dataset

    :param unlabeled_ds: the unlabeled dataset
    :param labeled_ds: the labeled dataset
    :returns: 2D tensor 'distances' in which the entry at indices (i, j) contains the
        distance between unlabeled example i and labeled example j
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

    return distances
