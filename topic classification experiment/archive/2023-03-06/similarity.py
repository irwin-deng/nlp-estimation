import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from numpy.typing import ArrayLike
from common import device
from utils import TensorDictDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

similarity_model = SentenceTransformer("cardiffnlp/twitter-roberta-base-dec2021-tweet-topic-multi-all")
similarity_model.cpu()
similarity_model.eval()


def get_distance_matrix(unlabeled_ds: TensorDictDataset,
                        labeled_ds: TensorDictDataset, encoding_type: str,
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

    unlabeled_ds_size = len(unlabeled_ds)
    labeled_ds_size = len(labeled_ds)

    # Encode unlabeled and labeled data
    similarity_model.to(device)
    if verbose:
        print("Calculating vector encodings of unlabeled dataset...")
    if encoding_type == "sbert":
        assert isinstance(unlabeled_ds["premises"], list)
        vector_encodings = similarity_model.encode(unlabeled_ds["premises"],
            show_progress_bar=verbose, convert_to_tensor=True, device=device)
        assert isinstance(vector_encodings, torch.Tensor)
        unlabeled_ds["vector_encodings"] = vector_encodings.to(torch.float32)
    else:
        raise AssertionError

    if labeled_ds is not unlabeled_ds:
        if verbose:
            print("Calculating vector encodings of labeled dataset...")
        if encoding_type == "sbert":
            assert isinstance(labeled_ds["premises"], list)
            vector_encodings = similarity_model.encode(labeled_ds["premises"],
                show_progress_bar=verbose, convert_to_tensor=True, device=device)
            assert isinstance(vector_encodings, torch.Tensor)
            labeled_ds["vector_encodings"] = vector_encodings.to(torch.float32)
        else:
            raise AssertionError
    similarity_model.cpu()

    # Get Euclidean distance between each sample in the labeled dataset and each sample in the unlabeled dataset
    if verbose: 
        print("Calculating distances...")
    distances = torch.cdist(unlabeled_ds["vector_encodings"].unsqueeze(dim=0),
        labeled_ds["vector_encodings"].unsqueeze(dim=0), p=2).squeeze(dim=0)
    if debug:
        if distances.size() != (unlabeled_ds_size, labeled_ds_size):
            raise AssertionError(f"size of distances matrix {distances.size()} != ({unlabeled_ds_size}, {labeled_ds_size})")

    if verbose:
        # Plot PCA of the two datasets
        if unlabeled_ds is not labeled_ds:
            plot_pca(torch.cat((unlabeled_ds["vector_encodings"], labeled_ds["vector_encodings"]), dim=0),
            torch.cat((torch.zeros_like(unlabeled_ds["labels"]), torch.ones_like(labeled_ds["labels"])), dim=0),
            title="Distribution of encodings of the two datasets",
            label_names=[unlabeled_ds.name, labeled_ds.name])
        # Plot PCA of labels
        datasets_to_plot = [unlabeled_ds]
        if labeled_ds is not unlabeled_ds:
            datasets_to_plot.append(unlabeled_ds)
        for dataset in datasets_to_plot:
            unique_labels = list(set(dataset["correct_hypotheses"]))
            label_map = {label: indx for (indx, label) in enumerate(unique_labels)}
            plot_pca(dataset["vector_encodings"], [label_map[label] for label in dataset["correct_hypotheses"]],
            title=f"Distribution of labels in dataset {dataset.name}",
            label_names=unique_labels, cmap="tab10")
        # Plot PCA of correct/incorrect labels
        assert "is_correct" in labeled_ds
        if labeled_ds is unlabeled_ds:
            plot_pca(labeled_ds["vector_encodings"], labeled_ds["is_correct"],
            title=f"Distribution of correct or incorrect predictions in dataset {labeled_ds.name}",
            label_names=["incorrect", "correct"], cmap="Set1")
        else:
            assert "is_correct" in unlabeled_ds
            plot_pca(torch.cat((unlabeled_ds["vector_encodings"], labeled_ds["vector_encodings"]), dim=0),
                torch.cat((unlabeled_ds["is_correct"], 2+labeled_ds["is_correct"]), dim=0),
                title=f"Distribution of correct or incorrect predictions in datasets {labeled_ds.name} and {unlabeled_ds.name}",
                label_names=[f"{unlabeled_ds.name} incorrect", f"{unlabeled_ds.name} correct",
                            f"{labeled_ds.name} incorrect", f"{labeled_ds.name} correct"], cmap="tab20")

    unlabeled_ds["vector_encodings"] = unlabeled_ds["vector_encodings"].cpu()
    labeled_ds["vector_encodings"] = labeled_ds["vector_encodings"].cpu()
    return distances

# Visualize dataset using PCA
def plot_pca(vector_encodings: ArrayLike, labels: ArrayLike, title: str, label_names: list[str], cmap: str = "coolwarm"):
    # Compute first 2 principle components
    pca = PCA(n_components=2)
    if isinstance(vector_encodings, torch.Tensor):
        vector_encodings = vector_encodings.cpu().numpy()
    pca_transformed = pca.fit_transform(vector_encodings)
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # Plot
    plt.clf()
    plot = plt.scatter(pca_transformed[:,0], pca_transformed[:,1], c=labels, s=0.5, cmap=cmap)
    plt.legend(handles=plot.legend_elements()[0], labels=label_names)
    plt.title(title)
    plt.show(block=False)
    plt.savefig(f"{title}.png")

def torch_cdist(x1: torch.Tensor, x2: torch.Tensor, p: float, batch_size: int = 512) -> torch.Tensor:
    """
    Same as torch.cdist, but doesn't use as much memory by splitting into batches
    of batch_size * batch_size examples
    """
    assert x1.shape[0] == x2.shape[0]
    distances = torch.empty((x1.shape[0], x1.shape[1], x2.shape[1]), device=device)
    for indx1 in range(0, x1.shape[1], batch_size):
        range1 = torch.arange(indx1, min(indx1+batch_size, x1.shape[1]), device=device)
        for indx2 in range(0, x2.shape[1], batch_size):
            range2 = torch.arange(indx2, min(indx2+batch_size, x2.shape[1]), device=device)
            distances[:, range1, range2] = torch.cdist(x1[:,range1,:], x2[:,range2,:], p)
    return distances
