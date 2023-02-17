import torch
from tqdm import tqdm
import transformers.modeling_outputs
from transformers import BertModel, BertForSequenceClassification
from typing import Union
from common import device
from sklearn.feature_extraction.text import TfidfVectorizer


def bert_cls_vectors_batched(bert_model: Union[BertModel,BertForSequenceClassification],
        input_ids: torch.Tensor, attention_mask: torch.Tensor, batch_size: int,
        debug: bool = False) -> torch.Tensor:
    """
    Get the [CLS] embeddings from the specified bert_model, splitting into batches of size batch_size

    input_ids and attention_mask should be tensors of a common length n in dimension 0.
    The output is a tensor of shape (n, 786)
    """
    n_examples = input_ids.size(dim=0)

    def get_cls_vectors(bert_model: Union[BertModel,BertForSequenceClassification],
            input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Get the [CLS] embeddings from the specified bert_model as a single batch
        """
        if debug:
            if input_ids.size(dim=0) != attention_mask.size(dim=0):
                raise AssertionError(f"input_ids dim 0: {input_ids.size(dim=0)}, "
                    f"attention_mask dim 0: {attention_mask.size(dim=0)}")
        batch_size = input_ids.size(dim=0)

        output: transformers.modeling_outputs.SequenceClassifierOutput = bert_model.forward(
            input_ids.to(device), attention_mask.to(device), output_hidden_states=True)  # type: ignore
        hidden_states = output.hidden_states
        assert hidden_states is not None
        last_hidden_state = hidden_states[-1] # [batch_size, tokens, dimension]
        if debug:
            if last_hidden_state.size(dim=0) != batch_size:
                raise AssertionError(f"last_hidden_state dim 0 size: {last_hidden_state.size(dim=0)}")
        cls_vectors = last_hidden_state[:, 0, :]
        return torch.reshape(cls_vectors, (batch_size, -1))

    return torch.cat([
        get_cls_vectors(bert_model, input_ids=input_ids[batch_indices],
            attention_mask=attention_mask[batch_indices])
            for batch_indices in tqdm(torch.split(
                torch.arange(n_examples, device=device), batch_size))])


def bert_average_token_batched(bert_model: Union[BertModel,BertForSequenceClassification],
        input_ids: torch.Tensor, attention_mask: torch.Tensor, batch_size: int,
        debug: bool = False) -> torch.Tensor:
    """
    Get the average embedding of each token from the output of the specified bert_model,
    splitting into batches of size batch_size

    input_ids and attention_mask should be tensors of a common length n in dimension 0.
    The output is a tensor of shape (n, 786)
    """
    n_examples = input_ids.size(dim=0)

    def get_average_token(bert_model: Union[BertModel,BertForSequenceClassification],
            input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Get the average embedding of each token from the output of the specified bert_model
        as a single batch
        """
        if debug:
            if input_ids.size(dim=0) != attention_mask.size(dim=0):
                raise AssertionError(f"input_ids dim 0: {input_ids.size(dim=0)}, "
                    f"attention_mask dim 0: {attention_mask.size(dim=0)}")
        batch_size = input_ids.size(dim=0)

        output: transformers.modeling_outputs.SequenceClassifierOutput = bert_model.forward(
            input_ids.to(device), attention_mask.to(device), output_hidden_states=True)  # type: ignore
        hidden_states = output.hidden_states
        assert hidden_states is not None
        last_hidden_state = hidden_states[-1] # [batch_size, tokens, dimension]
        if debug:
            if last_hidden_state.size(dim=0) != batch_size:
                raise AssertionError(f"last_hidden_state dim 0 size: {last_hidden_state.size(dim=0)}")
        average_embeddings = torch.stack([last_hidden_state[i,torch.nonzero(attention_mask[i]),:].squeeze().mean(dim=0)
            for i in range(batch_size)])
        if debug:
            if average_embeddings.size(dim=0) != batch_size:
                raise AssertionError(f"average_embeddings dim 0: {average_embeddings.size(dim=0)}, "
                    f"n_examples: {n_examples}")
            if average_embeddings.size(dim=1) != 768:
                raise AssertionError(f"average_embeddings dim 1: {average_embeddings.size(dim=1)}")
        return torch.reshape(average_embeddings, (batch_size, -1))

    return torch.cat([
        get_average_token(bert_model, input_ids=input_ids[batch_indices],
            attention_mask=attention_mask[batch_indices])
            for batch_indices in tqdm(torch.split(
                torch.arange(n_examples, device=device), batch_size))])


def tf_idf_vectorizer(premise: list[str], hypothesis: list[str], corpus: list[str]) -> torch.Tensor:
    """
    Get the tf-idf vector for each premise+hypothesis pair

    premise and hypothesis should be lists of strings of some common length n.
    The output is a tensor of shape (n, m), where m is the number of unique words in the corpus
    """
    assert len(premise) == len(hypothesis)
    vectorizer = TfidfVectorizer(stop_words="english")
    vectorizer.fit(corpus)
    premise_vectors = torch.tensor(vectorizer.transform(premise).toarray(), device=device)  # type: ignore
    hypothesis_vectors = torch.tensor(vectorizer.transform(hypothesis).toarray(), device=device)  # type: ignore
    return torch.add(premise_vectors, hypothesis_vectors)


def get_distance_matrix(unlabeled_ds: dict[str, torch.Tensor],
                        labeled_ds: dict[str, torch.Tensor], encoding_type: str,
                        bert_model: Union[BertModel,BertForSequenceClassification,None] = None,
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

    batch_size = 64
    unlabeled_ds_size = len(unlabeled_ds["label"])
    labeled_ds_size = len(labeled_ds["label"])

    # Encode unlabeled and labeled data
    if bert_model is not None:
        bert_model.to(device)
    if verbose:
        print("Calculating [CLS] vectors of unlabeled dataset...")
    if encoding_type == "cls":
        assert bert_model is not None
        unlabeled_ds["vector_encoding"] = bert_cls_vectors_batched(bert_model=bert_model,
            input_ids=unlabeled_ds["input_ids"], attention_mask=unlabeled_ds["attention_mask"],
            batch_size=batch_size, debug=debug)
    elif encoding_type == "avg":
        assert bert_model is not None
        unlabeled_ds["vector_encoding"] = bert_average_token_batched(bert_model=bert_model,
            input_ids=unlabeled_ds["input_ids"], attention_mask=unlabeled_ds["attention_mask"],
            batch_size=batch_size, debug=debug)
    else:
        raise AssertionError

    if labeled_ds is not unlabeled_ds:
        if verbose:
            print("Calculating [CLS] vectors of labeled dataset...")
        if encoding_type == "cls":
            assert bert_model is not None
            labeled_ds["vector_encoding"] = bert_cls_vectors_batched(bert_model=bert_model,
                input_ids=labeled_ds["input_ids"], attention_mask=labeled_ds["attention_mask"],
                batch_size=batch_size, debug=debug)
        elif encoding_type == "avg":
            assert bert_model is not None
            labeled_ds["vector_encoding"] = bert_average_token_batched(bert_model=bert_model,
                input_ids=labeled_ds["input_ids"], attention_mask=labeled_ds["attention_mask"],
                batch_size=batch_size, debug=debug)
        else:
            raise AssertionError
    bert_model.cpu()

    # Get Euclidean distance between each sample in the labeled dataset and each sample in the unlabeled dataset
    if verbose: 
        print("Calculating distances...")
    distances = torch.cdist(unlabeled_ds["vector_encoding"].unsqueeze(dim=0),
        labeled_ds["vector_encoding"].unsqueeze(dim=0), p=2).squeeze(dim=0)
    if debug:
        if distances.size() != (unlabeled_ds_size, labeled_ds_size):
            raise AssertionError(f"size of distances matrix {distances.size()} != ({unlabeled_ds_size}, {labeled_ds_size})")

    return distances
