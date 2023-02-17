import torch
from tqdm import tqdm
import transformers.modeling_outputs
from transformers import BertModel, BertForSequenceClassification
from typing import Union
from sklearn.feature_extraction.text import TfidfVectorizer
from common import bert_base, device


def bert_cls_vectors_batched(bert_model: Union[BertModel,BertForSequenceClassification],
        input_ids: torch.Tensor, attention_mask: torch.Tensor, batch_size: int,
        debug: bool = False) -> torch.Tensor:
    """
    Get the [CLS] embedding from the output of the base BERT model
    """
    n_examples = input_ids.size(dim=0)

    def get_cls_vectors(bert_model: Union[BertModel,BertForSequenceClassification],
            input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
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
    Get the average embedding of each token from the output of the base BERT model
    """
    n_examples = input_ids.size(dim=0)

    def get_average_token(bert_model: Union[BertModel,BertForSequenceClassification],
            input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
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


def tf_idf_vectorizer(premise: list[str], hypothesis: list[str], corpus: list[str],
        debug: bool = False) -> torch.Tensor:
    """
    Get the tf-idf vector for each premise+hypothesis pair
    """
    assert len(premise) == len(hypothesis)
    vectorizer = TfidfVectorizer(stop_words="english")
    vectorizer.fit(corpus)
    premise_vectors = torch.tensor(vectorizer.transform(premise).toarray(), device=device)  # type: ignore
    hypothesis_vectors = torch.tensor(vectorizer.transform(hypothesis).toarray(), device=device)  # type: ignore
    return torch.add(premise_vectors, hypothesis_vectors)


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

    batch_size = 64
    unlabeled_ds_size = len(unlabeled_ds["label"])
    labeled_ds_size = len(labeled_ds["label"])

    # Encode unlabeled and labeled data
    bert_base.to(device)
    if verbose:
        print("Calculating [CLS] vectors of unlabeled dataset...")
    unlabeled_ds["vector_encoding"] = bert_cls_vectors_batched(bert_model=bert_base,
        input_ids=unlabeled_ds["input_ids"], attention_mask=unlabeled_ds["attention_mask"],
        batch_size=batch_size, debug=debug)

    if labeled_ds != unlabeled_ds:
        if verbose:
            print("Calculating [CLS] vectors of labeled dataset...")
        labeled_ds["vector_encoding"] = bert_cls_vectors_batched(bert_model=bert_base,
        input_ids=labeled_ds["input_ids"], attention_mask=labeled_ds["attention_mask"],
        batch_size=batch_size, debug=debug)
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
