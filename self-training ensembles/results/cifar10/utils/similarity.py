import numpy as np
import torch
from tqdm import tqdm
import transformers.modeling_outputs
from transformers import BertModel, BertForSequenceClassification
import torchvision
from torchvision import transforms
from typing import Union

bert_tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')


def get_bert_encoding(batch: list[str]) -> transformers.BatchEncoding:
    encoding = bert_tokenizer.batch_encode_plus(
        [f"[CLS] {review} [SEP]" for review in batch],
        add_special_tokens = True,  # Add [CLS] and [SEP]
        max_length = 128,
        padding = "max_length",
        truncation = True,
        return_attention_mask = True,
        return_tensors = "pt"
    )
    return encoding


def bert_cls_vectors_batched(bert_model: Union[BertModel,BertForSequenceClassification],
        inputs: list[str], batch_size: int,
        debug: bool = False) -> torch.Tensor:
    """
    Get the [CLS] embeddings from the specified bert_model, splitting into batches of size batch_size

    input_ids and attention_mask should be tensors of a common length n in dimension 0.
    The output is a tensor of shape (n, 786)
    """
    n_examples = len(inputs)
    bert_model.cuda()

    def get_cls_vectors(bert_model: Union[BertModel,BertForSequenceClassification],
            batch: list[str]) -> torch.Tensor:
        """
        Get the [CLS] embeddings from the specified bert_model as a single batch
        """
        batch_size = len(batch)
        bert_encoding = get_bert_encoding(batch)

        output: transformers.modeling_outputs.SequenceClassifierOutput = bert_model.forward(
            bert_encoding["input_ids"].cuda(), bert_encoding["attention_mask"].cuda(),
            output_hidden_states=True)  # type: ignore
        hidden_states = output.hidden_states
        assert hidden_states is not None
        last_hidden_state = hidden_states[-1]  # [batch_size, tokens, dimension]
        if debug:
            if last_hidden_state.size(dim=0) != batch_size:
                raise AssertionError(f"last_hidden_state dim 0 size: {last_hidden_state.size(dim=0)}")
        cls_vectors = last_hidden_state[:, 0, :]
        return torch.reshape(cls_vectors, (batch_size, -1))

    encodings = torch.cat([
        get_cls_vectors(bert_model, batch=inputs[batch_start:batch_start+batch_size])
            for batch_start in tqdm(range(0, n_examples, batch_size))])
    bert_model.cpu()
    return encodings


def bert_average_token_batched(bert_model: Union[BertModel,BertForSequenceClassification],
        inputs: list[str], batch_size: int,
        debug: bool = False) -> torch.Tensor:
    """
    Get the average embedding of each token from the output of the specified bert_model,
    splitting into batches of size batch_size

    input_ids and attention_mask should be tensors of a common length n in dimension 0.
    The output is a tensor of shape (n, 786)
    """
    n_examples = len(inputs)
    bert_model.cuda()

    def get_average_token(bert_model: Union[BertModel,BertForSequenceClassification],
            batch: list[str]) -> torch.Tensor:
        """
        Get the average embedding of each token from the output of the specified bert_model
        as a single batch
        """
        batch_size = len(batch)
        bert_encoding = get_bert_encoding(batch)

        output: transformers.modeling_outputs.SequenceClassifierOutput = bert_model.forward(
            bert_encoding["input_ids"].cuda(), bert_encoding["attention_mask"].cuda(),
            output_hidden_states=True)  # type: ignore
        hidden_states = output.hidden_states
        assert hidden_states is not None
        last_hidden_state = hidden_states[-1]  # [batch_size, tokens, dimension]
        if debug:
            if last_hidden_state.size(dim=0) != batch_size:
                raise AssertionError(f"last_hidden_state dim 0 size: {last_hidden_state.size(dim=0)}")
        average_embeddings = torch.stack(
            [last_hidden_state[i, torch.nonzero(bert_encoding["attention_mask"][i]), :].squeeze().mean(dim=0)
                for i in range(batch_size)])
        if debug:
            if average_embeddings.size(dim=0) != batch_size:
                raise AssertionError(f"average_embeddings dim 0: {average_embeddings.size(dim=0)}, "
                    f"n_examples: {n_examples}")
            if average_embeddings.size(dim=1) != 768:
                raise AssertionError(f"average_embeddings dim 1: {average_embeddings.size(dim=1)}")
        return torch.reshape(average_embeddings, (batch_size, -1))

    encodings = torch.cat([
        get_average_token(bert_model, batch=inputs[batch_start:batch_start+batch_size])
            for batch_start in tqdm(range(0, n_examples, batch_size))])
    bert_model.cpu()
    return encodings


def resnet_outputs_batched(resnet_model, inputs: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    Get the output of a resnet_model. For resnet18, the output is a tensor of shape (n, 512)
    """

    n_examples = inputs.shape[0]
    resnet_model.cuda()
    def resnet_forward(resnet_model, batch: torch.Tensor) -> torch.Tensor:
        real_batch_size = batch.shape[0]
        return resnet_model(batch.cuda()).reshape(real_batch_size, -1)

    encodings = torch.cat([
        resnet_forward(resnet_model, inputs[batch_start : batch_start + batch_size])
            for batch_start in range(0, n_examples, batch_size)])
    resnet_model.cpu()
    return encodings


def get_distance_matrix(unlabeled_ds: [list[str],np.ndarray], labeled_ds: Union[list[str],np.ndarray], encoding_type: str,
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

    batch_size = 16
    if isinstance(unlabeled_ds, list):
        unlabeled_ds_size = len(unlabeled_ds)
    elif isinstance(unlabeled_ds, np.ndarray):
        unlabeled_ds_size = unlabeled_ds.shape[0]
        if unlabeled_ds.shape[1:] != (3, 32, 32):
            raise AssertionError(f"found shape {unlabeled_ds.shape}")
    else:
        raise NotImplementedError()
    if isinstance(labeled_ds, list):
        labeled_ds_size = len(labeled_ds)
    elif isinstance(labeled_ds, np.ndarray):
        labeled_ds_size = labeled_ds.shape[0]
        if labeled_ds.shape[1:] != (3, 32, 32):
            raise AssertionError(f"found shape {labeled_ds.shape}")
    else:
        raise NotImplementedError()

    with torch.no_grad():
        # Encode unlabeled and labeled data
        if verbose:
            print("Calculating vectors of unlabeled dataset...")
        if encoding_type == "cls":
            if bert_model is None:
                bert_model = BertForSequenceClassification.from_pretrained(
                    "bert-base-uncased", num_labels = 2, output_attentions = False,
                    output_hidden_states = False)
            unlabeled_vectors = bert_cls_vectors_batched(bert_model=bert_model,
                inputs=unlabeled_ds, batch_size=batch_size, debug=debug)
        elif encoding_type == "avg":
            if bert_model is None:
                bert_model = BertForSequenceClassification.from_pretrained(
                    "bert-base-uncased", num_labels = 2, output_attentions = False,
                    output_hidden_states = False)
            unlabeled_vectors = bert_average_token_batched(bert_model=bert_model,
                inputs=unlabeled_ds, batch_size=batch_size, debug=debug)
        elif encoding_type == "resnet":
            resnet18_model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
            resnet18_model.eval()
            # Normalize to ImageNet distribution
            img_normalizer = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            normalized_images = img_normalizer(torch.from_numpy(unlabeled_ds).float())
            unlabeled_vectors = resnet_outputs_batched(resnet_model=resnet18_model,
                inputs=normalized_images, batch_size=batch_size)
        else:
            raise AssertionError

        if labeled_ds is not unlabeled_ds:
            if verbose:
                print("Calculating vectors of labeled dataset...")
            if encoding_type == "cls":
                assert bert_model is not None
                labeled_vectors = bert_cls_vectors_batched(bert_model=bert_model,
                    inputs=labeled_ds, batch_size=batch_size, debug=debug)
            elif encoding_type == "avg":
                assert bert_model is not None
                labeled_vectors = bert_average_token_batched(bert_model=bert_model,
                    inputs=labeled_ds, batch_size=batch_size, debug=debug)
            elif encoding_type == "resnet":
                normalized_images = img_normalizer(torch.from_numpy(labeled_ds).float())
                labeled_vectors = resnet_outputs_batched(resnet_model=resnet18_model,
                    inputs=normalized_images, batch_size=batch_size)
            else:
                raise AssertionError

        # Get Euclidean distance between each sample in the labeled dataset and each sample in the unlabeled dataset
        if verbose:
            print("Calculating distances...")
        distances = torch.cdist(unlabeled_vectors.unsqueeze(dim=0),
            labeled_vectors.unsqueeze(dim=0), p=2).squeeze(dim=0)
        if debug:
            if distances.size() != (unlabeled_ds_size, labeled_ds_size):
                raise AssertionError(f"size of distances matrix {distances.size()} != ({unlabeled_ds_size}, {labeled_ds_size})")

    return distances
