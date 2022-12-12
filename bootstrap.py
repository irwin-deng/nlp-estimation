from typing import Any
import math
import numpy as np
from numpy.typing import NDArray
import torch
import datasets
from datasets import load_dataset
from similarity import get_approx_knn_matrix
import transformers
from tqdm import tqdm
from bert_model import BertClassifier

torch.cuda.empty_cache()
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA device {torch.cuda.get_device_name(device)}")
else:
    device = torch.device("cpu")
    print(f"Using CPU")

bert_tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')
bert_base = transformers.AutoModel.from_pretrained('bert-base-uncased').to(device)
bert_classifier = BertClassifier(n_labels=3).to(device)

mnli_ds_name = "multi_nli"
snli_ds_name = "snli"

def get_bert_encoding(batch: datasets.arrow_dataset.Batch) -> transformers.BatchEncoding:
    encoding = bert_tokenizer.batch_encode_plus(
        [f"[CLS] {premise} [SEP] {hypothesis} [SEP]"
            for premise, hypothesis in zip(batch["premise"], batch["hypothesis"])],
        add_special_tokens = True,  # Add [CLS] and [SEP]
        max_length = 128,
        padding = "max_length",
        truncation = True,
        return_attention_mask = True,
        return_tensors = "pt"
    )
    return encoding


def bert_cls_vector_batched(batch: datasets.arrow_dataset.Batch) -> NDArray[np.int32]:
    """
    Get the [CLS] embedding from the output of the base BERT model
    """
    encoding = get_bert_encoding(batch).to(device)

    outputs = bert_base.forward(encoding["input_ids"], encoding["attention_mask"], output_hidden_states=True)
    last_layer_outputs = outputs[0]  # [batch_size, tokens, dimension]
    cls_vector = last_layer_outputs[:, 0, :]
    return cls_vector.cpu().detach().numpy()


def bert_cls_vector(sample: dict[str, torch.Tensor]) -> NDArray[np.int32]:
    """
    Get the [CLS] embedding from the output of the base BERT model
    """
    encoding = bert_tokenizer.encode(
        text = f"[CLS] {sample['premise']} [SEP] {sample['hypothesis']} [SEP]",
        add_special_tokens = True,  # Add [CLS] and [SEP]
        max_length = 128,
        padding = "max_length",
        return_attention_mask = True,
        return_tensors = "pt"
    )

    outputs = bert_base.forward(encoding, output_hidden_states=True)
    last_layer_outputs = outputs[0]  # [batch_size, tokens, dimension]
    cls_vector = last_layer_outputs[0, 0, :]
    return cls_vector.detach().numpy()


def bayesian_sampler(labeled_ds: datasets.arrow_dataset.Dataset,
        unlabeled_indices: NDArray[np.int32], nearest_indices: NDArray[np.int32],
        probabilities: NDArray[np.number]) -> dict[str, torch.tensor]:
    """
    Generate samples for a batch of unlabeled samples by performing weighted
    sampling from the labeled dataset. For each unlabeled sample provided,
    a labeled sample is selected.

    :param labeled_ds: the labeled dataset
    :param unlabeled_indices: an array consisting of the subset of the unlabeled
        dataset's indices for which to draw samples for
    :param nearest_indices: The ith row contains the indices of the k closest
        unlabeled samples to the ith labeled sample
    :param probabilities: The entry at indices (i, j) contains the probability
        of sampling the jth index in the nearest_indices array given the
        unlabeled sample with index i in the unlabeled dataset
    :returns: a dict in which the keys are the column names,
        and the values are the values of that column for the batch
    """

    column_names = labeled_ds.format["columns"]

    # For each sample in the unlabeled batch, get a sample from the labeled batch
    def get_labeled_sample(unlabeled_indx: np.int32) -> np.int32:
        labeled_indx = np.random.choice(nearest_indices[unlabeled_indx],
            p=probabilities[unlabeled_indx])
        return labeled_indx
    labeled_indices = np.vectorize(get_labeled_sample)(unlabeled_indices)

    # Convert to a dict of tensors
    sampled_batch = {}
    for column in column_names:
        sampled_batch[column] = torch.stack([labeled_ds[labeled_indx.item()][column]
            for labeled_indx in labeled_indices])
    return sampled_batch


def random_sampler(labeled_ds: datasets.arrow_dataset.Dataset,
        batch_size: int) -> dict[str, torch.tensor]:
    """
    Generate a batch of size batch_size consisting of randomly sampled
    observations (with replacement) from the labeled dataset

    :param labeled_ds: the labeled dataset
    :param batch_size: the number of samples to get for this batch
    :returns: a dict in which the keys are the column names,
        and the values are the values of that column for the batch
    """

    column_names = labeled_ds.format["columns"]
    labeled_ds_size = len(labeled_ds)

    labeled_indices = np.random.choice(labeled_ds_size, size=batch_size)

    # Convert to a dict of tensors
    sampled_batch = {}
    for column in column_names:
        sampled_batch[column] = torch.stack([labeled_ds[labeled_indx.item()][column]
            for labeled_indx in labeled_indices])
    return sampled_batch


def nlp_experiment(seed: int, bayesian: bool = True, verbose: bool = False):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if verbose:
        print("Downloading datasets...")
    mnli_train: datasets.arrow_dataset.Dataset = load_dataset(mnli_ds_name, split="train").with_format("np")  # type: ignore
    snli_train: datasets.arrow_dataset.Dataset = load_dataset(snli_ds_name, split="train").with_format("np")  # type: ignore
    snli_test: datasets.arrow_dataset.Dataset = load_dataset(snli_ds_name, split="validation").with_format("np")  # type: ignore

    if bayesian:
        # Get kNN matrix
        if verbose:
            print("Calculating kNN matrix...")
        nearest_indices, distances = get_approx_knn_matrix(
            unlabeled_ds = snli_train,
            labeled_ds = mnli_train,
            encoder = bert_cls_vector_batched,
            k = 10,
            seed = seed,
            verbose = verbose
        )

        # Convert distances to probabilities
        def get_probabilities(distances: NDArray[np.number]) -> NDArray[np.number]:
            """
            Convert a list of distances into probabilities, weighted by inverse of
            distance
            """
            inv_distances = distances ** -1
            normalized = inv_distances / sum(inv_distances)
            return normalized
        probabilities = np.apply_along_axis(get_probabilities, axis=1, arr=distances)

    batch_size = 64

    # Encode datasets
    if verbose:
        print("Encoding datasets with BERT encoder...")
    mnli_train = mnli_train.map(lambda batch: get_bert_encoding(batch), batched=True, batch_size=batch_size)
    mnli_train.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])
    snli_train = snli_train.map(lambda batch: get_bert_encoding(batch), batched=True, batch_size=batch_size)
    snli_train.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])
    snli_test = snli_test.map(lambda batch: get_bert_encoding(batch), batched=True, batch_size=batch_size)
    snli_test.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])

    optimizer = torch.optim.AdamW(bert_classifier.parameters())
    loss_func = torch.nn.CrossEntropyLoss()
    validation_loader = torch.utils.data.DataLoader(snli_test, batch_size = batch_size)

    # Returns a dict of evaluation metrics for a classification model given a
    # target tensor and the model's output tensor
    def get_eval_metrics(target: torch.tensor, output: torch.tensor) -> dict[str, Any]:
        """
        Returns a dict of evaluation metrics for a classification model given a
        target tensor and the model's output tensor
        """
        loss = loss_func(output, target)
        predictions = output.argmax(dim=1, keepdim=True).squeeze()
        n_correct = (predictions == target).sum().item()
        accuracy = n_correct / batch_size
        return {"loss": loss, "accuracy": accuracy}

    n_train_samples = len(snli_train)
    train_indices = np.arange(n_train_samples)
    n_batches = math.ceil(n_train_samples / batch_size)
    n_epochs = 5
    if verbose:
        print("Training BERT Classifier...")
    for epoch in range(n_epochs):
        np.random.shuffle(train_indices)

        # Allow model to be trained
        bert_classifier.train()

        # Iterate over all training batches
        with tqdm(np.array_split(train_indices, n_batches), unit="batch") as tqdm_epoch:
            tqdm_epoch.set_description(f"Epoch {epoch}")
            loss_sum, accuracy_sum, num_samples = 0.0, 0.0, 0

            for batch_indices in tqdm_epoch:
                if bayesian:
                    batch = bayesian_sampler(labeled_ds = mnli_train,
                        unlabeled_indices = batch_indices, nearest_indices = nearest_indices,
                        probabilities = probabilities)
                else:
                    batch = random_sampler(labeled_ds = mnli_train, batch_size = batch_size)
                target = batch["label"].to(device)

                optimizer.zero_grad()
                output = bert_classifier.forward(batch["input_ids"].to(device),
                    batch["attention_mask"].to(device))

                metrics = get_eval_metrics(target, output)
                loss = metrics["loss"]
                loss.backward()
                optimizer.step()

                batch_size = len(batch_indices)
                num_samples += batch_size
                loss_sum += loss.item() * batch_size
                accuracy_sum += metrics["accuracy"] * batch_size
                tqdm_epoch.set_postfix({"train_loss": loss_sum / num_samples,
                    "train_accuracy": accuracy_sum / num_samples})
            tqdm_epoch.refresh()
    
        # Calculate validation loss + accuracy
        bert_classifier.train(False)
        for batch in validation_loader:
            loss_sum, accuracy_sum, num_samples = 0.0, 0.0, 0

            target = batch["label"].to(device)
            output = bert_classifier.forward(batch["input_ids"].to(device),
                batch["attention_mask"].to(device))
        
            batch_size = len(batch_indices)
            num_samples += batch_size
            loss_sum += metrics["loss"].item() * batch_size
            accuracy_sum += metrics["accuracy"] * batch_size
        print(f"\tvalidation_loss: {loss_sum / num_samples}, "
              f"validation_accuracy: {accuracy_sum / num_samples}")
    
    # Save model
    if verbose:
        print("Saving BERT Classifier to bert_classifier.pt")
    torch.save(bert_classifier.state_dict(), "bert_classifier.pt")


if __name__ == '__main__':
    nlp_experiment(seed = 0, bayesian = True, verbose = True)
