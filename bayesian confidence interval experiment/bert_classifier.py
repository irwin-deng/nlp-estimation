import torch
import math
import numpy as np
import torch
import datasets
from datasets import load_dataset
import transformers
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm
from utils import get_batch_eval_metrics


class BertClassifier(torch.nn.Module):
    def __init__(self, n_labels: int = 3):
        super(BertClassifier, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=n_labels)
    
    def forward(self, input_ids, attention_mask):
        return self.bert.forward(input_ids=input_ids,
            attention_mask=attention_mask, return_dict=False)[0]

bert_tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')


def get_bert_encoding(batch: dict[str, torch.tensor]) -> transformers.BatchEncoding:
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


def preprocess_nli_dataset(dataset: datasets.arrow_dataset.Dataset,
                           batch_size: int = 256) -> datasets.arrow_dataset.Dataset:
    """
    Preprocess datasets by removing invalid data points and encoding inputs with BERT encoder
    """

    # Remove invalid data points
    dataset = dataset.filter(lambda example: example["label"] != -1)

    # Encode inputs with BERT encoder
    dataset = dataset.map(lambda batch: get_bert_encoding(batch), batched=True, batch_size=batch_size)

    # Remove unused columns
    cols_to_remove = dataset.column_names
    for column_to_keep in ["input_ids", "token_type_ids", "attention_mask", "label"]:
        cols_to_remove.remove(column_to_keep)
    dataset = dataset.remove_columns(cols_to_remove)

    dataset.set_format("torch")
    return dataset


def finetune_nli(dataset_name: str, split: str = "train", seed: int = 0,
                 save_checkpoints: bool = False, verbose: bool = False) -> None:
    """
    Finetunes a BERT classifier on the specified dataset, specified by its name
    on HuggingFace
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    if verbose:
        print(f"Downloading {dataset_name} dataset...")
    dataset: datasets.arrow_dataset.Dataset = load_dataset(dataset_name, split=split).with_format("torch")  # type: ignore

    batch_size = 64

    # Preprocess datasets by removing invalid data points and encoding inputs with BERT encoder
    if verbose:
        print("Preprocessing dataset...")
    dataset = preprocess_nli_dataset(dataset)
    print(f"Preprocessed dataset size: {len(dataset)}")

    if verbose:
        print("Downloading base BERT model...")
    bert_classifier = BertClassifier(n_labels=3)

    optimizer = torch.optim.AdamW(bert_classifier.parameters(), lr=2e-5)
    loss_func = torch.nn.CrossEntropyLoss()
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    n_train_samples = len(dataset)
    n_batches = math.ceil(n_train_samples / batch_size)
    n_train_epochs = 3
    n_train_steps = n_train_epochs * n_batches
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = n_train_steps // 10,
        num_training_steps = n_train_steps
    )

    if verbose:
        print(f"Fine-tuning BERT Classifier with batch size {batch_size}...")
    bert_classifier.to(device)
    bert_classifier.train()
    for epoch in range(1, n_train_epochs + 1):
        epoch_loss_sum, epoch_n_correct, epoch_n_samples = 0.0, 0.0, 0
        with tqdm(train_loader, unit="batch") as tqdm_epoch:
            tqdm_epoch.set_description(f"Epoch {epoch} / {n_train_epochs}")
            for batch in tqdm_epoch:
                target = batch["label"].to(device)

                optimizer.zero_grad()
                output = bert_classifier.forward(batch["input_ids"].to(device),
                    batch["attention_mask"].to(device))

                metrics = get_batch_eval_metrics(target, output, loss_func)
                loss = metrics["loss"]
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_n_samples += metrics["batch_size"]
                epoch_loss_sum += loss.item() * metrics["batch_size"]
                epoch_n_correct += metrics["accuracy"] * metrics["batch_size"]
                tqdm_epoch.set_postfix({"train loss": epoch_loss_sum / epoch_n_samples,
                    "train accuracy": epoch_n_correct / epoch_n_samples})
            tqdm_epoch.refresh()
        # Save model checkpoint
        if save_checkpoints and epoch < n_train_epochs:
            if verbose:
                print(f"Saving BERT Classifier checkpoint to bert_classifier_finetuned_{dataset_name}_checkpoint_{epoch}.pt")
            torch.save(bert_classifier.state_dict(), f"bert_classifier_finetuned_{dataset_name}_checkpoint_{epoch}.pt")
    # Save model
    if verbose:
        print(f"Saving BERT Classifier to bert_classifier_finetuned_{dataset_name}.pt")
    torch.save(bert_classifier.state_dict(), f"bert_classifier_finetuned_{dataset_name}.pt")


if __name__ == '__main__':
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print(f"Using CPU")

    finetune_nli(dataset_name="snli", split="train", seed=0, save_checkpoints=False, verbose=True)
