# Implementation based on https://github.com/jfc43/self-training-ensembles/tree/e5fe8fda3cb5a40dd19d19861f85caddb8372239

import argparse
import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer
from tqdm import tqdm
from common import device
from utils import set_seed, TensorDictDataset, get_batch_eval_metrics
from eval_TE import DataProcessor, convert_examples_to_features, get_hypothesis

tokenizer: BertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def get_dataset(dataset_name: str) -> TensorDictDataset:
    labeling = "single"
    max_seq_length = 128
    sources_info = {
        "yahoo":
            {
                "data_path": "./data/yahoo/test.txt",
                "type2hypothesis": "./data/yahoo/label_names.txt",
                "included_types": set((4, 5, 6, 9))
            },
        "agnews":
            {
                "data_path": "./data/agnews/test.txt",
                "type2hypothesis": "./data/agnews/label_names.txt",
                "included_types": None
            }
        }

    processor = DataProcessor()
    source_info = sources_info[dataset_name]
    examples, _ = processor.get_examples_Yahoo_test(
        filename=source_info["data_path"],
        type2hypothesis=get_hypothesis(source_info["type2hypothesis"], True),
        labeling=labeling, included_types=source_info["included_types"],
        limit=200 if DEBUG else None)
    features = convert_examples_to_features(
        examples=examples, label_list=["entailment", "not_entailment"],
        max_seq_length=max_seq_length, tokenizer=tokenizer)
    dataset = TensorDictDataset(name=dataset_name, data={
        "input_ids": torch.tensor([f.input_ids for f in features], dtype=torch.long),
        "input_mask": torch.tensor([f.input_mask for f in features], dtype=torch.long),
        "segment_ids": torch.tensor([f.segment_ids for f in features], dtype=torch.long),
        "labels": torch.tensor([f.label_id for f in features], dtype=torch.long)})
    if VERBOSITY >= 1:
        print(f"Length of {dataset_name} dataset: {len(dataset)}")
    if DEBUG:
        print(dataset)
    return dataset


def get_output(classifier: BertForSequenceClassification,
               data: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the model's predictions
    """
    classifier.to(device)
    output = torch.empty(len(data["labels"]), device=device)
    predictions = torch.empty(len(data["labels"]), device=device)

    batch_size = 64
    for batch_indices in tqdm(
            torch.split(torch.arange(len(data["labels"]), device=device), batch_size)):
        target = data["labels"][batch_indices]
        batch_output = classifier.forward(
            input_ids=data["input_ids"][batch_indices],
            attention_mask=data["input_mask"][batch_indices],
            token_type_ids=data["segment_ids"][batch_indices])[0]
        assert isinstance(output, torch.Tensor)
        batch_preds = output.argmax(dim=1, keepdim=True).squeeze()
        if DEBUG:
            if target.size() != (len(batch_indices),):
                raise AssertionError(f"target size: {target.size()}")
            if batch_output.size() != (len(batch_indices), classifier.num_labels):
                raise AssertionError(f"output size: {output.size()}")
            if batch_preds.size() != (len(batch_indices),):
                raise AssertionError(f"predictions size: {predictions.size()}, ")
        output[batch_indices] = batch_output
        predictions[batch_indices] = batch_preds

    classifier.cpu()
    return output, predictions


def ensemble_self_training(model2, dataloader_source, pseudo_weight, optimizer):
    model2.train()
    len_dataloader = len(dataloader_source)
    data_source_iter = iter(dataloader_source)

    i = 0
    while i < len_dataloader:
        batch = data_source_iter.next()
        output, _ = get_output(model2, batch)
        eval_metrics = get_batch_eval_metrics(target=batch["labels"], output=output)

        p_s_weight = batch.weight + pseudo_weight * (batch.weight == 0).type(torch.float32)
        # domain-invariant loss
        loss = (p_s_weight * eval_metrics["loss"]).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i += 1


def eval_ensemble(model: BertForSequenceClassification,
                  labeled_ds: TensorDictDataset, unlabeled_ds: TensorDictDataset
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = 128
    nround = 5
    nepoch = 5
    gamma = 0.1
    num_classes = 10

    check_model_paths = []
    for i in range(nepoch):
        check_model_path = "./checkpoints/ensemble_dann_arch_source_models/{:d}/{}/checkpoint.pth".format(i, unlabeled_ds)
        check_model_paths.append(check_model_path)

    pseudo_weight = gamma
    # if unlabeled_ds == "usps":
    #     pseudo_weight = pseudo_weight * 10
    labeled_ds.to(device)
    _, labeled_preds = get_output(model, labeled_ds.data)
    unlabeled_ds.to(device)
    _, unlabeled_preds = get_output(model, unlabeled_ds.data)

    pseudo_train_ds = labeled_ds.get_copy()
    pseudo_train_ds.data["weights"] = torch.ones(len(pseudo_train_ds))
    pseudo_train_dataloader = DataLoader(pseudo_train_ds, batch_size=batch_size,
                                         shuffle=True, drop_last=True)

    for i in range(nround):
        pred_record = torch.zeros((unlabeled_preds.shape[0], num_classes),
                                  dtype=torch.float64, device=device)

        for epoch in range(nepoch):
            model2 = torch.load(check_model_paths[epoch]).to(device)
            optimizer = torch.optim.Adam(model2.parameters(), lr=1e-3)
            ensemble_self_training(model2, pseudo_train_dataloader, pseudo_weight, optimizer)
            _, unlabeled_preds2 = get_output(model2, model2)
            pred_record[torch.arange(unlabeled_preds2.shape[0]), unlabeled_preds2] += 1

        target_test_pseudo_labels = torch.argmax(pred_record, dim=1)
        disagree_record = ~torch.eq(target_test_pseudo_labels.to(device), labeled_preds.to(device))
        disagree_indices = torch.where(disagree_record)[0]

        pseudo_train_ds = TensorDictDataset(name="pseudo train", data={
            col: torch.concat((labeled_ds.data[col], unlabeled_ds.data[col][disagree_indices]), dim=0)
            for col in unlabeled_ds.data
        })
        pseudo_train_ds.data["weights"] = torch.concat((
            torch.ones(len(labeled_ds), dtype=torch.float32, device=device),
            torch.zeros(disagree_indices.shape[0], dtype=torch.float32, device=device)), dim=0)
        pseudo_train_dataloader = DataLoader(pseudo_train_ds, batch_size=batch_size,
                                             shuffle=True, drop_last=True)

    estimated_acc = 1.0 - torch.mean(disagree_record)
    t_test_acc = torch.mean(torch.eq(labeled_preds, labeled_ds.data["labels"]))
    estimated_error = abs(estimated_acc-t_test_acc)

    return t_test_acc, estimated_acc, estimated_error


def main():
    processor = DataProcessor()
    label_list = processor.get_labels()  # [0, 1]
    classifier: BertForSequenceClassification = BertForSequenceClassification \
        .from_pretrained("./model/TE_WikiCate", num_labels=len(label_list))  # type: ignore
    classifier.cpu()
    classifier.eval()

    if VERBOSITY >= 1:
        print("Downloading datasets...")

    yahoo_ds = get_dataset("yahoo")
    agnews_ds = get_dataset("agnews")

    t_test_acc, estimated_acc, estimated_error = eval_ensemble(classifier, yahoo_ds, agnews_ds)
    print(f"test accuracy: {t_test_acc}")
    print(f"estimated accuracy: {estimated_acc}")
    print(f"estimated error: {estimated_error}")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbosity", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)
    VERBOSITY: int = args.verbosity
    DEBUG: bool = args.debug
    if DEBUG:
        VERBOSITY = 2
    main()
